"""
Layouts (extracted from the .ttgir):
  - Q_lora    [64, 512]  -> #blocked3 (sizePerThread=[1,8], threadsPerWarp=[1,64], warpsPerCTA=[4,1], order=[1,0])
                            stored to #shared (vec=8, perPhase=1, maxPhase=16, order=[1,0])
                            loaded as DotOperand opIdx=0 of #mma (k_width=8)
  - Q_rope    [64, 64]   -> #blocked2 (sizePerThread=[1,8], threadsPerWarp=[8,8], warpsPerCTA=[4,1], order=[1,0])
                            stored to #shared1 (vec=8, perPhase=2, maxPhase=8, order=[1,0])
                            loaded as DotOperand opIdx=0 of #mma (k_width=8)
  - K_lora_T  [512, 16]  -> #blocked1 (sizePerThread=[8,1], threadsPerWarp=[64,1], warpsPerCTA=[1,4], order=[0,1])
                            stored to #shared3 (vec=8, perPhase=1, maxPhase=16, order=[0,1])
                            loaded as DotOperand opIdx=1 of #mma (k_width=8)
                            also transposed and used as V_lora [16, 512] via #shared4
                            (vec=4, perPhase=1, maxPhase=16, order=[1,0]) → opIdx=1 of #mma1 (k_width=4)
  - K_rope_T  [64, 16]   -> #blocked  (sizePerThread=[2,1], threadsPerWarp=[32,2], warpsPerCTA=[1,4], order=[0,1])
                            stored to #shared2 (vec=8, perPhase=2, maxPhase=8, order=[0,1]) — double-buffered
                            loaded as DotOperand opIdx=1 of #mma (k_width=8)
  - S         [64, 16]   -> #mma  (instr_shape=[16,16,16], warps_per_cta=[4,1], transposed=True)
  - acc       [64, 512]  -> #mma1 (instr_shape=[16,16,16], warps_per_cta=[4,1], transposed=True)

MFMA instruction (from amdgcn): v_mfma_f32_16x16x16_bf16 for both dots.

Pipeline:
  Prologue:  Q_lora + Q_rope → shared via async DMA (group A).
             K_lora tile 0 + K_rope tile 0 → shared via async DMA (group B, double-buffered).
             wait_group(1) → Q in shared; load Q dot operands once.
  Loop body for tile t (0..N-2):
    - prefetch next-tile topk_pos
    - K_lora[t+1] + K_rope[t+1] → shared via async DMA (one new group)
    - wait_group(1)  → K[t] in shared (older group retired)
    - read K_lora[cur_buf], V_lora[cur_buf] (permute view), K_rope[cur_buf]
    - S = Q_lora @ K_lora_T + Q_rope @ K_rope_T
    - softmax update + acc += P @ V_lora
  Epilogue: wait_group(0) → process last tile, write O and LSE.
"""

import math
import os

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# =====================================================================
# Layouts (constexpr factories so they're sharable across kernels)
# =====================================================================

# MFMA layouts
# S = [BLOCK_H=64, TILE_K=16] bf16 inputs => fp32 accumulator
# acc = [BLOCK_H=64, D_V=512] fp32 accumulator
def _mfma_s():
    # Triton's #mma uses instrShape=[16,16,32] with kWidth=8 — that is two
    # k=16 MFMA instructions fused. We use the smaller native 16x16x16 here.
    return gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16],
        transposed=True, warps_per_cta=[4, 1],
    )


def _mfma_acc():
    return gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16],
        transposed=True, warps_per_cta=[4, 1],
    )


@gluon.jit
def _sparse_mla_fwd_gl_kernel(
    Q_ptr,          # [total_tokens, num_heads, D_QK] bf16
    KV_ptr,         # [total_tokens, 1, D_QK]         bf16
    TopK_ptr,       # [total_tokens, TOPK]            int32
    O_ptr,          # [total_tokens, num_heads, D_V]  bf16
    LSE_ptr,        # [total_tokens, num_heads]       fp32
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    TOPK: gl.constexpr,
    BLOCK_H: gl.constexpr,
    TILE_K: gl.constexpr,
    D_V: gl.constexpr,
    D_ROPE: gl.constexpr,
):
    # ---------- constexpr layouts ----------
    mfma_s: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16],
        transposed=True, warps_per_cta=[4, 1],
    )
    mfma_acc: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16],
        transposed=True, warps_per_cta=[4, 1],
    )

    # Blocked layouts for global loads.
    #
    # blk_qlora: this layout is used as the source of an async
    # buffer_load_to_shared for Q_lora. The CDNA4 direct-to-LDS lowering
    # (BufferLoadToLocalOpConversion → canLoadDirectToLDS) requires the
    # warp's per-tile coalesced span along the contiguous dim to *equal*
    # the tensor's inner dim (`vec * threadsPerWarp_along_contig == innerDim`),
    # otherwise `canCoalesceWriteIntoSharedMemory` fails and the op is
    # left unlowered (manifests as `unrealized_conversion_cast` translation
    # failure). Concretely we need `size_per_thread[1] * threads_per_warp[1]
    # == D_V`. We keep `size_per_thread[1] = 8` so each LDS write is 128
    # bits (the only width supported by ds_load_tr at bf16 besides 32).
    # Note this layout is also reused for the O write epilogue (sync
    # buffer_store), which is not subject to the same constraint.
    _qlora_tpw_k: gl.constexpr = min(64, D_V // 8)
    _qlora_tpw_m: gl.constexpr = 64 // _qlora_tpw_k
    blk_qlora: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[_qlora_tpw_m, _qlora_tpw_k],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    blk_qrope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    # blk_klora: source of an async buffer_load_to_shared for K_lora (order=[0,1] →
    # dim 0 = D_V is contiguous). Same CDNA4 coalesced-write constraint as blk_qlora:
    # `size_per_thread[0] * threads_per_warp[0] == D_V`. Keep size_per_thread[0]=8 so
    # each LDS write is 128 bits.
    _klora_tpw_m: gl.constexpr = min(64, D_V // 8)
    _klora_tpw_n: gl.constexpr = 64 // _klora_tpw_m
    blk_klora: gl.constexpr = gl.BlockedLayout(   # [D_V, TILE_K]
        size_per_thread=[8, 1],
        threads_per_warp=[_klora_tpw_m, _klora_tpw_n],
        warps_per_cta=[1, 4], order=[0, 1],
    )
    blk_krope: gl.constexpr = gl.BlockedLayout(   # [D_ROPE, TILE_K] = [64, 16]
        size_per_thread=[2, 1], threads_per_warp=[32, 2],
        warps_per_cta=[1, 4], order=[0, 1],
    )
    blk_topk: gl.constexpr = gl.BlockedLayout(    # [TILE_K] int32
        size_per_thread=[1], threads_per_warp=[64],
        warps_per_cta=[4], order=[0],
    )
    blk_lse: gl.constexpr = gl.BlockedLayout(     # [BLOCK_H] fp32
        size_per_thread=[1], threads_per_warp=[64],
        warps_per_cta=[4], order=[0],
    )

    # Shared layouts
    sh_qlora: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0],
    )
    sh_qrope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=8, order=[1, 0],
    )
    sh_klora: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1],
    )
    sh_krope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=8, order=[0, 1],
    )

    # Dot operand layouts
    dot_qlora_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_s, k_width=8)
    dot_qrope_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_s, k_width=8)
    dot_klora_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_s, k_width=8)
    dot_krope_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_s, k_width=8)
    dot_p_a: gl.constexpr     = gl.DotOperandLayout(operand_index=0, parent=mfma_acc, k_width=4)
    dot_v_b: gl.constexpr     = gl.DotOperandLayout(operand_index=1, parent=mfma_acc, k_width=4)

    # ---------- program ids ----------
    token_idx = gl.program_id(axis=0)
    hg_idx = gl.program_id(axis=1)
    hg_offset = hg_idx * BLOCK_H

    # ---------- offsets for Q ----------
    # Q_lora  [BLOCK_H, D_V]
    offs_h_qlora = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_qlora = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_qlora = offs_h_qlora < num_heads

    q_base = token_idx.to(tl.int64) * stride_q_t
    q_offs_lora = (
        q_base
        + offs_h_qlora[:, None].to(tl.int64) * stride_q_h
        + offs_v_qlora[None, :].to(tl.int64)
    )
    q_mask_lora = mask_h_qlora[:, None]

    # Q_rope  [BLOCK_H, D_ROPE]
    offs_h_qrope = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qrope))
    offs_r_qrope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(0, blk_qrope))
    mask_h_qrope = offs_h_qrope < num_heads

    q_offs_rope = (
        q_base
        + offs_h_qrope[:, None].to(tl.int64) * stride_q_h
        + (D_V + offs_r_qrope[None, :]).to(tl.int64)
    )
    q_mask_rope = mask_h_qrope[:, None]

    # Stage Q_lora and Q_rope directly to shared via async DMA. Committed as one
    # group so it can overlap with the K_rope tile-0 issue below. We keep the
    # dot-operand-layout views (Q_lora_dot, Q_rope_dot) in registers for the
    # entire loop; those are loaded after the K_rope tile-0 commit + wait_group.
    smem_qlora = gl.allocate_shared_memory(Q_ptr.dtype.element_ty, [BLOCK_H, D_V],  layout=sh_qlora)
    smem_qrope = gl.allocate_shared_memory(Q_ptr.dtype.element_ty, [BLOCK_H, D_ROPE], layout=sh_qrope)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_qlora,
        ptr=Q_ptr,
        offsets=q_offs_lora.to(tl.int32),
        mask=q_mask_lora,
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=smem_qrope,
        ptr=Q_ptr,
        offsets=q_offs_rope.to(tl.int32),
        mask=q_mask_rope,
    )
    gl.amd.cdna4.async_copy.commit_group()

    # ---------- topk and KV offsets ----------
    NUM_TILES: gl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx.to(tl.int64) * stride_topk_t

    # offs_tile in three layouts (sliced from each of the three loaders)
    offs_tile_klora = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_klora))
    offs_tile_krope = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, blk_krope))
    offs_tile_mma = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, mfma_s))
    offs_tile_topk = gl.arange(0, TILE_K, layout=blk_topk)

    offs_v_klora = gl.arange(0, D_V, layout=gl.SliceLayout(1, blk_klora))
    offs_r_krope = gl.arange(0, D_ROPE, layout=gl.SliceLayout(1, blk_krope))

    # First topk_pos load (tile 0)
    tpos_offs0 = topk_base.to(tl.int32) + offs_tile_topk
    topk_pos_reg = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr, offsets=tpos_offs0,
        mask=offs_tile_topk < TOPK, other=-1,
    )

    # ---------- shared mem allocations for the K loop ----------
    # K_rope is double-buffered exactly like Triton's ttgir.
    smem_krope = gl.allocate_shared_memory(
        KV_ptr.dtype.element_ty, [2, D_ROPE, TILE_K], layout=sh_krope,
    )
    # K_lora is double-buffered (parallels K_rope). We async-DMA each tile into
    # smem_klora.index(buf), then read the current buffer twice: once as opIdx=1
    # of mfma_s (S dot) and once permuted to [TILE_K, D_V] as opIdx=1 of mfma_acc
    # (V_lora dot). The permute is a memdesc view — no data movement.
    smem_klora = gl.allocate_shared_memory(
        KV_ptr.dtype.element_ty, [2, D_V, TILE_K], layout=sh_klora,
    )

    # ---------- accumulators ----------
    m_i = gl.full([BLOCK_H], float("-inf"), dtype=gl.float32,
                  layout=gl.SliceLayout(1, mfma_s))
    l_i = gl.full([BLOCK_H], 0.0, dtype=gl.float32,
                  layout=gl.SliceLayout(1, mfma_s))
    acc = gl.zeros([BLOCK_H, D_V], dtype=gl.float32, layout=mfma_acc)

    # ---------- tile-0 prefetch (prologue) ----------
    # Load K_lora and K_rope for tile 0.
    valid0_klora = (topk_pos_reg != -1) & (offs_tile_topk < TOPK)
    # Cast valid masks into the layouts of the K loaders.
    # We'll just rebuild per-loader masks using the per-loader offs_tile and a
    # per-loader topk_pos (loaded with the same blocked layout).
    # To avoid extra TopK loads, we re-express by recomputing topk_pos per
    # blocked layout: use convert_layout where possible, or reload from memory.
    # Reload per-layout is the simplest and matches what the Triton IR actually does.
    topk_pos_klora = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + offs_tile_klora,
        mask=offs_tile_klora < TOPK, other=-1,
    )
    topk_pos_krope = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + offs_tile_krope,
        mask=offs_tile_krope < TOPK, other=-1,
    )
    topk_pos_mma = gl.amd.cdna4.buffer_load(
        ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + offs_tile_mma,
        mask=offs_tile_mma < TOPK, other=-1,
    )

    valid_klora = (topk_pos_klora != -1)  # tile_start=0 -> offs_tile<TOPK already true
    valid_krope = (topk_pos_krope != -1)
    valid_mma = (topk_pos_mma != -1)

    safe_klora = gl.where(valid_klora, topk_pos_klora, 0)
    safe_krope = gl.where(valid_krope, topk_pos_krope, 0)

    # K_lora async DMA into smem_klora[0]
    klora_offs = (
        safe_klora[None, :].to(tl.int64) * stride_kv_t
        + offs_v_klora[:, None].to(tl.int64)
    )
    klora_smem0 = smem_klora.index(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=klora_smem0,
        ptr=KV_ptr,
        offsets=klora_offs.to(tl.int32),
        mask=valid_klora[None, :],
    )

    # K_rope async DMA into smem_krope[0] — same group as K_lora.
    krope_offs = (
        safe_krope[None, :].to(tl.int64) * stride_kv_t
        + (D_V + offs_r_krope[:, None]).to(tl.int64)
    )
    krope_smem0 = smem_krope.index(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=krope_smem0,
        ptr=KV_ptr,
        offsets=krope_offs.to(tl.int32),
        mask=valid_krope[None, :],
    )
    gl.amd.cdna4.async_copy.commit_group()

    # Two groups in flight: [Q (older), K_rope tile 0 (newer)]. Wait until at
    # most 1 remains → Q is in shared. Convert Q to dot-operand layout once and
    # keep in regs for the entire loop. K_rope tile 0 stays in flight; the
    # loop's wait_group(1) will retire it on iter 0.
    gl.amd.cdna4.async_copy.wait_group(1)
    Q_lora_dot = smem_qlora.load(dot_qlora_a)
    Q_rope_dot = smem_qrope.load(dot_qrope_a)

    # ---------- main loop: prefetch t+1, compute t ----------
    cur_buf = 0
    for t in range(NUM_TILES - 1):
        # prefetch next-tile topk indices (per-layout)
        next_offs_topk = (t + 1) * TILE_K + offs_tile_topk
        next_offs_klora = (t + 1) * TILE_K + offs_tile_klora
        next_offs_krope = (t + 1) * TILE_K + offs_tile_krope
        next_offs_mma = (t + 1) * TILE_K + offs_tile_mma

        topk_pos_klora_next = gl.amd.cdna4.buffer_load(
            ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + next_offs_klora,
            mask=next_offs_klora < TOPK, other=-1,
        )
        topk_pos_krope_next = gl.amd.cdna4.buffer_load(
            ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + next_offs_krope,
            mask=next_offs_krope < TOPK, other=-1,
        )
        topk_pos_mma_next = gl.amd.cdna4.buffer_load(
            ptr=TopK_ptr, offsets=topk_base.to(tl.int32) + next_offs_mma,
            mask=next_offs_mma < TOPK, other=-1,
        )

        valid_klora_next = (next_offs_klora < TOPK) & (topk_pos_klora_next != -1)
        valid_krope_next = (next_offs_krope < TOPK) & (topk_pos_krope_next != -1)
        valid_mma_next = (next_offs_mma < TOPK) & (topk_pos_mma_next != -1)

        safe_klora_next = gl.where(valid_klora_next, topk_pos_klora_next, 0)
        safe_krope_next = gl.where(valid_krope_next, topk_pos_krope_next, 0)

        # K_lora_next + K_rope_next async DMA, both into the next buffer slot,
        # committed as a single group so the loop's wait_group(1) retires both at once.
        next_buf = 1 - cur_buf

        klora_offs_next = (
            safe_klora_next[None, :].to(tl.int64) * stride_kv_t
            + offs_v_klora[:, None].to(tl.int64)
        )
        klora_smem_next = smem_klora.index(next_buf)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=klora_smem_next,
            ptr=KV_ptr,
            offsets=klora_offs_next.to(tl.int32),
            mask=valid_klora_next[None, :],
        )

        krope_offs_next = (
            safe_krope_next[None, :].to(tl.int64) * stride_kv_t
            + (D_V + offs_r_krope[:, None]).to(tl.int64)
        )
        krope_smem_next = smem_krope.index(next_buf)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=krope_smem_next,
            ptr=KV_ptr,
            offsets=krope_offs_next.to(tl.int32),
            mask=valid_krope_next[None, :],
        )
        gl.amd.cdna4.async_copy.commit_group()

        # Wait for the *current* K_rope tile (the older async commit; one outstanding
        # is the just-issued next prefetch).
        gl.amd.cdna4.async_copy.wait_group(1)

        # ---------- compute current tile ----------
        # Read K_lora from current shared buffer in two views: dot-operand and
        # transposed (V_lora). The permute is a memdesc view — no data movement.
        klora_smem_cur = smem_klora.index(cur_buf)
        K_lora_T_dot = klora_smem_cur.load(dot_klora_b)              # opIdx=1 of mfma_s
        V_lora_dot = klora_smem_cur.permute([1, 0]).load(dot_v_b)    # opIdx=1 of mfma_acc

        # Load K_rope from smem (current buffer).
        krope_smem_cur = smem_krope.index(cur_buf)
        K_rope_T_dot = krope_smem_cur.load(dot_krope_b)

        # S = Q_lora @ K_lora_T + Q_rope @ K_rope_T
        S_zero = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s)
        S = gl.amd.cdna4.mfma(Q_lora_dot, K_lora_T_dot, S_zero)
        S = gl.amd.cdna4.mfma(Q_rope_dot, K_rope_T_dot, S)
        S = S * scale

        # Mask invalid columns/rows to -inf
        offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
        mask_h_mma = offs_h_mma < num_heads
        valid_mask = valid_mma[None, :] & mask_h_mma[:, None]
        S = gl.where(valid_mask, S, float("-inf"))

        # Online softmax
        m_j = gl.max(S, axis=1)
        m_new = gl.maximum(m_i, m_j)
        m_new = gl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = gl.exp(m_i - m_new)
        P = gl.exp(S - m_new[:, None])
        l_new = alpha * l_i + gl.sum(P, axis=1)

        # acc = acc * alpha + P @ V_lora
        acc_alpha = gl.convert_layout(alpha[:, None], gl.SliceLayout(1, mfma_acc).parent)
        # The above might fail; do it via broadcast and convert_layout instead:
        # alpha is fp32 in SliceLayout(1, mfma_s); we need to apply per-row scaling on acc (mfma_acc).
        # Simpler: scale in mfma_acc layout by reusing the same vector but converted.
        alpha_acc = gl.convert_layout(alpha, gl.SliceLayout(1, mfma_acc))
        acc = acc * alpha_acc[:, None]

        P_bf = P.to(Q_ptr.dtype.element_ty)
        P_dot = gl.convert_layout(P_bf, dot_p_a)
        acc = gl.amd.cdna4.mfma(P_dot, V_lora_dot, acc)

        m_i = m_new
        l_i = l_new

        # Promote prefetched values to "current"
        cur_buf = next_buf
        topk_pos_klora = topk_pos_klora_next
        topk_pos_krope = topk_pos_krope_next
        topk_pos_mma = topk_pos_mma_next
        valid_klora = valid_klora_next
        valid_krope = valid_krope_next
        valid_mma = valid_mma_next
        safe_klora = safe_klora_next
        safe_krope = safe_krope_next

    # ---------- epilogue: process the last tile (NUM_TILES-1) ----------
    gl.amd.cdna4.async_copy.wait_group(0)

    klora_smem_cur = smem_klora.index(cur_buf)
    K_lora_T_dot = klora_smem_cur.load(dot_klora_b)
    V_lora_dot = klora_smem_cur.permute([1, 0]).load(dot_v_b)
    krope_smem_cur = smem_krope.index(cur_buf)
    K_rope_T_dot = krope_smem_cur.load(dot_krope_b)

    S_zero = gl.zeros([BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_s)
    S = gl.amd.cdna4.mfma(Q_lora_dot, K_lora_T_dot, S_zero)
    S = gl.amd.cdna4.mfma(Q_rope_dot, K_rope_T_dot, S)
    S = S * scale

    offs_h_mma = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_s))
    mask_h_mma = offs_h_mma < num_heads
    valid_mask = valid_mma[None, :] & mask_h_mma[:, None]
    S = gl.where(valid_mask, S, float("-inf"))

    m_j = gl.max(S, axis=1)
    m_new = gl.maximum(m_i, m_j)
    m_new = gl.where(m_new > float("-inf"), m_new, 0.0)
    alpha = gl.exp(m_i - m_new)
    P = gl.exp(S - m_new[:, None])
    l_new = alpha * l_i + gl.sum(P, axis=1)

    alpha_acc = gl.convert_layout(alpha, gl.SliceLayout(1, mfma_acc))
    acc = acc * alpha_acc[:, None]

    P_bf = P.to(Q_ptr.dtype.element_ty)
    P_dot = gl.convert_layout(P_bf, dot_p_a)
    acc = gl.amd.cdna4.mfma(P_dot, V_lora_dot, acc)

    m_i = m_new
    l_i = l_new

    # ---------- epilogue: divide and write outputs ----------
    l_i_acc = gl.convert_layout(l_i, gl.SliceLayout(1, mfma_acc))
    acc = acc / l_i_acc[:, None]
    lse = m_i + gl.log(l_i)

    # Output O[token_idx, h, v]
    offs_h_o = hg_offset + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blk_qlora))
    offs_v_o = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk_qlora))
    mask_h_o = offs_h_o < num_heads
    o_base = token_idx.to(tl.int64) * stride_o_t
    o_offs = (
        o_base
        + offs_h_o[:, None].to(tl.int64) * stride_o_h
        + offs_v_o[None, :].to(tl.int64)
    )
    acc_bf = acc.to(O_ptr.dtype.element_ty)
    acc_bf_blk = gl.convert_layout(acc_bf, blk_qlora)
    gl.amd.cdna4.buffer_store(
        stored_value=acc_bf_blk,
        ptr=O_ptr,
        offsets=o_offs.to(tl.int32),
        mask=mask_h_o[:, None],
    )

    # LSE[token_idx, h]
    offs_h_lse = hg_offset + gl.arange(0, BLOCK_H, layout=blk_lse)
    mask_h_lse = offs_h_lse < num_heads
    lse_base = token_idx * num_heads
    lse_offs = lse_base + offs_h_lse
    lse_blk = gl.convert_layout(lse, blk_lse)
    gl.amd.cdna4.buffer_store(
        stored_value=lse_blk,
        ptr=LSE_ptr,
        offsets=lse_offs.to(tl.int32),
        mask=mask_h_lse,
    )


# =====================================================================
# Launcher
# =====================================================================
def sparse_mla_fwd_gl(q, kv, topk_indices, kv_lora_rank=512, scale=None):
    """
    Sparse MLA forward (Gluon, MI350X / CDNA4).

    Args:
        q:             [total_tokens, num_heads, d_qk] bfloat16
        kv:            [total_tokens, 1, d_qk] bfloat16 (or [total_tokens, d_qk])
        topk_indices:  [total_tokens, topk] int32
        kv_lora_rank:  int, default 512
        scale:         float, default 1/sqrt(d_qk)

    Returns:
        o:   [total_tokens, num_heads, kv_lora_rank] same dtype as q
        lse: [total_tokens, num_heads] float32
    """
    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert topk_indices.is_contiguous()

    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]

    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)

    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.shape[0] == total_tokens and kv.shape[-1] == d_qk

    o = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    # Hard-coded best config (matches Triton autotune winner).
    BLOCK_H = 64
    TILE_K = 16
    num_warps = 4
    num_stages = 2  # informational; we have an explicit pipeline

    grid = (total_tokens, triton.cdiv(num_heads, BLOCK_H))

    _sparse_mla_fwd_gl_kernel[grid](
        Q_ptr=q, KV_ptr=kv, TopK_ptr=topk_indices,
        O_ptr=o, LSE_ptr=lse,
        stride_q_t=q.stride(0), stride_q_h=q.stride(1),
        stride_kv_t=kv.stride(0),
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        stride_topk_t=topk_indices.stride(0),
        scale=scale, num_heads=num_heads,
        TOPK=topk, BLOCK_H=BLOCK_H, TILE_K=TILE_K,
        D_V=kv_lora_rank, D_ROPE=rope_rank,
        num_warps=num_warps,
    )

    return o, lse


# =====================================================================
# Verify and profile
# =====================================================================
# TODO: Clean the verify and profile code for production
def verify_correctness():
    from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
        sparse_mla_fwd as sparse_mla_fwd_triton,
    )

    torch.manual_seed(0)
    B, S, H, D_V, D_ROPE, TOPK = 1, 128, 16, 256, 64, 64
    D_QK = D_V + D_ROPE
    total = B * S

    q = torch.randn(total, H, D_QK, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(total, 1, D_QK, dtype=torch.bfloat16, device="cuda")
    topk_indices = torch.randint(0, total, (total, TOPK), dtype=torch.int32, device="cuda")

    o_ref, lse_ref = sparse_mla_fwd_triton(q, kv, topk_indices, kv_lora_rank=D_V)
    o_gl, lse_gl = sparse_mla_fwd_gl(q, kv, topk_indices, kv_lora_rank=D_V)

    diff = (o_gl.float() - o_ref.float()).abs()
    err_o_abs = diff.max().item()
    # Two relative-error views, gated by ref magnitude to avoid blowup near zero.
    # bf16 has ~3-4 decimal digits, so for outputs in the 0.01-0.1 range a 1-2%
    # difference between two MFMA pipelines is normal.
    sig_lo = o_ref.float().abs() > 1e-2
    sig_hi = o_ref.float().abs() > 1e-1
    err_o_rel_lo = (diff[sig_lo] / o_ref.float().abs()[sig_lo]).max().item() if sig_lo.any() else 0.0
    err_o_rel_hi = (diff[sig_hi] / o_ref.float().abs()[sig_hi]).max().item() if sig_hi.any() else 0.0
    err_lse = (lse_gl - lse_ref).abs().max().item()
    print(f"max abs err (O):   {err_o_abs:.4e}")
    print(f"max rel err (O,|ref|>1e-2): {err_o_rel_lo:.4e}")
    print(f"max abs err (O,|ref|>1e-1): {err_o_rel_hi:.4e}")
    print(f"max abs err (LSE): {err_lse:.4e}")
    # Task spec bound. bf16 attention noise floor is ~2e-3 abs / 1e-2 rel for "large" outputs.
    assert err_o_abs < 0.5,         f"abs err {err_o_abs} >= 0.5"
    assert err_o_rel_hi < 1e-2,     f"rel err (|ref|>0.1) {err_o_rel_hi} >= 1e-2"
    print("Correctness PASSED.")


def profile_kernel(B=1, S=8192, H=128, D_V=512, D_ROPE=64, TOPK=1024,
                   warmup=10, steps=50):
    from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
        sparse_mla_fwd as sparse_mla_fwd_triton,
    )

    torch.manual_seed(0)
    D_QK = D_V + D_ROPE
    total = B * S
    q = torch.randn(total, H, D_QK, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(total, 1, D_QK, dtype=torch.bfloat16, device="cuda")
    topk_indices = torch.randint(0, total, (total, TOPK), dtype=torch.int32, device="cuda")

    # Warmup both
    for _ in range(warmup):
        sparse_mla_fwd_triton(q, kv, topk_indices, kv_lora_rank=D_V)
        sparse_mla_fwd_gl(q, kv, topk_indices, kv_lora_rank=D_V)
    torch.cuda.synchronize()

    # Time Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(steps):
        sparse_mla_fwd_triton(q, kv, topk_indices, kv_lora_rank=D_V)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / steps

    start.record()
    for _ in range(steps):
        sparse_mla_fwd_gl(q, kv, topk_indices, kv_lora_rank=D_V)
    end.record()
    torch.cuda.synchronize()
    gl_ms = start.elapsed_time(end) / steps

    print(f"Triton: {triton_ms:.3f} ms")
    print(f"Gluon : {gl_ms:.3f} ms")
    print(f"Speedup: {triton_ms / gl_ms:.2f}x")


if __name__ == "__main__":
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    # verify_correctness()
    profile_kernel()
