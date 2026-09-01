# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4 as cdna4_ops
from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as async_cp
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.experimental.gluon.language.amd.cdna4 import (
    mfma_scaled as mfma_scaled_cdna4,
)

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

_LOGGER = AiterTritonLogger()

# log(2)
_LN2 = gl.constexpr(0.6931471824645996)

# ---------------------------------------------------------------------------
# Layout helpers (constexpr -- evaluated at compile time)
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _bits(n):
    """log2 of a power of two."""
    return n.bit_length() - 1


@gluon.constexpr_function
def elem_bits_of(dtype):
    """Storage width of one element, in bits."""
    return dtype.primitive_bitwidth


@gluon.constexpr_function
def pad_interval(elem_bits):
    """Padding interval in elements: one 128-bit lane vector per lane of a warp.

    Also a lowering requirement, not just a bank-conflict choice -- the
    direct-to-LDS copy caps its vector at interval/warp_size, and CDNA4 only
    supports 128- and 32-bit direct-to-LDS, so a smaller interval makes
    buffer_load_to_shared unlowerable at 8-bit.
    """
    return 8192 // elem_bits


@gluon.constexpr_function
def dma_elems(elem_bits):
    """Elements per lane in one 128-bit global->LDS copy."""
    return 128 // elem_bits


@gluon.constexpr_function
def bases_to_source_layout(offset_bases, contiguity, num_warps, shape, warp_size=64):
    """DMA source layout matching a padded shared layout, bases partitioned the way
    CoalesceAsyncCopy does it: log2(contiguity) to registers, then a warp's worth to
    lanes, then log2(num_warps) to warps, then whatever is left back to registers.

    Used with `compute_efficient_padded_shared_layout`, whose bases the hand-rolled
    `dma_source_layout` below does not reproduce at 8-bit.
    """
    rank = len(shape)
    lg2_c = _bits(contiguity)
    lg2_ws = _bits(warp_size)
    lg2_nw = _bits(num_warps)
    i = 0
    reg = list(offset_bases[i : i + lg2_c])
    i += lg2_c
    lane = list(offset_bases[i : i + lg2_ws])
    i += lg2_ws
    warp = list(offset_bases[i : i + lg2_nw])
    i += lg2_nw
    warp = warp + [[0] * rank] * (lg2_nw - len(warp))
    reg = reg + list(offset_bases[i:])
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=lane,
        warp_bases=warp,
        block_bases=[],
        shape=list(shape),
    )


@gluon.constexpr_function
def blocked_row_major(shape, vec, num_warps, warp_size=64):
    """Blocked layout with ``vec`` contiguous elements along dim 1 (the fast axis)."""
    n_vec_cols = max(shape[1] // vec, 1)
    t1 = min(n_vec_cols, warp_size)
    t0 = max(warp_size // t1, 1)
    return gl.BlockedLayout([1, vec], [t0, t1], [num_warps, 1], [1, 0])


@gluon.constexpr_function
def swizzled(order, vec=8, max_phase=16):
    """Swizzled shared layout whose fastest-varying axis is ``order[0]``.

    ``vec`` must match the number of elements each lane writes contiguously during the
    global->LDS DMA (16 bytes = 8 fp16), otherwise the DMA and the LDS reads disagree on
    the swizzle and the kernel silently produces wrong results.
    """
    return gl.SwizzledSharedLayout(
        vec=vec, per_phase=1, max_phase=max_phase, order=order
    )


@gluon.constexpr_function
def _contig_strided(rows, cols, contig_dim):
    """(extent along the contiguous axis, extent along the strided axis)."""
    return (rows, cols) if contig_dim == 0 else (cols, rows)


@gluon.constexpr_function
def dma_vec(rows, cols, contig_dim, num_warps, elem_bits, warp_size=64):
    """Elements per lane a global->LDS DMA can move for this tile, or 0 if none can.

    ``buffer_load_to_shared`` wants every lane to carry a full 128 bits.  The ISA
    also admits a 32-bit form, but the CDNA4 lowering cannot coalesce the resulting
    4-byte LDS writes against a padded destination, so only 128 bits is offered;
    :func:`_pick_tile` reshapes small head dims to suit.
    """
    c, _ = _contig_strided(rows, cols, contig_dim)
    n_threads = num_warps * warp_size
    if (rows * cols) % n_threads != 0:
        return 0
    per_lane = rows * cols // n_threads
    for vec in (128 // elem_bits,):
        if vec < 1 or vec > per_lane or vec > c:
            continue
        # The contiguous axis must fit in the register + lane bits available to it.
        if (c // vec) > warp_size:
            continue
        # ... and what is left of the strided axis must cover warps, the remaining
        # lane bits and the remaining register bits exactly.
        r_lane_bits = _bits(warp_size) - _bits(c // vec)
        if _bits(rows * cols // c) < _bits(num_warps) + r_lane_bits:
            continue
        return vec
    return 0


@gluon.constexpr_function
def dma_source_layout(rows, cols, contig_dim, num_warps, vec, warp_size=64):
    """Address layout for a global->LDS DMA of a ``[rows, cols]`` tile.

    ``buffer_load_to_shared`` needs each lane to name ``vec`` *contiguous* elements
    and needs the resulting LDS writes to be coalesced.  Only one bit order satisfies
    both: the low ``log2(vec)`` bits of the contiguous axis go to registers, the rest
    of that axis to lanes, and the strided axis splits low-to-high across warps, then
    registers, then whatever lane bits the contiguous axis did not need.
    """
    c, r = _contig_strided(rows, cols, contig_dim)

    def base(cv, rv):
        return [cv, rv] if contig_dim == 0 else [rv, cv]

    c_lane_bits = _bits(c // vec)
    c_reg_bits = _bits(c) - c_lane_bits
    r_warp_bits = _bits(num_warps)
    r_lane_bits = _bits(warp_size) - c_lane_bits
    r_reg_bits = _bits(r) - r_warp_bits - r_lane_bits

    reg = [base(1 << i, 0) for i in range(c_reg_bits)]
    reg += [base(0, 1 << (r_warp_bits + i)) for i in range(r_reg_bits)]
    lane = [base(1 << (c_reg_bits + i), 0) for i in range(c_lane_bits)]
    lane += [base(0, 1 << (r_warp_bits + r_reg_bits + i)) for i in range(r_lane_bits)]
    warp = [base(0, 1 << i) for i in range(r_warp_bits)]
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=lane,
        warp_bases=warp,
        block_bases=[],
        shape=[rows, cols],
    )


@gluon.constexpr_function
def dma_shared_layout(rows, cols, contig_dim, pad, interval):
    """Row-staggered padded shared layout for a K^T or V tile.

    Both tiles are written by the DMA along their contiguous axis but read back by
    the MFMA along the other one, which is bank-conflict free only when consecutive
    strided-axis entries are staggered far apart in LDS.  ``with_identity_for`` does
    not produce that stagger, so the offset bases are built explicitly: contiguous
    axis in the low bits, then the *high* bits of the strided index, then its low
    bits.  ``pad`` differs per tile because the reads do -- K^T is read along its
    contiguous axis (``ds_read_b128``) while V is read across it by the
    hardware-transposing ``ds_read_b64_tr_b16``.
    """
    c, r = _contig_strided(rows, cols, contig_dim)

    def base(cv, rv):
        return [cv, rv] if contig_dim == 0 else [rv, cv]

    rows_per_interval = max(interval // c, 1)
    split = max(r // rows_per_interval, 1)
    bases = [base(1 << i, 0) for i in range(_bits(c))]
    bases += [base(0, split << i) for i in range(_bits(rows_per_interval))]
    bases += [base(0, 1 << i) for i in range(_bits(split))]
    return gl.PaddedSharedLayout([[interval, pad]], bases, [], [rows, cols])


@gluon.constexpr_function
def dma_layouts_ok(head_dim, block_n, num_warps, elem_bits):
    """Can both the K^T and the V tile be moved by ``buffer_load_to_shared``?"""
    if dma_vec(head_dim, block_n, 0, num_warps, elem_bits) == 0:
        return False
    if dma_vec(block_n, head_dim, 1, num_warps, elem_bits) == 0:
        return False
    # The padded layout needs the strided axis to split evenly across padding intervals.
    return block_n % max(pad_interval(elem_bits) // head_dim, 1) == 0


# ---------------------------------------------------------------------------
# Softmax, cut where the tutorial cuts it
# ---------------------------------------------------------------------------


@gluon.jit
def _max_propagating_nan(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def _row_max(x):
    """Row-wise reduce-max using IEEE 754 maximum (propagates NaN)."""
    return gl.reduce(x, 1, _max_propagating_nan)


@gluon.jit
def _softmax_vec1(qk, m_run, qk_scale, SCALE_ON_Q: gl.constexpr):
    """VEC1 -- softmax numerator: new row max, the ``exp2`` burst, and ``alpha``.

    Lives in the ``dot2`` cluster: ``exp2`` is the most expensive item in the softmax
    (a TRANS op issues at half the rate of a plain VALU) and the PV chain has room for
    it.  Its outputs are consumed one pipeline stage later.
    """
    if SCALE_ON_Q:
        # qk already carries the scale (folded into Q before the loop), so the row max
        # needs no multiply and the exponent argument is a plain subtract.
        m_ij = _row_max(qk)
        m_new = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
        p = gl.exp2(qk - m_new[:, None])
    else:
        m_ij = _row_max(qk) * qk_scale
        m_new = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
        # Fuse the multiply and the subtract at the source (one llvm.fmuladd) rather
        # than leaving an fmul/fsub pair for the backend to contract later.
        p = gl.exp2(gl.fma(qk, qk_scale, -m_new[:, None]))
    alpha = gl.exp2(m_run - m_new)
    return m_new, p, alpha


@gluon.jit
def _softmax_vec2(acc, l_i, p, alpha, P_LAYOUT: gl.constexpr, DTYPE: gl.constexpr):
    """VEC2 -- softmax denominator, accumulator rescale, and the P operand downcast.

    Lives in the ``dot1`` cluster, beside the Q@K^T MFMA.  ``p`` and ``alpha`` were
    produced by VEC1 in the *previous* iteration.

    Op order matters: the accumulator rescale goes first.  This cluster's shadow
    cannot fully absorb it, and leading with it keeps the uncovered remainder ahead
    of the MFMAs, where an exposed packed op pays no back-to-back hazard.
    """
    acc = acc * alpha[:, None]
    l_ij = gl.sum(p, axis=1)
    l_i = l_i * alpha + l_ij
    p_dot = gl.convert_layout(p.to(DTYPE), P_LAYOUT)
    return acc, l_i, p_dot


# ---------------------------------------------------------------------------
# Pipeline building blocks
# ---------------------------------------------------------------------------


@gluon.jit
def _mma(a, b, acc, IS_FP8: gl.constexpr):
    """One MFMA. fp8 needs the scaled instruction: it is the only CDNA4 encoding
    that reaches the 32x32x64 shape, and the plain fp8 MFMA is the same 32x32x16
    rate as bf16. Null scales lower to a constant exponent of 1."""
    if IS_FP8:
        return mfma_scaled_cdna4(a, None, "e4m3", b, None, "e4m3", acc)
    else:
        return mfma_cdna4(a, b, acc)


@gluon.jit
def _dma(smem, base, offsets, mask, HAS_MASK: gl.constexpr):
    """Kick off one global->LDS DMA tile and close its commit group."""
    if HAS_MASK:
        async_cp.buffer_load_to_shared(smem, base, offsets, mask=mask, other=0.0)
    else:
        async_cp.buffer_load_to_shared(smem, base, offsets)
    async_cp.commit_group()


@gluon.jit
def _kv_mask(
    offs_n,
    offs_d,
    start_n,
    IS_K: gl.constexpr,
    SEQLEN_K: gl.constexpr,
    ACTUAL_HEAD_DIM: gl.constexpr,
    PADDED_HEAD: gl.constexpr,
):
    """DMA mask for one K^T or V tile: in-range KV tokens, and the real head columns.

    ``buffer_load_to_shared`` broadcasts the mask against the offsets, so each half is
    handed over with only the axis it actually constrains.  K^T is [HEAD_DIM, BLOCK_N]
    and V is [BLOCK_N, HEAD_DIM], which is the only difference between the two.
    """
    if IS_K:
        mask = (start_n + offs_n)[None, :] < SEQLEN_K
        if PADDED_HEAD:
            mask = mask & (offs_d[:, None] < ACTUAL_HEAD_DIM)
    else:
        mask = (start_n + offs_n)[:, None] < SEQLEN_K
        if PADDED_HEAD:
            mask = mask & (offs_d[None, :] < ACTUAL_HEAD_DIM)
    return mask


@gluon.jit
def _generic_dma(
    kt_smem,
    v_smem,
    slot,
    blk,
    n_full_blocks,
    k_base,
    v_base,
    kt_off,
    v_off,
    kt_offs_n,
    kt_offs_d,
    v_offs_n,
    v_offs_d,
    k_head_mask,
    v_head_mask,
    stride_kn,
    stride_vn,
    BLOCK_N: gl.constexpr,
    SEQLEN_K: gl.constexpr,
    ACTUAL_HEAD_DIM: gl.constexpr,
    PADDED_HEAD: gl.constexpr,
    MASKED_BLOCKS: gl.constexpr,
):
    """Stage tile ``blk`` of K and V into LDS slot ``slot`` for the generic loop.

    The KV-token mask is only built past the unmasked range; a full block takes the
    same copy the pipelined loop uses.  Both branches still carry the head-dim mask,
    which is a property of the tensor rather than of the block.
    """
    start_n = blk * BLOCK_N
    k_ptr = k_base + start_n * stride_kn
    v_ptr = v_base + start_n * stride_vn
    if MASKED_BLOCKS > 0:
        if blk >= n_full_blocks:
            _dma(
                kt_smem.index(slot),
                k_ptr,
                kt_off,
                _kv_mask(
                    kt_offs_n,
                    kt_offs_d,
                    start_n,
                    True,
                    SEQLEN_K,
                    ACTUAL_HEAD_DIM,
                    PADDED_HEAD,
                ),
                True,
            )
            _dma(
                v_smem.index(slot),
                v_ptr,
                v_off,
                _kv_mask(
                    v_offs_n,
                    v_offs_d,
                    start_n,
                    False,
                    SEQLEN_K,
                    ACTUAL_HEAD_DIM,
                    PADDED_HEAD,
                ),
                True,
            )
        else:
            _dma(kt_smem.index(slot), k_ptr, kt_off, k_head_mask, PADDED_HEAD)
            _dma(v_smem.index(slot), v_ptr, v_off, v_head_mask, PADDED_HEAD)
    else:
        _dma(kt_smem.index(slot), k_ptr, kt_off, k_head_mask, PADDED_HEAD)
        _dma(v_smem.index(slot), v_ptr, v_off, v_head_mask, PADDED_HEAD)


@gluon.jit
def _pipe_tile(
    acc,
    l_i,
    m_run,
    p_c,
    alpha_c,
    kt_dot,
    q_dot,
    kt_smem,
    v_smem,
    k_base,
    v_base,
    kt_off,
    v_off,
    k_mask,
    v_mask,
    kt_step,
    v_step,
    blk,
    CUR: gl.constexpr,
    NXT: gl.constexpr,
    MFMA_LAYOUT: gl.constexpr,
    KT_DOT: gl.constexpr,
    V_DOT: gl.constexpr,
    P_DOT: gl.constexpr,
    qk_scale,
    SCALE_ON_Q: gl.constexpr,
    DTYPE: gl.constexpr,
    HAS_MASK: gl.constexpr,
    IS_FP8: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """One tile of the rotated four-cluster loop.

    Tile ``blk`` owns LDS slot ``CUR`` for both K and V.  Reading down the clusters:

      dot1  Q@K^T for tile blk+1        VEC2 for tile blk   (rescale, sum, downcast)
      mem1  read V[blk] out of LDS      DMA K[blk+3] -> slot NXT
      dot2  P@V for tile blk            VEC1 for tile blk+1 (row max, exp2)
      mem2  read K[blk+2] out of LDS    DMA V[blk+2] -> slot CUR

    Every LDS index is a compile-time constant because the caller unrolls by two;
    the stage boundaries keep the two waves on a SIMD one cluster apart.
    """
    with warp_pipeline_stage("dot1"):
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=MFMA_LAYOUT)
        qk = _mma(q_dot, kt_dot, qk, IS_FP8)
        acc, l_i, p_dot = _softmax_vec2(acc, l_i, p_c, alpha_c, P_DOT, DTYPE)

    # wait_group(2) is the depth the pipeline needs, but the backend then derives a
    # too-loose s_waitcnt vmcnt and lets a ds_read race ahead of the copy filling it.
    # Draining one extra group closes that; the group is not needed yet.
    async_cp.wait_group(1)
    with warp_pipeline_stage("mem1"):
        v_dot = async_cp.load_shared_relaxed(v_smem.index(CUR), V_DOT)
        _dma(kt_smem.index(NXT), k_base + (blk + 3) * kt_step, kt_off, k_mask, HAS_MASK)

    with warp_pipeline_stage("dot2"):
        acc = _mma(p_dot, v_dot, acc, IS_FP8)
        m_run, p_c, alpha_c = _softmax_vec1(qk, m_run, qk_scale, SCALE_ON_Q)

    async_cp.wait_group(1)
    with warp_pipeline_stage("mem2"):
        kt_dot = async_cp.load_shared_relaxed(kt_smem.index(CUR), KT_DOT)
        _dma(v_smem.index(CUR), v_base + (blk + 2) * v_step, v_off, v_mask, HAS_MASK)

    return acc, l_i, m_run, p_c, alpha_c, kt_dot


# ---------------------------------------------------------------------------
# Shared kernel body
# ---------------------------------------------------------------------------
# The whole rotated four-cluster pipeline lives here, once.  The two kernel
# entry points below differ only in what they must fetch before the loop can
# start -- bf16 nothing, fp8 three descale scalars -- so they own their own
# signatures and hand the results down as ordinary values.


@gluon.jit
def _program_ids(HQ: gl.constexpr, HK: gl.constexpr):
    """(start_m, off_h_q, off_h_k, off_z) for this workgroup.

    Shared by both kernel entries: the fp8 one needs ``off_h_k`` before the body
    runs, to index the descales, so the derivation cannot live in the body alone.
    """
    start_m = gl.program_id(0)
    off_h_q = gl.program_id(1)
    off_z = gl.program_id(2)
    gl.assume(start_m >= 0)
    gl.assume(off_h_q >= 0)
    gl.assume(off_z >= 0)
    GROUP_SIZE: gl.constexpr = HQ // HK
    return start_m, off_h_q, off_h_q // GROUP_SIZE, off_z


@gluon.jit
def _mha_fwd_body(
    Q,
    K,
    V,
    Out,
    L,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    start_m,
    off_h_q,
    off_h_k,
    off_z,
    qk_scale,
    v_descale,
    HQ: gl.constexpr,
    SEQLEN_Q: gl.constexpr,
    SEQLEN_K: gl.constexpr,
    ACTUAL_HEAD_DIM: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    WRITE_LSE: gl.constexpr,
    DEAD_ROW_LSE: gl.constexpr,
    DTYPE: gl.constexpr,
    SCALE_ON_Q: gl.constexpr,
    PIPELINED: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """FlashAttention-2 forward for one [BLOCK_M, HEAD_DIM] output tile.

    ``qk_scale`` already carries ``log2(e)`` and, at fp8, the Q and K descales;
    ``v_descale`` is loop-invariant and folds into the epilogue.  Both are runtime
    values that constant-fold at bf16, where the caller passes literals.
    """
    # ---------------- layouts ----------------
    ELEM_BITS: gl.constexpr = elem_bits_of(DTYPE)
    IS_FP8: gl.constexpr = ELEM_BITS == 8

    # Not a ratio of the element width: fp8's 32x32x64 is a different, double-rate
    # instruction reachable only through the scaled MFMA, not merely a wider K. The
    # plain fp8 MFMA is 32x32x16, the same rate as bf16.
    MFMA_K: gl.constexpr = 64 if IS_FP8 else 16
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, MFMA_K],
        transposed=True,
        warps_per_cta=[NUM_WARPS, 1],
    )
    # QK reads its operands 128 bits at a time. PV reads V through the
    # hardware-transposing 64-bit LDS read, hence half the width -- except at fp8,
    # where the scaled MFMA requires both operands to share one k_width.
    QK_KW: gl.constexpr = 128 // ELEM_BITS
    PV_KW: gl.constexpr = 128 // ELEM_BITS if IS_FP8 else 64 // ELEM_BITS
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma, k_width=QK_KW
    )
    kt_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma, k_width=QK_KW
    )
    p_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma, k_width=PV_KW
    )
    v_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma, k_width=PV_KW
    )

    m_slice: gl.constexpr = gl.SliceLayout(1, mfma)  # per-row vectors (m_i, l_i)
    n_slice: gl.constexpr = gl.SliceLayout(0, mfma)  # per-column vectors

    # `order=[1, 0]` puts the head dim contiguous. Q and O need different vector
    # widths when their element widths differ (fp8 in, fp32 out), so they get
    # separate layouts; at 16-bit both resolve to the same one.
    O_BITS: gl.constexpr = elem_bits_of(Out.dtype.element_ty)
    q_blocked: gl.constexpr = blocked_row_major(
        [BLOCK_M, HEAD_DIM], dma_elems(ELEM_BITS), NUM_WARPS
    )
    o_blocked: gl.constexpr = blocked_row_major(
        [BLOCK_M, HEAD_DIM], dma_elems(O_BITS), NUM_WARPS
    )
    q_shared: gl.constexpr = swizzled([1, 0], vec=dma_elems(ELEM_BITS))
    offs_m_blocked: gl.constexpr = gl.SliceLayout(1, q_blocked)
    offs_d_blocked: gl.constexpr = gl.SliceLayout(0, q_blocked)
    offs_m_o_blk: gl.constexpr = gl.SliceLayout(1, o_blocked)
    offs_d_o_blk: gl.constexpr = gl.SliceLayout(0, o_blocked)

    # The LSE store is 1-D over BLOCK_M and wants consecutive rows on consecutive lanes,
    # so it needs its own M-major arrangement rather than a slice of `q_blocked`, which
    # would leave only 4 lanes along M and 8 stride-4 stores.
    lse_blocked: gl.constexpr = gl.BlockedLayout(
        [1, 8], [16, 4], [NUM_WARPS, 1], [1, 0]
    )
    offs_m_lse: gl.constexpr = gl.SliceLayout(1, lse_blocked)

    KV_VEC: gl.constexpr = dma_vec(HEAD_DIM, BLOCK_N, 0, NUM_WARPS, ELEM_BITS)
    PAD_IV: gl.constexpr = pad_interval(ELEM_BITS)
    # K is read transposed -- a [HEAD_DIM, BLOCK_N] tile whose dim 0 (the head dim) is
    # the contiguous one -- while V keeps its natural [BLOCK_N, HEAD_DIM] shape.
    if IS_FP8:
        # At 8-bit the stagger built below is not the one the transposed read wants,
        # so defer to the compiler's own chooser and derive the matching copy-source
        # layout from the bases it returns.
        kt_shared: gl.constexpr = cdna4_ops.compute_efficient_padded_shared_layout(
            kt_dot_layout, [HEAD_DIM, BLOCK_N], DTYPE, is_k_contig=True
        )
        v_shared: gl.constexpr = cdna4_ops.compute_efficient_padded_shared_layout(
            v_dot_layout, [BLOCK_N, HEAD_DIM], DTYPE, is_k_contig=False
        )
        kt_src: gl.constexpr = bases_to_source_layout(
            kt_shared.offset_bases, KV_VEC, NUM_WARPS, [HEAD_DIM, BLOCK_N]
        )
        v_src: gl.constexpr = bases_to_source_layout(
            v_shared.offset_bases, KV_VEC, NUM_WARPS, [BLOCK_N, HEAD_DIM]
        )
    else:
        kt_shared: gl.constexpr = dma_shared_layout(
            HEAD_DIM, BLOCK_N, 0, pad=QK_KW, interval=PAD_IV
        )
        v_shared: gl.constexpr = dma_shared_layout(
            BLOCK_N, HEAD_DIM, 1, pad=32, interval=PAD_IV
        )
        kt_src: gl.constexpr = dma_source_layout(
            HEAD_DIM, BLOCK_N, 0, NUM_WARPS, KV_VEC
        )
        v_src: gl.constexpr = dma_source_layout(BLOCK_N, HEAD_DIM, 1, NUM_WARPS, KV_VEC)

    PADDED_HEAD: gl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    BUF_DEPTH: gl.constexpr = 2

    # ---------------- how many KV blocks does this workgroup touch? ----------------
    n_blocks = gl.cdiv(SEQLEN_K, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_seqlen = gl.cdiv(
            (start_m + 1) * BLOCK_M + SEQLEN_K - SEQLEN_Q, BLOCK_N
        )
        n_blocks = min(n_blocks, n_blocks_seqlen)

    if n_blocks <= 0:
        # Every row of this Q block is fully masked: write zeros to O and +inf to LSE.
        offs_m_z = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_o_blk)
        offs_d_z = gl.arange(0, HEAD_DIM, layout=offs_d_o_blk)
        o_base_z = Out + off_z * stride_oz + off_h_q * stride_oh
        o_offs_z = offs_m_z[:, None] * stride_om + offs_d_z[None, :] * stride_on
        o_mask_z = offs_m_z[:, None] < SEQLEN_Q
        if PADDED_HEAD:
            o_mask_z = o_mask_z & (offs_d_z[None, :] < ACTUAL_HEAD_DIM)
        zeros = gl.zeros(
            [BLOCK_M, HEAD_DIM], dtype=Out.dtype.element_ty, layout=o_blocked
        )
        gl.amd.cdna4.buffer_store(zeros, o_base_z, o_offs_z, mask=o_mask_z)
        if WRITE_LSE:
            offs_l = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_lse)
            l_ptrs = L + (off_z * HQ + off_h_q) * SEQLEN_Q + offs_l
            # What a row that attends to nothing stores is caller-chosen, because
            # the two hosts disagree: flash_attn_3 (fwd_prefill.py's `invalid_mask`
            # store) and aiter/test_mha_common.py::opus_ref_lse use -inf, while
            # mha.py's own forward writes 0.0 there.  Live rows are identical
            # either way.
            gl.store(
                l_ptrs,
                gl.full([BLOCK_M], DEAD_ROW_LSE, dtype=gl.float32, layout=offs_m_lse),
                mask=offs_l < SEQLEN_Q,
            )
        return

    # ---------------- Q: load once, into registers, for the whole kernel ----------------
    # q_smem's live range ends at the load below, so Triton overlays the K/V buffers on
    # top of it -- which is the only reason a 256x128 Q tile and two double-buffered
    # K/V tiles fit in LDS together.  Keep the read here, before the K/V allocation.
    q_smem = gl.allocate_shared_memory(DTYPE, [BLOCK_M, HEAD_DIM], q_shared)

    q_base = Q + off_z * stride_qz + off_h_q * stride_qh
    q_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_blocked)
    q_offs_d = gl.arange(0, HEAD_DIM, layout=offs_d_blocked)
    q_offs = q_offs_m[:, None] * stride_qm + q_offs_d[None, :] * stride_qk
    q_mask = q_offs_m[:, None] < SEQLEN_Q
    if PADDED_HEAD:
        q_mask = q_mask & (q_offs_d[None, :] < ACTUAL_HEAD_DIM)
        # A masked async copy simply does not write the masked lanes -- `other` is not
        # materialised in LDS.  The masked lanes are always the same ones (the head-dim
        # padding), so zeroing the buffer once up front is enough and keeps the garbage
        # from poisoning the MFMA accumulators.
        q_smem.store(gl.zeros([BLOCK_M, HEAD_DIM], dtype=DTYPE, layout=q_blocked))
        gl.barrier()
    async_cp.buffer_load_to_shared(q_smem, q_base, q_offs, mask=q_mask, other=0.0)
    async_cp.commit_group()
    async_cp.wait_group(0)
    q_dot = q_smem.load(layout=q_dot_layout)
    if SCALE_ON_Q:
        # Fold qk_scale into the Q operand once, here, rather than into every tile's
        # score matrix inside the loop.  q_dot lives in registers for every iteration,
        # so this is a single pass over 32 VGPRs.
        q_dot = (q_dot.to(gl.float32) * qk_scale).to(DTYPE)

    # ---------------- K / V shared memory and DMA addresses ----------------
    kt_smem = gl.allocate_shared_memory(
        DTYPE, [BUF_DEPTH, HEAD_DIM, BLOCK_N], kt_shared
    )
    v_smem = gl.allocate_shared_memory(DTYPE, [BUF_DEPTH, BLOCK_N, HEAD_DIM], v_shared)

    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    v_base = V + off_z * stride_vz + off_h_k * stride_vh

    # The intra-tile offset pattern never changes; successive tiles advance the scalar
    # base pointer instead, which keeps the address VALU out of the loop entirely.
    kt_offs_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, kt_src))
    kt_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, kt_src))
    kt_off = kt_offs_d[:, None] * stride_kk + kt_offs_n[None, :] * stride_kn
    v_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, v_src))
    v_offs_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, v_src))
    v_off = v_offs_n[:, None] * stride_vn + v_offs_d[None, :] * stride_vk
    kt_step = BLOCK_N * stride_kn
    v_step = BLOCK_N * stride_vn

    if PADDED_HEAD:
        # buffer_load_to_shared broadcasts the mask against the offsets, so a mask
        # that only constrains one axis can be handed over as-is.
        k_head_mask = kt_offs_d[:, None] < ACTUAL_HEAD_DIM
        v_head_mask = v_offs_d[None, :] < ACTUAL_HEAD_DIM
        for buf in gl.static_range(BUF_DEPTH):
            kt_smem.index(buf).store(
                gl.zeros([HEAD_DIM, BLOCK_N], dtype=DTYPE, layout=kt_src)
            )
            v_smem.index(buf).store(
                gl.zeros([BLOCK_N, HEAD_DIM], dtype=DTYPE, layout=v_src)
            )
        gl.barrier()
    else:
        k_head_mask = None
        v_head_mask = None

    # ---------------- accumulators ----------------
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=m_slice)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=m_slice)
    acc = gl.zeros([BLOCK_M, HEAD_DIM], dtype=gl.float32, layout=mfma)

    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=m_slice)
    offs_n = gl.arange(0, BLOCK_N, layout=n_slice)

    # ---------------- split into unmasked / masked block ranges ----------------
    IS_MODULO_MN: gl.constexpr = (SEQLEN_K % BLOCK_N == 0) and (SEQLEN_Q % BLOCK_M == 0)
    if IS_CAUSAL:
        MASKED_BLOCKS: gl.constexpr = BLOCK_M // BLOCK_N + (0 if IS_MODULO_MN else 1)
    else:
        MASKED_BLOCKS: gl.constexpr = 0 if (SEQLEN_K % BLOCK_N == 0) else 1
    masked_blocks = min(MASKED_BLOCKS, n_blocks)
    n_full_blocks = n_blocks - masked_blocks

    # ---------------- the rotated four-cluster pipeline ----------------
    # MIN_PIPE_BLOCKS is a *correctness* bound, not a tuning knob.  The WAR
    # protection between an LDS slot's read and the async copy that overwrites it
    # comes from the warp-pipeline stage barriers, which only line up once the loop
    # is in steady state.  Below eight full tiles the three-tile prologue and
    # three-tile drain overlap, those barriers stop separating the pair, and the
    # kernel returns run-to-run-varying garbage.
    #
    # Checked per workgroup, not per launch: under a causal mask each M-block sees a
    # different number of full tiles, so even a long sequence gives its first few
    # M-blocks only a handful.  Those take the generic loop below.
    MIN_PIPE_BLOCKS: gl.constexpr = 8

    tail_start = 0
    if PIPELINED and n_full_blocks >= MIN_PIPE_BLOCKS:
        # -- Prologue --------------------------------------------------------
        # Prime the rotation for output tile 0: compute all of tile 0's ahead-work
        # (qk[0], m[0], p[0], alpha[0]) and the K registers for tile 1, and stage
        # K[0..2] / V[0..1] into the two LDS slots.  K runs three tiles ahead, so slot 0
        # is reused for K[2] once LRK[0] has read K[0] -- hence the barrier.
        # Commit order K0, V0, K1, K2, V1 leaves {K2, V1} pending, which is exactly the
        # loop's steady-state entry condition.
        _dma(kt_smem.index(0), k_base, kt_off, k_head_mask, PADDED_HEAD)
        _dma(v_smem.index(0), v_base, v_off, v_head_mask, PADDED_HEAD)
        _dma(kt_smem.index(1), k_base + kt_step, kt_off, k_head_mask, PADDED_HEAD)

        async_cp.wait_group(2)  # K[0] has landed
        kt0 = async_cp.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfma)
        qk = _mma(q_dot, kt0, qk, IS_FP8)
        m_run, p_c, alpha_c = _softmax_vec1(qk, m_i, qk_scale, SCALE_ON_Q)

        gl.barrier()  # WAR: LRK[0]'s ds_read against K[2]'s write into the same slot
        _dma(kt_smem.index(0), k_base + 2 * kt_step, kt_off, k_head_mask, PADDED_HEAD)
        async_cp.wait_group(1)  # K[1] has landed
        kt_dot = async_cp.load_shared_relaxed(kt_smem.index(1), kt_dot_layout)
        _dma(v_smem.index(1), v_base + v_step, v_off, v_head_mask, PADDED_HEAD)

        # -- Main loop, unrolled 2x so the LDS slots are compile-time constants -----
        # Over the two tiles of a pair the K and V buffers exchange places; unrolling by
        # BUF_DEPTH returns each to where it started, so no tile has to be copied just
        # to restore the naming.
        pairs = (n_full_blocks - 3) // 2
        for pair in tl.range(0, pairs):
            blk = pair * 2
            acc, l_i, m_run, p_c, alpha_c, kt_dot = _pipe_tile(
                acc,
                l_i,
                m_run,
                p_c,
                alpha_c,
                kt_dot,
                q_dot,
                kt_smem,
                v_smem,
                k_base,
                v_base,
                kt_off,
                v_off,
                k_head_mask,
                v_head_mask,
                kt_step,
                v_step,
                blk,
                0,
                1,
                mfma,
                kt_dot_layout,
                v_dot_layout,
                p_dot_layout,
                qk_scale,
                SCALE_ON_Q,
                DTYPE,
                PADDED_HEAD,
                IS_FP8,
                BLOCK_M,
                BLOCK_N,
            )
            acc, l_i, m_run, p_c, alpha_c, kt_dot = _pipe_tile(
                acc,
                l_i,
                m_run,
                p_c,
                alpha_c,
                kt_dot,
                q_dot,
                kt_smem,
                v_smem,
                k_base,
                v_base,
                kt_off,
                v_off,
                k_head_mask,
                v_head_mask,
                kt_step,
                v_step,
                blk + 1,
                1,
                0,
                mfma,
                kt_dot_layout,
                v_dot_layout,
                p_dot_layout,
                qk_scale,
                SCALE_ON_Q,
                DTYPE,
                PADDED_HEAD,
                IS_FP8,
                BLOCK_M,
                BLOCK_N,
            )

        # An odd number of pipelined tiles leaves one over.  It is always an "even" tile
        # (slots 0/1), because each pair returns the buffers to where they started.
        if (n_full_blocks - 3) % 2 == 1:
            acc, l_i, m_run, p_c, alpha_c, kt_dot = _pipe_tile(
                acc,
                l_i,
                m_run,
                p_c,
                alpha_c,
                kt_dot,
                q_dot,
                kt_smem,
                v_smem,
                k_base,
                v_base,
                kt_off,
                v_off,
                k_head_mask,
                v_head_mask,
                kt_step,
                v_step,
                pairs * 2,
                0,
                1,
                mfma,
                kt_dot_layout,
                v_dot_layout,
                p_dot_layout,
                qk_scale,
                SCALE_ON_Q,
                DTYPE,
                PADDED_HEAD,
                IS_FP8,
                BLOCK_M,
                BLOCK_N,
            )

        # -- Drain: the last three output tiles, with no prefetch left to issue ------
        # On entry: outputs [.., n-4] are done, K[0..n-1] and V[0..n-2] are in LDS,
        # kt_dot holds K[n-2], and (m_run, p_c, alpha_c) belong to tile n-3.
        nm3 = n_full_blocks - 3
        nm2 = n_full_blocks - 2
        nm1 = n_full_blocks - 1
        s_nm3 = (nm3 % BUF_DEPTH).to(tl.int32)
        s_nm2 = (nm2 % BUF_DEPTH).to(tl.int32)
        s_nm1 = (nm1 % BUF_DEPTH).to(tl.int32)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfma)
        qk = _mma(q_dot, kt_dot, qk, IS_FP8)
        async_cp.wait_group(2)
        v_dot = async_cp.load_shared_relaxed(v_smem.index(s_nm3), v_dot_layout)
        acc, l_i, p_dot = _softmax_vec2(acc, l_i, p_c, alpha_c, p_dot_layout, DTYPE)
        acc = _mma(p_dot, v_dot, acc, IS_FP8)
        m_run, p_c, alpha_c = _softmax_vec1(qk, m_run, qk_scale, SCALE_ON_Q)
        gl.barrier()  # WAR: LRV[n-3] against V[n-1]'s write into the same slot
        _dma(
            v_smem.index(s_nm1), v_base + nm1 * v_step, v_off, v_head_mask, PADDED_HEAD
        )
        async_cp.wait_group(2)
        kt_dot = async_cp.load_shared_relaxed(kt_smem.index(s_nm1), kt_dot_layout)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfma)
        qk = _mma(q_dot, kt_dot, qk, IS_FP8)
        async_cp.wait_group(1)
        v_dot = async_cp.load_shared_relaxed(v_smem.index(s_nm2), v_dot_layout)
        acc, l_i, p_dot = _softmax_vec2(acc, l_i, p_c, alpha_c, p_dot_layout, DTYPE)
        acc = _mma(p_dot, v_dot, acc, IS_FP8)
        m_run, p_c, alpha_c = _softmax_vec1(qk, m_run, qk_scale, SCALE_ON_Q)

        async_cp.wait_group(0)
        v_dot = async_cp.load_shared_relaxed(v_smem.index(s_nm1), v_dot_layout)
        acc, l_i, p_dot = _softmax_vec2(acc, l_i, p_c, alpha_c, p_dot_layout, DTYPE)
        acc = _mma(p_dot, v_dot, acc, IS_FP8)

        m_i = m_run
        tail_start = n_full_blocks

    # ---------------- generic loop: the masked tail, and everything the ----------------
    # ---------------- rotated pipeline declined to take ----------------
    # Double-buffered and one tile ahead, but with the softmax left whole and no warp
    # pipeline.  Runs the causal diagonal, sequences too short to prime the rotation,
    # and every tile of a head dim too big for the rotated loop's carried state.
    if tail_start < n_blocks:
        # A masked async copy leaves the masked LDS lanes untouched -- `other` is never
        # materialised.  Later tiles inherit real (finite) data from earlier ones, which
        # multiplies against p == 0 harmlessly, but the first tile would inherit
        # uninitialised LDS, and 0 * NaN is NaN.  Zero both slots once.  The leading
        # barrier also covers the WAR hazard against the rotated loop's last LDS reads.
        gl.barrier()
        for buf in gl.static_range(BUF_DEPTH):
            kt_smem.index(buf).store(
                gl.zeros([HEAD_DIM, BLOCK_N], dtype=DTYPE, layout=kt_src)
            )
            v_smem.index(buf).store(
                gl.zeros([BLOCK_N, HEAD_DIM], dtype=DTYPE, layout=v_src)
            )
        gl.barrier()

        _generic_dma(
            kt_smem,
            v_smem,
            0,
            tail_start,
            n_full_blocks,
            k_base,
            v_base,
            kt_off,
            v_off,
            kt_offs_n,
            kt_offs_d,
            v_offs_n,
            v_offs_d,
            k_head_mask,
            v_head_mask,
            stride_kn,
            stride_vn,
            BLOCK_N,
            SEQLEN_K,
            ACTUAL_HEAD_DIM,
            PADDED_HEAD,
            MASKED_BLOCKS,
        )

        for blk in range(tail_start, n_blocks):
            cur = (blk - tail_start) % BUF_DEPTH
            nxt = 1 - cur
            # Clamped so the speculative prefetch can never run off the end of K/V; on
            # the last iteration it reloads the same tile, which is harmless.
            _generic_dma(
                kt_smem,
                v_smem,
                nxt,
                min(blk + 1, n_blocks - 1),
                n_full_blocks,
                k_base,
                v_base,
                kt_off,
                v_off,
                kt_offs_n,
                kt_offs_d,
                v_offs_n,
                v_offs_d,
                k_head_mask,
                v_head_mask,
                stride_kn,
                stride_vn,
                BLOCK_N,
                SEQLEN_K,
                ACTUAL_HEAD_DIM,
                PADDED_HEAD,
                MASKED_BLOCKS,
            )
            async_cp.wait_group(2)  # this tile's K and V have landed

            start_n = blk * BLOCK_N
            kt = async_cp.load_shared_relaxed(kt_smem.index(cur), kt_dot_layout)
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfma)
            qk = _mma(q_dot, kt, qk, IS_FP8)

            # Only blocks past the unmasked range need masking.  When this loop runs
            # the whole range most of its tiles are full, and the two `where`s would
            # be pure overhead there.
            if MASKED_BLOCKS > 0 and blk >= n_full_blocks:
                # Out-of-range KV tokens.
                qk = gl.where((start_n + offs_n)[None, :] < SEQLEN_K, qk, float("-inf"))
                if IS_CAUSAL:
                    causal_bound = start_n + offs_n + (SEQLEN_Q - SEQLEN_K)
                    qk = gl.where(
                        offs_m[:, None] >= causal_bound[None, :], qk, float("-inf")
                    )

            m_i, p, alpha = _softmax_vec1(qk, m_i, qk_scale, SCALE_ON_Q)
            acc, l_i, p_dot = _softmax_vec2(acc, l_i, p, alpha, p_dot_layout, DTYPE)

            v = async_cp.load_shared_relaxed(v_smem.index(cur), v_dot_layout)
            acc = _mma(p_dot, v, acc, IS_FP8)
            gl.barrier()  # WAR: this tile's LDS reads against the next prefetch's writes
        async_cp.wait_group(0)

    # ---------------- epilogue ----------------
    # Reciprocal on the [BLOCK_M] vector rather than on the [BLOCK_M, HEAD_DIM]
    # accumulator: a full IEEE divide per accumulator element costs ~4 VALU ops each.
    l_recip = 1.0 / l_i
    if IS_FP8:
        # v_descale is loop-invariant, so folding it in here is algebraically the
        # same as the reference's per-tile multiply but rounds once instead of
        # n_blocks times -- and it leaves the mfma(p, v, acc) chain untouched.  It
        # rides on the [BLOCK_M] vector, not the [BLOCK_M, HEAD_DIM] accumulator.
        # Guarded rather than relying on v_descale == 1.0 folding, so the 16-bit
        # path emits byte-identical code to before this parameter existed.
        l_recip = l_recip * v_descale
    acc = acc * l_recip[:, None]

    if IS_CAUSAL:
        # For seqlen_q > seqlen_k, whole rows can be masked out; they produced NaNs above.
        causal_start_idx = SEQLEN_Q - SEQLEN_K
        acc = gl.where(offs_m[:, None] >= causal_start_idx, acc, 0.0)

    if WRITE_LSE:
        offs_l = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_lse)
        l_ptrs = L + (off_z * HQ + off_h_q) * SEQLEN_Q + offs_l
        # m_i and l_i are carried in log2 units, so `* LN2` lands on the natural-log
        # LSE flash_attn_3 returns: (m*log2(e) + log2(l)) * ln2 == m + ln(l).
        #
        # The `where` guards rows that attend to nothing: their running max never
        # leaves -inf, so `alpha = exp2(m_run - m_new)` is exp2(NaN), which poisons
        # l_i.  `acc` is scrubbed by the causal `where` above, l_i is not.  Reachable
        # when an M-block holds live and dead rows at once -- causal with
        # SEQLEN_Q > SEQLEN_K, the difference not a multiple of BLOCK_M.
        #
        # `m_i == -inf` is the *detection* and must stay; DEAD_ROW_LSE is only what
        # gets stored, and differs per caller (see the early-exit store above).
        lse = gl.where(m_i == float("-inf"), DEAD_ROW_LSE, (m_i + gl.log2(l_i)) * _LN2)
        lse = gl.convert_layout(lse, offs_m_lse)
        gl.store(l_ptrs, lse, mask=offs_l < SEQLEN_Q)

    # Downcast first, then convert the layout: a convert out of the MFMA layout goes
    # through LDS, and doing it in fp16/bf16 halves the bytes that round trip.  The
    # blocked destination gives every lane 8 contiguous elements of one row, i.e. one
    # dwordx4 store instead of the MFMA layout's four strided dwordx2.
    offs_d_o = gl.arange(0, HEAD_DIM, layout=offs_d_o_blk)
    offs_m_o = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_o_blk)
    o_base = Out + off_z * stride_oz + off_h_q * stride_oh
    o_offs = offs_m_o[:, None] * stride_om + offs_d_o[None, :] * stride_on
    o_mask = offs_m_o[:, None] < SEQLEN_Q
    if PADDED_HEAD:
        o_mask = o_mask & (offs_d_o[None, :] < ACTUAL_HEAD_DIM)
    acc_out = gl.convert_layout(acc.to(Out.dtype.element_ty), o_blocked)
    gl.amd.cdna4.buffer_store(acc_out, o_base, o_offs, mask=o_mask)


# ---------------------------------------------------------------------------
# Kernel entry points
# ---------------------------------------------------------------------------


@gluon.jit
def _mha_fwd_bf16_kernel(
    Q,
    K,
    V,
    Out,
    L,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    SM_SCALE: gl.constexpr,
    HQ: gl.constexpr,
    HK: gl.constexpr,
    SEQLEN_Q: gl.constexpr,
    SEQLEN_K: gl.constexpr,
    ACTUAL_HEAD_DIM: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    WRITE_LSE: gl.constexpr,
    DEAD_ROW_LSE: gl.constexpr,
    DTYPE: gl.constexpr,
    PIPELINED: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """16-bit (bf16 / fp16) dense forward.

    There are no descales, and the softmax scale is a compile-time constant, so it
    is folded into the Q operand once before the loop (``SCALE_ON_Q``) instead of
    into every tile's score matrix.  ``qk_scale`` still reaches the body because
    the Q fold needs it; being a constexpr expression it costs nothing.
    """
    start_m, off_h_q, off_h_k, off_z = _program_ids(HQ, HK)
    QK_SCALE: gl.constexpr = SM_SCALE * 1.44269504089  # * log2(e)
    _mha_fwd_body(
        Q,
        K,
        V,
        Out,
        L,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        start_m,
        off_h_q,
        off_h_k,
        off_z,
        QK_SCALE,
        1.0,  # v_descale -- unused, IS_FP8 gates the epilogue fold
        HQ,
        SEQLEN_Q,
        SEQLEN_K,
        ACTUAL_HEAD_DIM,
        HEAD_DIM,
        IS_CAUSAL,
        WRITE_LSE,
        DEAD_ROW_LSE,
        DTYPE,
        True,  # SCALE_ON_Q
        PIPELINED,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
    )


@gluon.jit
def _mha_fwd_fp8_kernel(
    Q,
    K,
    V,
    Out,
    L,
    Q_Descale,
    K_Descale,
    V_Descale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_dqz,
    stride_dkz,
    stride_dvz,
    SM_SCALE: gl.constexpr,
    HQ: gl.constexpr,
    HK: gl.constexpr,
    SEQLEN_Q: gl.constexpr,
    SEQLEN_K: gl.constexpr,
    ACTUAL_HEAD_DIM: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    WRITE_LSE: gl.constexpr,
    DEAD_ROW_LSE: gl.constexpr,
    DTYPE: gl.constexpr,
    PIPELINED: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """fp8 (e4m3) dense forward.

    The three descales are fp32 ``[batch, nheads_k]`` with the head axis
    contiguous, so only their batch stride is passed.  All three are indexed by
    ``off_h_k`` -- which equals ``off_h_q`` when ``HQ == HK`` -- matching
    ``_triton_kernels/flash_attn_triton_amd/fwd_prefill.py``.  They are workgroup
    scalars, so this is three s_loads outside the loop.

    Q cannot absorb the softmax scale the way the 16-bit path does (it is fp8, and
    scaling it would quantise twice), so ``SCALE_ON_Q`` is off and the scale rides
    into the softmax instead, carrying the Q and K descales with it.  The V descale
    only ever multiplies ``acc``, so it is loop-invariant and folds into the
    epilogue rather than into every tile.

    P is cast straight to fp8 with no rescale and no clamp: the reference hard-codes
    ``FP8_P_DESCALE = False``, and p is already in [0, 1] after the exp2.
    """
    start_m, off_h_q, off_h_k, off_z = _program_ids(HQ, HK)

    q_descale = gl.load(Q_Descale + off_z * stride_dqz + off_h_k)
    k_descale = gl.load(K_Descale + off_z * stride_dkz + off_h_k)
    v_descale = gl.load(V_Descale + off_z * stride_dvz + off_h_k)

    # log2(e) folded in here so the softmax's exp2 needs no further scaling.
    qk_scale = SM_SCALE * q_descale * k_descale * 1.44269504089

    _mha_fwd_body(
        Q,
        K,
        V,
        Out,
        L,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        start_m,
        off_h_q,
        off_h_k,
        off_z,
        qk_scale,
        v_descale,
        HQ,
        SEQLEN_Q,
        SEQLEN_K,
        ACTUAL_HEAD_DIM,
        HEAD_DIM,
        IS_CAUSAL,
        WRITE_LSE,
        DEAD_ROW_LSE,
        DTYPE,
        False,  # SCALE_ON_Q -- an fp8 Q cannot absorb the scale
        PIPELINED,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
    )


# ---------------------------------------------------------------------------
# Host side: tile heuristics, eligibility predicate, launcher
# ---------------------------------------------------------------------------

_TL_DTYPE = {torch.float16: "fp16", torch.bfloat16: "bf16"}

# Keeps the accumulators in VGPRs, off the v_accvgpr path.  Only valid when the
# live set fits without the AGPRs (_fits_arch_vgprs); otherwise the accumulator
# spills instead.  The tuple form is required -- as a plain string, "0,0" would be
# split on its comma.
_AGPR_ATTR = ("amdgpu-agpr-alloc", "0,0")

# The pipeline's prologue and drain hold the accumulator, both score tiles and a
# K or V tile at once, and LLVM's default schedule spills there.  The
# minimum-register scheduler removes those spills.  Applied only alongside
# _AGPR_ATTR: it is a win when the kernel is meant to live in the architected
# VGPRs, and a large loss when the accumulator must sit in AGPRs.
_SCHED_ATTR = ("amdgpu-sched-strategy", "iterative-minreg")

_LLVM_FN_ATTRS = (_AGPR_ATTR, _SCHED_ATTR)


def _elem_bits(torch_dtype) -> int:
    """Storage width of one element, in bits -- the host-side twin of
    :func:`elem_bits_of`, which cannot be called outside a Gluon context."""
    return torch.finfo(torch_dtype).bits


# (BLOCK_M, BLOCK_N, num_warps) per masking mode, tuned on gfx950 with
# op_tests/op_benchmarks/triton/bench_mha.py -impl gluon.  num_warps is not free:
# the MFMA tiling fixes BLOCK_M = 32 * num_warps.  Causal takes the narrower M
# tile, which wastes less work on the diagonal; non-causal takes the wider one,
# which amortises the pipeline fill/drain better.  Splitting them costs no extra
# kernel specializations -- IS_CAUSAL is already a constexpr.
_TILE_NON_CAUSAL = (256, 64, 8)
_TILE_CAUSAL = (128, 64, 4)

# Causal crossover, in average full KV blocks per workgroup (see _block_split).
# Below it flash_attn_3's smaller BLOCK_M wins on the diagonal, so the caller
# falls back.  Non-causal needs no such floor.
_CAUSAL_MIN_AVG_FULL_BLOCKS = 24

# Largest fp8 head dim that fits in LDS with the tiles above: measured 65,536 B at
# d=64, 131,072 B at d=128, 262,144 B at d=192 and d=256, against a 163,840 B
# limit, so d >= 192 does not launch.  Temporary -- that is ~2x what the *16-bit*
# kernel needs for the same shape (68,016 B at d=128) despite fp8 elements being
# half as wide, and the shared layouts were checked to cover exactly rows*cols with
# ~3% padding at both widths.  So this is an allocation inefficiency to chase, not
# a real requirement; d=256 should be the largest fp8 win once it lifts.
_FP8_MAX_HEAD_DIM = 128


def _tile(causal: bool) -> tuple[int, int, int]:
    """(BLOCK_M, BLOCK_N, num_warps) for this masking mode.

    fp8 shares these for now.  They were tuned at 16-bit and the balance does
    shift -- the fp8 MFMA covers 4x the K per issue while the softmax VALU work is
    unchanged -- so a sweep may want to split them later; BLOCK_N=128 is the lever
    and becomes affordable at fp8.
    """
    return _TILE_CAUSAL if causal else _TILE_NON_CAUSAL


def _pick_tile(head_dim, block_n, num_warps, elem_bits, warp_size=64):
    """Pick (padded head dim, BLOCK_N) so the K/V tiles can be DMA'd into LDS.

    ``buffer_load_to_shared`` hands every lane 16 contiguous *bytes*, so a K or V
    tile needs at least ``(128 // elem_bits) * num_warps * warp_size`` elements --
    8 per lane at 16-bit, 16 at fp8.  A small head dim buys those either by
    widening the tile along the KV axis (free -- the extra columns are real work)
    or by padding the head dim (wasted MFMA).  Widen first, and only pad once
    BLOCK_N would exceed 128, where the score tile starts to dominate the register
    budget.
    """
    padded = max(1 << (head_dim - 1).bit_length(), 16)
    need_n = (128 // elem_bits) * num_warps * warp_size // padded
    while need_n > 128:
        padded *= 2
        need_n //= 2
    return padded, max(block_n, need_n)


def _block_split(sq, sk, block_m, block_n, causal):
    """(masked blocks per workgroup, average full blocks per workgroup).

    Mirrors the split the kernel makes, so the host can tell how much work the
    rotated pipeline would get.  Under a bottom-right-aligned causal mask each
    M-block sees a different number of KV blocks, hence the average.
    """
    n_max = -(-sk // block_n)
    m_blocks = max(-(-sq // block_m), 1)
    if causal:
        modulo = (sk % block_n == 0) and (sq % block_m == 0)
        masked = block_m // block_n + (0 if modulo else 1)
        total = sum(
            min(n_max, max(0, -(-((m + 1) * block_m + sk - sq) // block_n)))
            for m in range(m_blocks)
        )
        avg = total / m_blocks
    else:
        masked = 0 if sk % block_n == 0 else 1
        avg = n_max
    return masked, max(0.0, avg - min(masked, n_max))


def _use_pipeline(sq, sk, block_m, block_n, causal):
    """Is the rotated four-cluster loop worth building for this shape?

    The fill and drain are six tiles of straight-line code, so the rotation needs
    full tiles to amortise them.  A shape that also needs the generic loop -- causal
    diagonal, ragged K -- puts both loops in one function competing for registers,
    which raises the bar.
    """
    masked, avg_full = _block_split(sq, sk, block_m, block_n, causal)
    # 8 is the kernel's MIN_PIPE_BLOCKS, below which the rotation is incorrect, not
    # merely unprofitable.  20 is the profitability bar when the generic loop is
    # live too.
    return avg_full >= (8 if masked == 0 else 20)


def _fits_arch_vgprs(head_dim, block_m, block_n, num_warps, elem_bits, warp_size=64):
    """Does one wave's live set fit in the 256 architected VGPRs?

    ``waves_per_eu=2`` and the two LLVM attributes all hang on this and stand or
    fall together.  The live set does not shrink with the tile: the accumulator
    alone is ``head_dim / 2`` registers per lane, because ``BLOCK_M`` cancels
    against ``num_warps``.  Head dims that do not fit take the whole register file,
    the AGPRs, and no ping-pong.

    Only the operand terms scale with ``elem_bits`` -- the accumulator and the
    score tiles are fp32 whatever the inputs are.  ``slack`` was calibrated at
    16-bit; see risk R4 in the fp8 plan if head_dim=256 spills.
    """
    lanes = num_warps * warp_size
    per_reg = 32 // elem_bits  # input elements per 32-bit VGPR
    acc = head_dim // 2  # [BLOCK_M, HEAD_DIM] fp32
    q_operand = head_dim // 2 // per_reg  # [BLOCK_M, HEAD_DIM] at elem_bits
    scores = 2 * block_m * block_n // lanes  # qk plus the carried p, fp32
    kv = head_dim * block_n // lanes // per_reg  # one K or V tile, at elem_bits
    slack = 24  # addresses, masks, m_i / l_i
    return acc + q_operand + scores + kv + slack <= 256


def _shape_and_strides(q, k, v, o, layout):
    """(batch, hq, hk, sq, sk, head_dim, q/k/v/o strides as (z, h, s, d))."""
    if layout == "bhsd":
        batch, hq, sq, head_dim = q.shape
        hk, sk = k.shape[1], k.shape[2]
        perm = (0, 1, 2, 3)
    elif layout == "bshd":
        batch, sq, hq, head_dim = q.shape
        hk, sk = k.shape[2], k.shape[1]
        perm = (0, 2, 1, 3)
    else:
        raise ValueError(f"unsupported layout {layout!r} (expected 'bhsd' or 'bshd')")
    strides = tuple(tuple(t.stride(i) for i in perm) for t in (q, k, v, o))
    return batch, hq, hk, sq, sk, head_dim, strides


def is_available() -> bool:
    """True iff this module is usable on the running device.

    Matches the architecture by name rather than through ``arch_info.is_cdna4``:
    that helper arrived with #4978 and was reverted by #5149, so it is not on
    main.  ``get_arch`` is the form the other Gluon modules gate on.
    """
    return arch_info.get_arch() == "gfx950"


def _geometry_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    layout: str,
    min_head_dim: int,
) -> bool:
    """The shape conditions both dtypes share -- can this kernel express the shape,
    and is it expected to beat the fallback on it?

    Everything here is about geometry and the register/LDS budget; the dtype
    admission itself is the caller's job.
    """
    if layout not in ("bshd", "bhsd"):
        return False
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False

    # flash_attn_3 allows a V head dim different from Q/K, and head dims that are
    # not a multiple of 16 (it pads the compute tile).  This kernel does neither:
    # it carries one head dim, and its masked global->LDS copy has to keep each
    # lane's 16-byte write whole, which only works on a 16-element boundary.
    head_dim = q.shape[-1]
    if v.shape[-1] != head_dim or head_dim > 256 or head_dim % 16 != 0:
        return False
    if head_dim < min_head_dim:
        return False

    _, hq, hk, sq, sk, _, _ = _shape_and_strides(q, k, v, q, layout)
    if hq % hk != 0:
        return False

    elem_bits = _elem_bits(q.dtype)
    block_m, block_n, num_warps = _tile(causal)
    padded_head_dim, block_n = _pick_tile(head_dim, block_n, num_warps, elem_bits)
    if not dma_layouts_ok(padded_head_dim, block_n, num_warps, elem_bits):
        return False

    if causal:
        # Two conditions have to hold before this kernel beats flash_attn_3 on
        # causal, where its smaller BLOCK_M wastes less work on the diagonal.
        #
        # First, the rotated pipeline has to be buildable at all: a head dim whose
        # accumulator alone is half the register file runs everything through the
        # generic loop, which has no answer on causal.  (Those head dims still win
        # on non-causal, where the DMA and layout work pays.)
        if not _fits_arch_vgprs(
            padded_head_dim, block_m, block_n, num_warps, elem_bits
        ):
            return False
        # Second, the sequence has to be long enough.  avg_full_blocks accounts for
        # sq != sk and the bottom-right alignment, so the crossover holds for
        # cross-attention shapes too.
        _, avg_full = _block_split(sq, sk, block_m, block_n, causal)
        if avg_full < _CAUSAL_MIN_AVG_FULL_BLOCKS:
            return False

    return True


def gluon_fwd_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    *,
    layout: str = "bshd",
    **unsupported: object,
) -> bool:
    """Can -- and should -- this 16-bit shape go through the Gluon forward kernel?

    Every keyword in ``unsupported`` is a feature the kernel does not implement
    (cu_seqlens, page_table, qv, descales, rotary, ...); any of them being set is an
    immediate no.  Taking them as ``**kwargs`` rather than enumerating them keeps the
    caller honest: a new argument added to the caller disqualifies the Gluon path
    until someone handles it deliberately.
    """
    if any(value is not None and value is not False for value in unsupported.values()):
        return False
    if q.dtype not in _TL_DTYPE or not (q.dtype == k.dtype == v.dtype):
        return False
    return _geometry_supported(q, k, v, causal, layout, min_head_dim=16)


def _descale_ok(descale: torch.Tensor | None, batch: int, hk: int) -> bool:
    """An fp8 descale must be fp32 ``[batch, nheads_k]`` with the head axis
    contiguous -- the kernel indexes it as ``base + z * stride(0) + h_k`` and
    passes no head stride."""
    return (
        isinstance(descale, torch.Tensor)
        and descale.dtype == torch.float32
        and descale.shape == (batch, hk)
        and descale.stride(1) == 1
    )


def gluon_fp8_fwd_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    *,
    layout: str = "bshd",
    **unsupported: object,
) -> bool:
    """Can -- and should -- this fp8 shape go through the Gluon forward kernel?

    fp8 is admitted by the *presence* of all three descales, in the exact layout the
    kernel indexes; 16-bit is admitted by their absence (see
    :func:`gluon_fwd_supported`).

    ``head_dim >= 64`` because the scaled MFMA is 32x32x64 and needs ``K % 32 == 0``.
    ``head_dim <= _FP8_MAX_HEAD_DIM`` is an LDS bound -- see that constant.
    """
    if any(value is not None and value is not False for value in unsupported.values()):
        return False
    fp8 = get_fp8_e4m3_dtype()
    if q.dtype != fp8 or not (q.dtype == k.dtype == v.dtype):
        return False
    if q.dim() != 4:
        return False
    if q.shape[-1] > _FP8_MAX_HEAD_DIM:
        return False
    batch, _, hk, _, _, _, _ = _shape_and_strides(q, k, v, q, layout)
    if not all(_descale_ok(d, batch, hk) for d in (q_descale, k_descale, v_descale)):
        return False
    return _geometry_supported(q, k, v, causal, layout, min_head_dim=64)


def _gl_dtype(torch_dtype):
    """torch dtype -> the Gluon dtype the kernel wants."""
    return {
        torch.float16: gl.float16,
        torch.bfloat16: gl.bfloat16,
        # gfx950 e4m3 is the OCP "fn" form; gfx942's "fnuz" is a different type and
        # is not reachable here (is_available() gates on CDNA4).
        torch.float8_e4m3fn: gl.float8e4nv,
    }[torch_dtype]


def gluon_mha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    layout: str = "bshd",
    out: torch.Tensor | None = None,
    dead_row_lse: float = float("-inf"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense 16-bit FlashAttention forward on gfx950.

    Returns ``(out, softmax_lse)`` in exactly the form ``flash_attn_3.fwd``
    produces for the dense path: ``out`` shaped like ``q`` in ``q.dtype``, and
    ``softmax_lse`` shaped ``[batch, nheads_q, seqlen_q]`` in fp32, holding the
    natural-log LSE with the softmax scale already folded in.

    ``dead_row_lse`` is what a row that attends to nothing stores.  The default
    ``-inf`` matches flash_attn_3 and ``test_mha_common.opus_ref_lse``; callers
    replacing ``mha.py``'s own forward must pass ``0.0``, which is what that path
    writes.  Live rows are identical either way.

    Call :func:`gluon_fwd_supported` first; this function assumes it passed.
    """
    if out is None:
        out = torch.empty_like(q)
    batch, hq, hk, sq, sk, head_dim, strides = _shape_and_strides(q, k, v, out, layout)

    elem_bits = _elem_bits(q.dtype)
    block_m, block_n, num_warps = _tile(causal)
    padded_head_dim, block_n = _pick_tile(head_dim, block_n, num_warps, elem_bits)
    fits = _fits_arch_vgprs(padded_head_dim, block_m, block_n, num_warps, elem_bits)
    pipelined = fits and _use_pipeline(sq, sk, block_m, block_n, causal)

    lse = torch.empty((batch, hq, sq), device=q.device, dtype=torch.float32)
    q_st, k_st, v_st, o_st = strides

    _LOGGER.info(
        f"MHA_FWD [gluon/{arch_info.get_arch()}]: q={tuple(q.shape)} k={tuple(k.shape)} "
        f"causal={causal} block_m={block_m} block_n={block_n} pipelined={pipelined}"
    )

    grid = (triton.cdiv(sq, block_m), hq, batch)
    _mha_fwd_bf16_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        *q_st,
        *k_st,
        *v_st,
        *o_st,
        SM_SCALE=softmax_scale,
        HQ=hq,
        HK=hk,
        SEQLEN_Q=sq,
        SEQLEN_K=sk,
        ACTUAL_HEAD_DIM=head_dim,
        HEAD_DIM=padded_head_dim,
        IS_CAUSAL=causal,
        WRITE_LSE=True,
        DEAD_ROW_LSE=dead_row_lse,
        DTYPE=_gl_dtype(q.dtype),
        PIPELINED=pipelined,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
        waves_per_eu=2 if fits else 1,
        # Both attributes or neither; head dims that need the AGPRs keep LLVM's
        # default schedule (see _SCHED_ATTR).
        llvm_fn_attrs=_LLVM_FN_ATTRS if fits else (),
    )
    return out, lse


def gluon_mha_fwd_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    layout: str = "bshd",
    out: torch.Tensor | None = None,
    dead_row_lse: float = float("-inf"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense fp8 (e4m3) FlashAttention forward on gfx950.

    Returns ``(out, softmax_lse)`` in exactly the form ``flash_attn_3.fwd``
    produces for the fp8 dense path: ``out`` in **fp32** (not q.dtype -- fp8 has no
    useful output range), and ``softmax_lse`` shaped ``[batch, nheads_q, seqlen_q]``
    in fp32.  The LSE is in descaled units: it includes
    ``q_descale * k_descale * softmax_scale`` and excludes ``v_descale``, which is
    what the reference produces and what the fp8 backward expects.

    Only each descale's batch stride is passed; the head axis must be contiguous.

    Call :func:`gluon_fp8_fwd_supported` first; this function assumes it passed.
    """
    if out is None:
        out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    batch, hq, hk, sq, sk, head_dim, strides = _shape_and_strides(q, k, v, out, layout)

    elem_bits = _elem_bits(q.dtype)
    block_m, block_n, num_warps = _tile(causal)
    padded_head_dim, block_n = _pick_tile(head_dim, block_n, num_warps, elem_bits)
    fits = _fits_arch_vgprs(padded_head_dim, block_m, block_n, num_warps, elem_bits)
    pipelined = fits and _use_pipeline(sq, sk, block_m, block_n, causal)

    lse = torch.empty((batch, hq, sq), device=q.device, dtype=torch.float32)
    q_st, k_st, v_st, o_st = strides

    _LOGGER.info(
        f"MHA_FWD_FP8 [gluon/{arch_info.get_arch()}]: q={tuple(q.shape)} "
        f"k={tuple(k.shape)} causal={causal} block_m={block_m} block_n={block_n} "
        f"pipelined={pipelined}"
    )

    grid = (triton.cdiv(sq, block_m), hq, batch)
    _mha_fwd_fp8_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        q_descale,
        k_descale,
        v_descale,
        *q_st,
        *k_st,
        *v_st,
        *o_st,
        q_descale.stride(0),
        k_descale.stride(0),
        v_descale.stride(0),
        SM_SCALE=softmax_scale,
        HQ=hq,
        HK=hk,
        SEQLEN_Q=sq,
        SEQLEN_K=sk,
        ACTUAL_HEAD_DIM=head_dim,
        HEAD_DIM=padded_head_dim,
        IS_CAUSAL=causal,
        WRITE_LSE=True,
        DEAD_ROW_LSE=dead_row_lse,
        DTYPE=_gl_dtype(q.dtype),
        PIPELINED=pipelined,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
        waves_per_eu=2 if fits else 1,
        llvm_fn_attrs=_LLVM_FN_ATTRS if fits else (),
    )
    return out, lse
