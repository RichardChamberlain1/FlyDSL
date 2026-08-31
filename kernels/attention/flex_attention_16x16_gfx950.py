# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Flash/flex-attention forward using MFMA 16x16x16 for the QK GEMM (gfx950).

This kernel uses 4x MFMA 16x16x16 instructions per K-step to compute a 32x32
score tile (matching the 32x32x16 kernel's tile size). The PV GEMM uses the
proven 32x32x16 MFMA path (hybrid approach) since the 16x16 C-fragment values
can be packed into 32x32 B-operands.

Target: gfx950 (MI350, CDNA4). Requires num_groups >= 2 for stagger.
"""

from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl, arith
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

from kernels.attention.pipeline import pipeline_stagger_enabled
from kernels.attention.flash_attn_utils import (
    _stagger_extra_barrier_if_one,
    _stagger_extra_barrier_if_zero,
)

try:
    from flydsl.expr.rocdl.universal import make_buffer_ptr as _make_buffer_ptr
except ImportError:
    from flydsl.expr import buffer_ops
    from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace

    def _make_buffer_ptr(ptr, num_records_bytes=None):
        if num_records_bytes is None:
            num_records_bytes = fx.Int64(0xFFFFFFFF)
        elif not isinstance(num_records_bytes, fx.Int64):
            num_records_bytes = fx.Int64(num_records_bytes)
        buf_ptr_ty = fx.PointerType.get(
            elem_ty=ptr.element_type.ir_type,
            address_space=TargetAddressSpace.BufferDesc,
            alignment=ptr.alignment,
        )
        return fx.make_ptr(
            buf_ptr_ty,
            [
                ptr,
                fx.Int16(0).ir_value(),
                num_records_bytes.ir_value(),
                fx.Int32(buffer_ops._get_buffer_flags()).ir_value(),
            ],
        )

GFX950_WAVE_SIZE = 64
GFX950_DMA_BYTES = 16

# s_waitcnt bitfield encoding for gfx942/gfx950 (CDNA3/4):
# vmcnt: bits [3:0] and [15:14], lgkmcnt: bits [11:8], expcnt: bits [6:4]
_WAIT_VMCNT_0 = 0x0F70    # vmcnt=0, lgkmcnt=15, expcnt=7
_WAIT_LGKMCNT_0 = 0xC07F  # vmcnt=63, lgkmcnt=0, expcnt=7
_WAIT_ALL = 0              # vmcnt=0, lgkmcnt=0, expcnt=0

FLEX_DTYPE_BF16 = 2
FLEX_DTYPE_FP16 = 3

_LOG2E = 1.4426950408889634

MASK_NONE = 0
MASK_CAUSAL = 1
MASK_SLIDING_WINDOW = 2

SCORE_NONE = 0
SCORE_ALIBI = 1

_FM = fx.arith.FastMathFlags.fast
_FM_CONTRACT = fx.arith.FastMathFlags.contract


# ── FlexMod classes (identical to 32x32 kernel) ──────────────────────────
class FlexMod:
    has_mask = False
    has_score = False
    needs_safe_norm = False

    def kv_range(self, q_min_wg, q_max_wg, n_kv_tiles, block_n):
        return fx.Int32(0), fx.Int32(n_kv_tiles)

    def tile_needs_mask(self, kv_tile_idx, q_idx, block_n):
        return fx.Int32(0) != fx.Int32(0)

    def apply_mask(self, score, q_idx, kv_idx):
        return score

    def apply_score(self, score, b, h, q_idx, kv_idx):
        return score


class CausalMask(FlexMod):
    has_mask = True
    needs_safe_norm = True

    def kv_range(self, q_min_wg, q_max_wg, n_kv_tiles, block_n):
        raw_hi = (q_max_wg + fx.Int32(block_n)) // fx.Int32(block_n)
        kv_hi = fx.Int32(arith.minsi(raw_hi.ir_value(), fx.Int32(n_kv_tiles).ir_value()))
        return fx.Int32(0), kv_hi

    def tile_needs_mask(self, kv_tile_idx, q_idx, block_n):
        kv_tile_end = kv_tile_idx * fx.Int32(block_n) + fx.Int32(block_n - 1)
        return kv_tile_end > q_idx

    def apply_mask(self, score, q_idx, kv_idx):
        return (kv_idx <= q_idx).select(score, fx.Float32(-1e9))


class SlidingWindowMask(FlexMod):
    has_mask = True
    needs_safe_norm = True

    def __init__(self, window):
        self.window = window

    def kv_range(self, q_min_wg, q_max_wg, n_kv_tiles, block_n):
        raw_hi = (q_max_wg + fx.Int32(block_n)) // fx.Int32(block_n)
        kv_hi = fx.Int32(arith.minsi(raw_hi.ir_value(), fx.Int32(n_kv_tiles).ir_value()))
        raw_lo = (q_min_wg - fx.Int32(self.window)) // fx.Int32(block_n)
        kv_lo = fx.Int32(arith.maxsi(raw_lo.ir_value(), fx.Int32(0).ir_value()))
        return kv_lo, kv_hi

    def tile_needs_mask(self, kv_tile_idx, q_idx, block_n):
        kv_tile_end = kv_tile_idx * fx.Int32(block_n) + fx.Int32(block_n - 1)
        kv_tile_start = kv_tile_idx * fx.Int32(block_n)
        too_far = kv_tile_end > q_idx
        out_of_window = (q_idx - kv_tile_start) > fx.Int32(self.window)
        return too_far | out_of_window

    def apply_mask(self, score, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        in_window = (q_idx - kv_idx) <= fx.Int32(self.window)
        return (causal & in_window).select(score, fx.Float32(-1e9))


class AlibiScore(FlexMod):
    has_score = True

    def __init__(self, slope):
        self.slope = slope

    def apply_score(self, score, b, h, q_idx, kv_idx):
        bias = (kv_idx - q_idx).to(fx.Float32) * fx.Float32(self.slope)
        return fx.Float32(score) + bias


class CompositeMod(FlexMod):
    def __init__(self, score_mod, mask_mod):
        self._score = score_mod
        self._mask = mask_mod
        self.has_score = score_mod.has_score
        self.has_mask = mask_mod.has_mask
        self.needs_safe_norm = mask_mod.needs_safe_norm

    def kv_range(self, q_min_wg, q_max_wg, n_kv_tiles, block_n):
        return self._mask.kv_range(q_min_wg, q_max_wg, n_kv_tiles, block_n)

    def tile_needs_mask(self, kv_tile_idx, q_idx, block_n):
        return self._mask.tile_needs_mask(kv_tile_idx, q_idx, block_n)

    def apply_mask(self, score, q_idx, kv_idx):
        return self._mask.apply_mask(score, q_idx, kv_idx)

    def apply_score(self, score, b, h, q_idx, kv_idx):
        return self._score.apply_score(score, b, h, q_idx, kv_idx)


def _make_k_lds_layout(block_n, head_dim):
    base_layout = fx.make_layout((block_n, head_dim), (head_dim, 1))
    if const_expr(head_dim == 128):
        k_swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))
        return fx.make_composed_layout(k_swizzle, base_layout)
    return base_layout


def _build_mod(mask_type, score_type, mask_window=0, score_alibi_slope=0.0):
    _mask = {
        MASK_NONE: FlexMod(),
        MASK_CAUSAL: CausalMask(),
        MASK_SLIDING_WINDOW: SlidingWindowMask(mask_window),
    }[mask_type]
    _score = {
        SCORE_NONE: FlexMod(),
        SCORE_ALIBI: AlibiScore(score_alibi_slope),
    }[score_type]
    if _mask.has_mask or _score.has_score:
        return CompositeMod(_score, _mask)
    return FlexMod()


# ── Param struct ──────────────────────────────────────────────────────────

@fx.struct
class FlexAttnParam16:
    dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]       # 32 (query rows per group)
    block_n: fx.Constexpr[int]       # 32 (KV tile rows)
    head_dim: fx.Constexpr[int]      # 128
    num_heads_q: fx.Constexpr[int]
    num_heads_kv: fx.Constexpr[int]
    num_groups: fx.Constexpr[int]    # independent query subtiles per WG
    mma_m_qk: fx.Constexpr[int]     # 16 for QK GEMM
    mma_k_qk: fx.Constexpr[int]     # 16 for QK GEMM
    mma_m_pv: fx.Constexpr[int]     # 32 for PV GEMM (hybrid)
    mma_k_pv: fx.Constexpr[int]     # 16 for PV GEMM
    group_threads: fx.Constexpr[int]
    block_threads: fx.Constexpr[int]
    gqa_group: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    n_kv_tiles: fx.Constexpr[int]
    mask_type: fx.Constexpr[int]
    score_type: fx.Constexpr[int]
    mask_window: fx.Constexpr[int]
    score_alibi_slope: fx.Constexpr[float]
    num_kv_splits: fx.Constexpr[int]


def make_flex_attn_param_16x16(
    seqlen_kv: int,
    dtype_id: int = FLEX_DTYPE_BF16,
    block_m: int = 32,
    block_n: int = 32,
    head_dim: int = 128,
    num_heads_q: int = 8,
    num_heads_kv: int = 8,
    num_groups: int = 8,
    num_kv_splits: int = 1,
    mask_type: int = MASK_NONE,
    score_type: int = SCORE_NONE,
    mask_window: int = 0,
    score_alibi_slope: float = 0.0,
) -> FlexAttnParam16:
    if dtype_id not in (FLEX_DTYPE_BF16, FLEX_DTYPE_FP16):
        raise ValueError(f"unsupported dtype_id={dtype_id}")
    if block_m != 32 or block_n != 32:
        raise ValueError("16x16 kernel requires block_m=block_n=32")
    if head_dim not in (64, 128):
        raise ValueError(f"unsupported head_dim={head_dim}")
    if num_heads_q % num_heads_kv != 0:
        raise ValueError("num_heads_q must be divisible by num_heads_kv (GQA)")
    if seqlen_kv % block_n != 0:
        raise ValueError(f"seqlen_kv ({seqlen_kv}) must be a multiple of block_n ({block_n})")
    if num_groups < 1:
        raise ValueError("num_groups must be >= 1")

    # m_waves=1, n_waves=1 per group (64 threads per group)
    # QK: 4x MFMA 16x16x16 tiles a 32x32 score block
    # PV: 32x32x16 MFMA (hybrid, reuse proven path)
    m_waves = 1
    n_waves = 1
    group_threads = m_waves * n_waves * GFX950_WAVE_SIZE  # 64
    block_threads = num_groups * group_threads

    _max_waves = 8
    if block_threads > _max_waves * GFX950_WAVE_SIZE:
        raise ValueError(
            f"block_threads ({block_threads}) exceeds {_max_waves} SIMDs/CU limit"
        )
    if block_threads < 4 * GFX950_WAVE_SIZE:
        raise ValueError("need >= 4 waves for stagger (num_groups >= 4)")

    return FlexAttnParam16(
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        num_groups=num_groups,
        mma_m_qk=16,
        mma_k_qk=16,
        mma_m_pv=32,
        mma_k_pv=16,
        group_threads=group_threads,
        block_threads=block_threads,
        gqa_group=num_heads_q // num_heads_kv,
        in_data_bytes=2,
        n_kv_tiles=seqlen_kv // block_n,
        mask_type=mask_type,
        score_type=score_type,
        mask_window=mask_window,
        score_alibi_slope=score_alibi_slope,
        num_kv_splits=num_kv_splits,
    )


def _elem_dtype(dtype_id):
    if dtype_id == FLEX_DTYPE_FP16:
        return fx.Float16
    return fx.BFloat16


def _hw_exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, fx.Float32(x).ir_value()))


def _permlane32_reduce(x, mode):
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import arith as _arith
    from flydsl._mlir.dialects import llvm
    v_i32 = fx.Int32(_arith.bitcast(T.i32, fx.Float32(x).ir_value()))
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, v_i32.ir_value(), v_i32.ir_value(), False, True)
    lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
    rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
    lhs = fx.Float32(_arith.bitcast(T.f32, lhs_i32))
    rhs = fx.Float32(_arith.bitcast(T.f32, rhs_i32))
    if mode == "max":
        return lhs.maximumf(rhs)
    else:
        return lhs.addf(rhs, fastmath=_FM)


def _mfma_acc(a, b, c, mma_atom):
    from flydsl._mlir.dialects import fly
    acc_ty = c.type
    return fly.mma_atom_call_ssa([acc_ty], mma_atom, a, b, c)


def _size_scalar(shape) -> int:
    s = fx.size(shape)
    if hasattr(s, "unpack"):
        return s.unpack()
    if hasattr(s, "is_static") and s.is_static:
        v = s.to_py_value()
        if isinstance(v, tuple):
            return int(v[0]) if len(v) == 1 else int(v)
        return int(v)
    raise TypeError(f"cannot get static size from {type(s)!r}")


def _to_elem(val, elem_ty):
    if hasattr(val, "to"):
        return val.to(elem_ty)
    return fx.Float32(val).to(elem_ty)


def _shuffle_xor_f32(val_f32, lane_mask):
    """Cross-lane f32 exchange via gpu.shuffle XOR."""
    from flydsl._mlir.dialects import arith as _arith
    v_i32 = fx.Int32(_arith.bitcast(T.i32, fx.Float32(val_f32).ir_value()))
    shuffled_ir = v_i32.shuffle_xor(lane_mask, GFX950_WAVE_SIZE)
    if hasattr(shuffled_ir, "ir_value"):
        shuffled_ir = shuffled_ir.ir_value()
    return fx.Float32(_arith.bitcast(T.f32, shuffled_ir))


def _reduce_16x16_max(val, lane_mask=16):
    """Reduce max across 4 k-groups of a 16x16 MFMA C fragment.

    Step 1: shuffle_xor(16) combines groups 0↔1 and 2↔3.
    Step 2: permlane32_swap combines the two halves (0,1) ↔ (2,3).
    """
    other = _shuffle_xor_f32(val, lane_mask)
    val = fx.Float32(val).maximumf(other)
    val = _permlane32_reduce(val, "max")
    return val


def _reduce_16x16_sum(val, lane_mask=16):
    """Reduce sum across 4 k-groups of a 16x16 MFMA C fragment."""
    other = _shuffle_xor_f32(val, lane_mask)
    val = fx.Float32(val).addf(other, fastmath=_FM)
    val = _permlane32_reduce(val, "sum")
    return val


# ── Kernel ────────────────────────────────────────────────────────────────

@flyc.kernel
def flex_attn_16x16_kernel(
    o: fx.Tensor,
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    seqlen_q: fx.Int32,
    seqlen_kv: fx.Int32,
    num_batches: fx.Int32,
    scale: fx.Float32,
    tiled_mma_qk: fx.TiledMma,
    tiled_mma_pv: fx.TiledMma,
    param: FlexAttnParam16,
):
    block_m = param.block_m       # 32
    block_n = param.block_n       # 32
    head_dim = param.head_dim     # 128
    elem_dtype = _elem_dtype(param.dtype_id)

    tid = fx.thread_idx.x
    num_groups = param.num_groups
    group_threads = param.group_threads  # 64
    group = tid // group_threads
    local_tid = tid % group_threads

    q_tile = fx.block_idx.x
    h_idx = fx.block_idx.y
    b_idx = fx.block_idx.z
    kv_head = h_idx // param.gqa_group

    q_start = (q_tile * num_groups + group) * block_m

    n_kv_tiles = param.n_kv_tiles

    # ── LDS layout: K/V double-buffered (shared across groups) ────────────
    kv_tile_elems = block_n * head_dim
    _v_subtile_elems = block_n * 32
    _v_step_elems = _v_subtile_elems // 4
    _lds_ring_slots = 2

    @fx.struct
    class SharedStorage:
        k_lds_0: fx.Array[elem_dtype, kv_tile_elems, 16]
        k_lds_1: fx.Array[elem_dtype, kv_tile_elems, 16]
        v_lds_0: fx.Array[elem_dtype, kv_tile_elems, 16]
        v_lds_1: fx.Array[elem_dtype, kv_tile_elems, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    sK_ptr = [storage.k_lds_0.peek().ptr, storage.k_lds_1.peek().ptr]
    sV_ptr = [storage.v_lds_0.peek().ptr, storage.v_lds_1.peek().ptr]

    _k_base_layout = _make_k_lds_layout(block_n, head_dim)
    sK = [fx.make_view(sK_ptr[i], _k_base_layout) for i in range_constexpr(_lds_ring_slots)]

    # ── BSHD tensor views ─────────────────────────────────────────────────
    hq = param.num_heads_q
    hkv = param.num_heads_kv
    q_off = b_idx * seqlen_q * hq * head_dim + h_idx * head_dim + q_start * hq * head_dim
    o_off = q_off
    k_off = b_idx * seqlen_kv * hkv * head_dim + kv_head * head_dim
    v_off = b_idx * seqlen_kv * hkv * head_dim + kv_head * head_dim

    _q_total_bytes = num_batches * seqlen_q * fx.Int32(hq * head_dim * param.in_data_bytes)
    q_it = _make_buffer_ptr(
        fx.recast_iter(elem_dtype, fx.get_iter(q)),
        num_records_bytes=_q_total_bytes,
    )
    gQ = fx.make_view(q_it + fx.Int32(q_off), fx.make_layout((block_m, head_dim), (hq * head_dim, 1)))

    # ── Q load into GEMM B fragment (K=A, Q=B for QK) ────────────────────
    # QK uses 32x32x16 tiled_mma for fragment creation, but MFMA calls are 16x16x16.
    # We load Q into the 32x32 B fragment layout, then slice for 16x16 MFMAs.
    thr_pv = tiled_mma_pv.thr_slice(local_tid)

    ca = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    uca = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)

    Vec = fx.Vector

    # ── Q load: raw per-element load for 16x16 MFMA B operands ───────────
    # 16x16 B layout: thread t → query_row = t%16, K_group = t//16 (0..3)
    # Value v → K = K_group*4 + v (v ∈ 0..3)
    # Per K-step (ki): 4 bf16 at D positions ki*16 + K_group*4 + {0,1,2,3}
    _t0 = local_tid % 16
    _t1 = local_tid // 16   # 0..3
    _n_k_steps = head_dim // 16   # 8 for D=128

    # Load Q into registers: Q_lo (query rows 0-15) and Q_hi (query rows 16-31)
    # Each is a list of _n_k_steps v4bf16 vectors.
    q_it = _make_buffer_ptr(
        fx.recast_iter(elem_dtype, fx.get_iter(q)),
        num_records_bytes=num_batches * seqlen_q * fx.Int32(hq * head_dim * param.in_data_bytes),
    )
    _q_base = fx.Int32(q_off)  # base offset into Q tensor
    _q_row_stride = fx.Int32(hq * head_dim)

    frag_Q_lo = [None] * _n_k_steps  # v4bf16 per ki, for query rows 0-15
    frag_Q_hi = [None] * _n_k_steps  # v4bf16 per ki, for query rows 16-31
    for ki in range_constexpr(_n_k_steps):
        d_base = ki * 16 + _t1 * 4  # D offset for this K-step and K-group
        # Q_lo: query row = _t0 (0..15)
        q_lo_off = _q_base + fx.Int32(_t0) * _q_row_stride + fx.Int32(d_base)
        lo_elems = [_to_elem(fx.ptr_load(q_it + fx.Int32(q_lo_off + fx.Int32(v))), elem_dtype)
                    for v in range_constexpr(4)]
        frag_Q_lo[ki] = Vec.from_elements(lo_elems, elem_dtype).ir_value()
        # Q_hi: query row = 16 + _t0
        q_hi_off = _q_base + fx.Int32(16 + _t0) * _q_row_stride + fx.Int32(d_base)
        hi_elems = [_to_elem(fx.ptr_load(q_it + fx.Int32(q_hi_off + fx.Int32(v))), elem_dtype)
                    for v in range_constexpr(4)]
        frag_Q_hi[ki] = Vec.from_elements(hi_elems, elem_dtype).ir_value()

    # Pre-scale Q
    for ki in range_constexpr(_n_k_steps):
        lo_v = Vec(frag_Q_lo[ki])
        hi_v = Vec(frag_Q_hi[ki])
        lo_scaled = [_to_elem(_to_elem(lo_v[v], fx.Float32) * scale, elem_dtype)
                     for v in range_constexpr(4)]
        hi_scaled = [_to_elem(_to_elem(hi_v[v], fx.Float32) * scale, elem_dtype)
                     for v in range_constexpr(4)]
        frag_Q_lo[ki] = Vec.from_elements(lo_scaled, elem_dtype).ir_value()
        frag_Q_hi[ki] = Vec.from_elements(hi_scaled, elem_dtype).ir_value()

    _n_d_chunks = head_dim // 32
    o_accs_init = [Vec.filled(16, 0.0, fx.Float32).ir_value()
                   for _ in range_constexpr(_n_d_chunks)]

    npair = 1
    scale_log2e = fx.Float32(_LOG2E)

    _M_NEG_FLOOR_SCALED = -60.0 * _LOG2E
    m_i = [fx.Float32(_M_NEG_FLOOR_SCALED) for _ in range_constexpr(npair)]
    l_i = [fx.Float32(0.0) for _ in range_constexpr(npair)]

    # ── K LDS read: 64-bit vectorized for 16x16 MFMA A operands ────────────
    # 16x16 A layout: thread t → score_row = t%16, K_group = t//16 (0..3)
    # Per K-step (ki): v4bf16 at D positions ki*16 + K_group*4 + {0,1,2,3}
    # The 4 consecutive D elements within a K-group are contiguous in swizzled LDS
    # because the XOR swizzle(3,3,3) operates on bits [3:5] and d_base is 4-aligned.
    _k_lds_copy_atom = fx.make_copy_atom(fx.UniversalCopy64b(), elem_dtype)

    # Pre-compute LDS offsets for K_lo and K_hi for all ki panels.
    # Each ki has one v4bf16 for K_lo (rows 0-15) and one for K_hi (rows 16-31).
    _k_lo_offsets = []
    _k_hi_offsets = []
    for ki in range_constexpr(_n_k_steps):
        d_base = ki * 16 + _t1 * 4
        lo_idx = fx.get_scalar(fx.crd2idx(
            fx.make_int_tuple((_t0, d_base)), _k_base_layout))
        hi_idx = fx.get_scalar(fx.crd2idx(
            fx.make_int_tuple((16 + _t0, d_base)), _k_base_layout))
        _k_lo_offsets.append(lo_idx)
        _k_hi_offsets.append(hi_idx)

    # Pre-allocate K fragment storage for double-buffering.
    frag_K_lo = [[None] * _n_k_steps for _ in range_constexpr(_lds_ring_slots)]
    frag_K_hi = [[None] * _n_k_steps for _ in range_constexpr(_lds_ring_slots)]

    def read_k_work(slot):
        """Read all K panels from LDS into frag_K for the given slot."""
        base = sK_ptr[slot]
        for ki in range_constexpr(_n_k_steps):
            _k_copy_layout = fx.make_layout(4, 1)
            src_lo = fx.make_view(
                fx.add_offset(base, fx.make_int_tuple(_k_lo_offsets[ki])),
                _k_copy_layout,
            )
            dst_lo = fx.make_rmem_tensor(_k_copy_layout, elem_dtype)
            fx.copy(_k_lds_copy_atom, src_lo, dst_lo)
            frag_K_lo[slot][ki] = dst_lo.load()
            src_hi = fx.make_view(
                fx.add_offset(base, fx.make_int_tuple(_k_hi_offsets[ki])),
                _k_copy_layout,
            )
            dst_hi = fx.make_rmem_tensor(_k_copy_layout, elem_dtype)
            fx.copy(_k_lds_copy_atom, src_hi, dst_hi)
            frag_K_hi[slot][ki] = dst_hi.load()
        return []

    _k_iters = _n_k_steps

    # ── V LDS read: scalar reads for 64 threads ──────────────────────────
    # 32x32 MFMA A (V) layout: Row(D) = t0_32, Col(score) = t1_32*8 + v
    # v_lo: scores 0-15, v_hi: scores 16-31
    # V LDS step format: step = dc*4 + score_row//8, within step: row*32 + d_col
    _t0_32_v = local_tid % 32
    _t1_32_v = local_tid // 32

    def _make_read_v(slot):
        _slot_v_ptr = sV_ptr[slot]
        def _read():
            v_lo_out = [None] * _n_d_chunks
            v_hi_out = [None] * _n_d_chunks
            d_row = _t0_32_v
            score_base_lo = _t1_32_v * 8
            score_base_hi = 16 + _t1_32_v * 8
            for dc in range_constexpr(_n_d_chunks):
                lo_elems = [None] * 8
                hi_elems = [None] * 8
                for v in range_constexpr(8):
                    score_lo = score_base_lo + v
                    score_hi = score_base_hi + v
                    d_col = d_row
                    step_lo = dc * 4 + score_lo // 8
                    row_lo = score_lo % 8
                    off_lo = step_lo * _v_step_elems + row_lo * 32 + d_col
                    lo_elems[v] = _to_elem(fx.ptr_load(fx.add_offset(_slot_v_ptr, fx.make_int_tuple(off_lo))), elem_dtype)
                    step_hi = dc * 4 + score_hi // 8
                    row_hi = score_hi % 8
                    off_hi = step_hi * _v_step_elems + row_hi * 32 + d_col
                    hi_elems[v] = _to_elem(fx.ptr_load(fx.add_offset(_slot_v_ptr, fx.make_int_tuple(off_hi))), elem_dtype)
                v_lo_out[dc] = Vec.from_elements(lo_elems, elem_dtype).ir_value()
                v_hi_out[dc] = Vec.from_elements(hi_elems, elem_dtype).ir_value()
            return v_lo_out, v_hi_out
        return _read

    read_v_slot = [_make_read_v(i) for i in range_constexpr(_lds_ring_slots)]

    # ── DMA: global → LDS ─────────────────────────────────────────────────
    block_threads = param.block_threads
    _dma_bytes = GFX950_DMA_BYTES
    _kv_tile_bytes = kv_tile_elems * param.in_data_bytes
    _dma_ops_per_thread = _kv_tile_bytes // (block_threads * _dma_bytes)
    dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    _k_row_stride_bytes = hkv * head_dim * param.in_data_bytes
    _k_row_bytes = head_dim * param.in_data_bytes
    _v_subtile_row_bytes = 32 * param.in_data_bytes
    _v_row_stride_bytes = hkv * head_dim * param.in_data_bytes

    gK_flat = fx.rocdl.make_buffer_tensor(
        fx.Tensor(fx.make_view(fx.recast_iter(fx.Int8, fx.get_iter(k)), fx.make_layout(0x7FFFFFFF, 1))),
        max_size=True,
    )
    gV_flat = fx.rocdl.make_buffer_tensor(
        fx.Tensor(fx.make_view(fx.recast_iter(fx.Int8, fx.get_iter(v)), fx.make_layout(0x7FFFFFFF, 1))),
        max_size=True,
    )
    k_div = fx.logical_divide(gK_flat, fx.make_layout(1, 1))
    v_div = fx.logical_divide(gV_flat, fx.make_layout(1, 1))
    sK_i8 = [fx.recast_iter(fx.Int8, sK_ptr[i]) for i in range_constexpr(_lds_ring_slots)]
    sV_i8 = [fx.recast_iter(fx.Int8, sV_ptr[i]) for i in range_constexpr(_lds_ring_slots)]

    def _k_swizzled_col(tile_row, tile_col_elem):
        elem_off = fx.get_scalar(fx.crd2idx(
            fx.make_int_tuple((tile_row, tile_col_elem)), _k_base_layout
        ))
        return elem_off % head_dim

    def load_kv(tile_idx, slot, ops=_dma_ops_per_thread, op_offset=0):
        wave_off = rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * _dma_bytes))
        _step_bytes = block_threads * _dma_bytes
        # K DMA
        k_global_base = k_off * param.in_data_bytes + tile_idx * block_n * _k_row_stride_bytes
        lds_k = fx.add_offset(sK_i8[slot], wave_off + op_offset * _step_bytes)
        for i in range_constexpr(ops):
            if const_expr(i > 0):
                lds_k = fx.add_offset(lds_k, _step_bytes)
            flat_byte = (op_offset + i) * block_threads * _dma_bytes + tid * _dma_bytes
            tile_row = flat_byte // _k_row_bytes
            tile_col_elem = (flat_byte % _k_row_bytes) // param.in_data_bytes
            swiz_col = _k_swizzled_col(tile_row, tile_col_elem)
            gmem_byte = k_global_base + tile_row * _k_row_stride_bytes + swiz_col * param.in_data_bytes
            fx.copy(dma_atom, fx.slice(k_div, (None, fx.Int32(gmem_byte))),
                    fx.make_view(lds_k, fx.make_layout(1, 1)))
        # V DMA
        v_global_base = v_off * param.in_data_bytes + tile_idx * fx.Int32(block_n * _v_row_stride_bytes)
        for i in range_constexpr(ops):
            flat_byte = (op_offset + i) * block_threads * _dma_bytes + tid * _dma_bytes
            tile_row = flat_byte // _v_subtile_row_bytes
            tile_col_byte = flat_byte % _v_subtile_row_bytes
            dc = tile_row // block_n
            score_row = tile_row % block_n
            v_step = dc * 4 + score_row // 8
            row_in_step = score_row % 8
            lds_byte = v_step * _v_step_elems * param.in_data_bytes + row_in_step * _v_subtile_row_bytes + tile_col_byte
            lds_v = fx.add_offset(sV_i8[slot], lds_byte)
            d_global_byte = dc * 32 * param.in_data_bytes + tile_col_byte
            gmem_byte = fx.Int32(v_global_base) + score_row * fx.Int32(_v_row_stride_bytes) + d_global_byte
            fx.copy(dma_atom, fx.slice(v_div, (None, fx.Int32(gmem_byte))),
                    fx.make_view(lds_v, fx.make_layout(1, 1)))
        return []

    # ── QK GEMM: 4x MFMA 16x16x16 per K-step ────────────────────────────
    # Uses raw MFMA atoms. K=A (score rows), Q=B (query cols).
    # 4 sub-tiles: (K_lo×Q_lo, K_hi×Q_lo, K_lo×Q_hi, K_hi×Q_hi)
    # We reuse K fragments loaded via 32x32 tiled_copy (same data, different slicing).
    _qk_mma_atom_16 = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, elem_dtype))
    _pv_mma_atom_32 = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, elem_dtype))

    _is_bf16 = int(param.dtype_id) == FLEX_DTYPE_BF16

    # For the QK GEMM, we need to extract 16x16 A (K) and B (Q) operands from
    # the 32x32 fragments that the layout API loaded.
    #
    # 32x32 A (K) fragment: Thr(32,2) × Val(8), stride Thr(1,256) × Val(32)
    # Per ki panel: 8 bf16 values per thread.
    # To extract 16x16 A sub-blocks:
    #   K_lo (score rows 0-15): threads 0-15 (t%32 < 16), first 4 values (v 0-3)
    #   K_hi (score rows 16-31): same threads, but from the upper half
    #
    # 32x32 B (Q) fragment: same layout
    #   Q_lo (query 0-15): threads 0-15
    #   Q_hi (query 16-31): threads 16-31
    #
    # The 16x16 MFMA A operand is v4bf16 (4 values), matching the first/second
    # half of the 32x32's v8bf16.

    # K global memory iterator for direct reads (bypassing LDS for correctness)
    k_it = _make_buffer_ptr(
        fx.recast_iter(elem_dtype, fx.get_iter(k)),
        num_records_bytes=num_batches * seqlen_kv * fx.Int32(hkv * head_dim * param.in_data_bytes),
    )
    _k_global_base = fx.Int32(k_off)

    def _read_k_global_16x16(kv_tile_idx, ki):
        """Read K_lo and K_hi directly from global memory (debug path)."""
        _k_row_stride_g = fx.Int32(hkv * head_dim)
        d_base = ki * 16 + _t1 * 4
        lo_elems = [None] * 4
        hi_elems = [None] * 4
        for v in range_constexpr(4):
            d_col = d_base + v
            # K_lo: score row = kv_tile * block_n + _t0
            k_row_lo = kv_tile_idx * fx.Int32(block_n) + fx.Int32(_t0)
            k_off_lo = _k_global_base + k_row_lo * _k_row_stride_g + fx.Int32(d_col)
            lo_elems[v] = _to_elem(fx.ptr_load(k_it + fx.Int32(k_off_lo)), elem_dtype)
            # K_hi: score row = kv_tile * block_n + 16 + _t0
            k_row_hi = kv_tile_idx * fx.Int32(block_n) + fx.Int32(16 + _t0)
            k_off_hi = _k_global_base + k_row_hi * _k_row_stride_g + fx.Int32(d_col)
            hi_elems[v] = _to_elem(fx.ptr_load(k_it + fx.Int32(k_off_hi)), elem_dtype)
        k_lo = Vec.from_elements(lo_elems, elem_dtype).ir_value()
        k_hi = Vec.from_elements(hi_elems, elem_dtype).ir_value()
        return k_lo, k_hi

    def gemm1_qk_16x16(slot):
        """QK GEMM using 4x MFMA 16x16x16 per K-step.

        Uses pre-loaded K fragments from LDS (read_k_work must be called first).
        Returns 4 accumulators (v4f32 each) for the 4 sub-tiles.
        """
        c00 = Vec.filled(4, 0.0, fx.Float32).ir_value()
        c01 = Vec.filled(4, 0.0, fx.Float32).ir_value()
        c10 = Vec.filled(4, 0.0, fx.Float32).ir_value()
        c11 = Vec.filled(4, 0.0, fx.Float32).ir_value()

        for ki in range_constexpr(_n_k_steps):
            k_lo = frag_K_lo[slot][ki]
            k_hi = frag_K_hi[slot][ki]
            q_lo = frag_Q_lo[ki]
            q_hi = frag_Q_hi[ki]

            c00 = _mfma_acc(k_lo, q_lo, c00, _qk_mma_atom_16)
            c01 = _mfma_acc(k_lo, q_hi, c01, _qk_mma_atom_16)
            c10 = _mfma_acc(k_hi, q_lo, c10, _qk_mma_atom_16)
            c11 = _mfma_acc(k_hi, q_hi, c11, _qk_mma_atom_16)

        return c00, c01, c10, c11

    # ── Flex mod application ──────────────────────────────────────────────
    flex_mod = _build_mod(int(param.mask_type), int(param.score_type),
                          int(param.mask_window), float(param.score_alibi_slope))
    mod_has_score = flex_mod.has_score
    mod_has_mask = flex_mod.has_mask
    _mod_apply_score = flex_mod.apply_score
    _mod_apply_mask = flex_mod.apply_mask
    b_i32 = fx.Int32(arith.index_cast(T.i32, b_idx))
    h_i32 = fx.Int32(arith.index_cast(T.i32, h_idx))

    # For 16x16 C layout: score = t1*4 + v0, query = t0
    # Per tile, thread has 4 score values. Tile-local coords.
    # Global score offset depends on which tile (lo=0, hi=+16).
    # Global query offset depends on which tile (lo=0, hi=+16).
    _q_idx_lo = fx.Int32(arith.index_cast(T.i32, q_start)) + fx.Int32(_t0)
    _q_idx_hi = _q_idx_lo + fx.Int32(16)
    _score_local = [fx.Int32(_t1 * 4 + v0) for v0 in range(4)]

    def apply_mods_16x16(c00, c01, c10, c11, kv_tile_idx):
        """Apply score/mask mods to the 4 sub-tiles."""
        kv_base = kv_tile_idx * fx.Int32(block_n)
        c00_e = [Vec(c00)[i] for i in range_constexpr(4)]
        c01_e = [Vec(c01)[i] for i in range_constexpr(4)]
        c10_e = [Vec(c10)[i] for i in range_constexpr(4)]
        c11_e = [Vec(c11)[i] for i in range_constexpr(4)]
        if const_expr(mod_has_score or mod_has_mask):
            for v0 in range_constexpr(4):
                kv_lo = kv_base + _score_local[v0]
                kv_hi = kv_base + fx.Int32(16) + _score_local[v0]
                if const_expr(mod_has_score):
                    c00_e[v0] = _mod_apply_score(c00_e[v0], b_i32, h_i32, _q_idx_lo, kv_lo)
                    c01_e[v0] = _mod_apply_score(c01_e[v0], b_i32, h_i32, _q_idx_hi, kv_lo)
                    c10_e[v0] = _mod_apply_score(c10_e[v0], b_i32, h_i32, _q_idx_lo, kv_hi)
                    c11_e[v0] = _mod_apply_score(c11_e[v0], b_i32, h_i32, _q_idx_hi, kv_hi)
                if const_expr(mod_has_mask):
                    c00_e[v0] = _mod_apply_mask(c00_e[v0], _q_idx_lo, kv_lo)
                    c01_e[v0] = _mod_apply_mask(c01_e[v0], _q_idx_hi, kv_lo)
                    c10_e[v0] = _mod_apply_mask(c10_e[v0], _q_idx_lo, kv_hi)
                    c11_e[v0] = _mod_apply_mask(c11_e[v0], _q_idx_hi, kv_hi)
        return (Vec.from_elements(c00_e, fx.Float32).ir_value(),
                Vec.from_elements(c01_e, fx.Float32).ir_value(),
                Vec.from_elements(c10_e, fx.Float32).ir_value(),
                Vec.from_elements(c11_e, fx.Float32).ir_value())

    # ── Softmax ───────────────────────────────────────────────────────────
    # Each thread has 2 queries: q_lo (t0) and q_hi (16+t0).
    # For q_lo: scores from tiles c00 (lo scores) + c10 (hi scores) = 8 values.
    # For q_hi: scores from tiles c01 (lo scores) + c11 (hi scores) = 8 values.
    # Reduce across t1 groups via shuffle_xor(16) + permlane32_swap.
    # After reduction, each thread has the per-query max/sum for both queries.

    # Per-thread flag: does this thread's PV query (t%32) correspond to query_hi?
    _is_hi_query = fx.Int32(local_tid % 32) >= fx.Int32(16)

    def softmax_start_16x16(c00, c01, c10, c11, m_i_in):
        """Compute per-query scaled scores and running max.

        m_i_in[0] is this thread's running max (already selected per PV-query).
        Returns: (corr_scalar, s_lo[8], s_hi[8], m_new)
        where m_new is the per-PV-query new max.
        """
        _sl2e = scale_log2e

        c00_v = Vec(c00)
        c10_v = Vec(c10)
        s_lo = [fx.Float32(c00_v[v0]) * _sl2e for v0 in range_constexpr(4)]
        s_lo += [fx.Float32(c10_v[v0]) * _sl2e for v0 in range_constexpr(4)]

        c01_v = Vec(c01)
        c11_v = Vec(c11)
        s_hi = [fx.Float32(c01_v[v0]) * _sl2e for v0 in range_constexpr(4)]
        s_hi += [fx.Float32(c11_v[v0]) * _sl2e for v0 in range_constexpr(4)]

        # Local max per query half
        tile_max_lo = s_lo[0]
        for i in range_constexpr(1, 8):
            tile_max_lo = tile_max_lo.maximumf(s_lo[i])
        tile_max_hi = s_hi[0]
        for i in range_constexpr(1, 8):
            tile_max_hi = tile_max_hi.maximumf(s_hi[i])

        # Cross-lane reduction (4 k-groups)
        tile_max_lo = _reduce_16x16_max(tile_max_lo)
        tile_max_hi = _reduce_16x16_max(tile_max_hi)

        # Select the max for this thread's PV query
        tile_max = _is_hi_query.select(tile_max_hi, tile_max_lo)
        m_new = m_i_in[0].maximumf(tile_max)
        corr_scalar = _hw_exp2(m_i_in[0] - m_new)

        return corr_scalar, s_lo, s_hi, m_new

    def softmax_finish_16x16(s_lo, s_hi, m_new, l_i_in, o_accs_in, corr_scalar):
        """Compute P values and update accumulators.

        Returns: (p_lo[8], p_hi[8], l_i_out, o_accs_out)
        """
        # m_new is per-PV-query, but for P computation we need the query-specific max.
        # Threads with PV query_lo use m_new (which came from tile_max_lo path).
        # Threads with PV query_hi use m_new (from tile_max_hi path).
        # Since m_new is already per-thread selected, we can use it directly.
        # But for P value computation, we use the QUERY-specific max for exp2:
        # p_lo values are for query_lo threads, p_hi for query_hi threads.
        # However, ALL threads compute ALL 16 P values (both lo and hi queries)
        # because the PV B-operand packing needs values from both query halves.

        # Compute P for BOTH query halves using their respective maxes.
        # We need both m_lo and m_hi. Reconstruct them from the per-thread m_new.
        # Thread with query_lo has m_new = m_lo; thread with query_hi has m_new = m_hi.
        # To get the other query's max, shuffle across the query boundary.
        # Threads 0-15 and 32-47 have query_lo; threads 16-31 and 48-63 have query_hi.
        # XOR 16 swaps between lo↔hi query threads within each half-wave.
        m_other = _shuffle_xor_f32(m_new, 16)
        m_lo = _is_hi_query.select(m_other, m_new)
        m_hi = _is_hi_query.select(m_new, m_other)

        p_lo = [_hw_exp2(s_lo[i] - m_lo) for i in range_constexpr(8)]
        p_hi = [_hw_exp2(s_hi[i] - m_hi) for i in range_constexpr(8)]

        # Per-query sum
        local_sum_lo = p_lo[0]
        for i in range_constexpr(1, 8):
            local_sum_lo = local_sum_lo.addf(p_lo[i], fastmath=_FM)
        local_sum_hi = p_hi[0]
        for i in range_constexpr(1, 8):
            local_sum_hi = local_sum_hi.addf(p_hi[i], fastmath=_FM)

        local_sum_lo = _reduce_16x16_sum(local_sum_lo)
        local_sum_hi = _reduce_16x16_sum(local_sum_hi)

        # Select the sum for this thread's PV query
        local_sum = _is_hi_query.select(local_sum_hi, local_sum_lo)
        l_new = l_i_in[0] * corr_scalar + local_sum

        # Rescale O accumulators
        scale_vec = Vec.from_elements([corr_scalar], fx.Float32).broadcast_to(16)
        o_accs_out = []
        for dc in range_constexpr(_n_d_chunks):
            o_accs_out.append((Vec(o_accs_in[dc]) * scale_vec).ir_value())

        l_i_out = [l_new]

        return p_lo, p_hi, l_i_out, o_accs_out

    # ── PV GEMM: register-only C→B conversion via shuffle_xor ──────────────
    # Convert 16×16 QK C-fragment P values to 32×32 PV B-operand format without LDS.
    # Each thread has p_lo[8] (query_lo scores) and p_hi[8] (query_hi scores).
    # The MFMA B operand needs 8 consecutive scores at each thread's PV query.
    # shuffle_xor(16) exchanges values between _t1 groups 0↔1 and 2↔3.
    _t0_32 = local_tid % 32
    _t1_32 = local_tid // 32  # 0 or 1
    _is_odd_t1 = fx.Int32(_t1 & 1) != fx.Int32(0)

    def _pack_8_f32_to_v8elem(vals_8):
        if const_expr(_is_bf16):
            pairs = []
            for j in range_constexpr(4):
                pairs.append(rocdl.cvt_pk_bf16_f32(vals_8[j * 2], vals_8[j * 2 + 1]))
            return Vec.from_elements(pairs, fx.Int32).bitcast(fx.BFloat16).ir_value()
        else:
            elems = []
            for j in range_constexpr(8):
                elems.append(fx.Float32(vals_8[j]).to(elem_dtype))
            return Vec.from_elements(elems, elem_dtype).ir_value()

    def pv_gemm_register(p_lo_vals, p_hi_vals, v_lo_regs, v_hi_regs, o_accs):
        """PV GEMM via register-only C→B conversion.

        p_lo[8]: P values at query=_t0 for scores {_t1*4..+3, 16+_t1*4..+3}
        p_hi[8]: P values at query=16+_t0 for same score pattern.
        Repackaged into 32x32 MFMA B operands via shuffle_xor(16).
        """
        # Shuffle all P values with partner thread (_t1 XOR 1).
        shuf_p_lo = [_shuffle_xor_f32(p_lo_vals[i], 16) for i in range_constexpr(8)]
        shuf_p_hi = [_shuffle_xor_f32(p_hi_vals[i], 16) for i in range_constexpr(8)]

        # Even _t1 (0,2): use p_lo as "own", shuffled p_lo as "partner"
        #   p_b_lo = [own[0..3], partner[0..3]]  (scores at this thread's query)
        #   p_b_hi = [own[4..7], partner[4..7]]
        # Odd _t1 (1,3): use p_hi as "own", shuffled p_hi as "partner"
        #   p_b_lo = [partner[0..3], own[0..3]]  (swapped order)
        #   p_b_hi = [partner[4..7], own[4..7]]
        p_b_lo_f32 = [None] * 8
        p_b_hi_f32 = [None] * 8
        for i in range_constexpr(4):
            # First 4 of p_b_lo/hi
            p_b_lo_f32[i] = _is_odd_t1.select(shuf_p_hi[i], p_lo_vals[i])
            p_b_hi_f32[i] = _is_odd_t1.select(shuf_p_hi[4 + i], p_lo_vals[4 + i])
            # Second 4 of p_b_lo/hi
            p_b_lo_f32[4 + i] = _is_odd_t1.select(p_hi_vals[i], shuf_p_lo[i])
            p_b_hi_f32[4 + i] = _is_odd_t1.select(p_hi_vals[4 + i], shuf_p_lo[4 + i])

        p_b_lo = _pack_8_f32_to_v8elem(p_b_lo_f32)
        p_b_hi = _pack_8_f32_to_v8elem(p_b_hi_f32)

        for dc in range_constexpr(_n_d_chunks):
            o_accs[dc] = _mfma_acc(v_lo_regs[dc], p_b_lo, o_accs[dc], _pv_mma_atom_32)
            o_accs[dc] = _mfma_acc(v_hi_regs[dc], p_b_hi, o_accs[dc], _pv_mma_atom_32)

    # ── Cluster sync ──────────────────────────────────────────────────────
    def dualwave_cluster_sync(cluster_index):
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

    # ── Stagger setup ─────────────────────────────────────────────────────
    _total_waves = block_threads // GFX950_WAVE_SIZE
    _enable_stagger = _total_waves >= 4
    _stagger_div = max(1, _total_waves // 2)
    _wave_id = fx.Int32(tid // GFX950_WAVE_SIZE)
    stagger_i32 = rocdl.readfirstlane(
        fx.Int32.ir_type,
        _wave_id // fx.Int32(_stagger_div),
    )

    # ── KV loop ───────────────────────────────────────────────────────────
    _q_min_wg = fx.Int32(arith.index_cast(T.i32, q_tile)) * fx.Int32(num_groups * block_m)
    _q_max_wg = _q_min_wg + fx.Int32(num_groups * block_m - 1)
    _kv_lo, _kv_hi = flex_mod.kv_range(_q_min_wg, _q_max_wg, n_kv_tiles, block_n)

    rocdl.s_barrier()
    rocdl.s_barrier()
    load_kv(_kv_lo, 0)
    rocdl.s_waitcnt(0)
    rocdl.s_barrier()

    if const_expr(_enable_stagger):
        rocdl.sched_barrier(0)
        _stagger_extra_barrier_if_one(stagger_i32)

    o_accs = o_accs_init

    _kv_last = _kv_hi - fx.Int32(1)

    def _safe_load_kv(tile_idx, slot):
        """Load KV, clamping tile index to avoid OOB global memory access."""
        clamped = fx.Int32(arith.minsi(fx.Int32(tile_idx).ir_value(), _kv_last.ir_value()))
        load_kv(clamped, slot)

    def _mask_scores(c00, c01, c10, c11, valid):
        """Apply valid mask to scores (replaces invalid with -inf)."""
        _neg_inf = fx.Float32(-1e9)
        c00_v, c01_v = fx.Vector(c00), fx.Vector(c01)
        c10_v, c11_v = fx.Vector(c10), fx.Vector(c11)
        c00 = fx.Vector.from_elements([valid.select(c00_v[i], _neg_inf) for i in range_constexpr(4)], fx.Float32).ir_value()
        c01 = fx.Vector.from_elements([valid.select(c01_v[i], _neg_inf) for i in range_constexpr(4)], fx.Float32).ir_value()
        c10 = fx.Vector.from_elements([valid.select(c10_v[i], _neg_inf) for i in range_constexpr(4)], fx.Float32).ir_value()
        c11 = fx.Vector.from_elements([valid.select(c11_v[i], _neg_inf) for i in range_constexpr(4)], fx.Float32).ir_value()
        return c00, c01, c10, c11

    def _qk_softmax_start(slot, kv_i32, m_i):
        """QK GEMM + mods + softmax_start. Returns (corr, s_lo, s_hi, m_new)."""
        c00, c01, c10, c11 = gemm1_qk_16x16(slot)
        if const_expr(mod_has_score or mod_has_mask):
            c00, c01, c10, c11 = apply_mods_16x16(c00, c01, c10, c11, kv_i32)
        corr, s_lo, s_hi, m_new = softmax_start_16x16(c00, c01, c10, c11, m_i)
        return corr, s_lo, s_hi, m_new

    def _qk_softmax_start_masked(slot, kv_i32, m_i, valid):
        """QK GEMM + mods + valid mask + softmax_start."""
        c00, c01, c10, c11 = gemm1_qk_16x16(slot)
        if const_expr(mod_has_score or mod_has_mask):
            c00, c01, c10, c11 = apply_mods_16x16(c00, c01, c10, c11, kv_i32)
        c00, c01, c10, c11 = _mask_scores(c00, c01, c10, c11, valid)
        corr, s_lo, s_hi, m_new = softmax_start_16x16(c00, c01, c10, c11, m_i)
        return corr, s_lo, s_hi, m_new

    def _finish_pv(s_lo, s_hi, m_i, l_i, o_accs, corr, v_lo, v_hi):
        """softmax_finish + register PV GEMM. Returns (l_i, o_accs)."""
        p_lo, p_hi, l_i, o_accs = softmax_finish_16x16(s_lo, s_hi, m_i[0], l_i, o_accs, corr)
        pv_gemm_register(p_lo, p_hi, v_lo, v_hi, o_accs)
        return l_i, o_accs

    # ── Overlapping softmax pipeline (4-cluster) ──────────────────────────

    def _do_tile_prologue(kv_i32, m_i, l_i, o_accs):
        """First pair: no deferred PV from previous."""
        odd_valid = (kv_i32 + fx.Int32(1)) < _kv_hi
        has_next = (kv_i32 + fx.Int32(2)) < _kv_hi
        # Cluster 0: mem tile 0
        rocdl.s_waitcnt(_WAIT_VMCNT_0)
        rocdl.s_barrier()
        read_k_work(0)
        v_lo_0, v_hi_0 = read_v_slot[0]()
        if odd_valid:
            _safe_load_kv(kv_i32 + fx.Int32(1), 1)
        rocdl.s_waitcnt(_WAIT_LGKMCNT_0)
        dualwave_cluster_sync(0)
        # Cluster 1: QK tile 0, no deferred PV
        corr_0, s_lo_0, s_hi_0, m_new = _qk_softmax_start(0, kv_i32, m_i)
        m_i = [m_new]
        dualwave_cluster_sync(1)
        # Cluster 2: mem tile 1
        rocdl.s_waitcnt(_WAIT_VMCNT_0)
        rocdl.s_barrier()
        read_k_work(1)
        v_lo_1, v_hi_1 = read_v_slot[1]()
        if has_next:
            _safe_load_kv(kv_i32 + fx.Int32(2), 0)
        rocdl.s_waitcnt(_WAIT_LGKMCNT_0)
        dualwave_cluster_sync(2)
        # Cluster 3: QK tile 1 + PV from tile 0
        l_i, o_accs = _finish_pv(s_lo_0, s_hi_0, m_i, l_i, o_accs, corr_0, v_lo_0, v_hi_0)
        corr_1, s_lo_1, s_hi_1, m_new = _qk_softmax_start_masked(1, kv_i32 + fx.Int32(1), m_i, odd_valid)
        m_i = [m_new]
        dualwave_cluster_sync(3)
        return (m_i, l_i, o_accs,
                s_lo_1, s_hi_1, corr_1, v_lo_1, v_hi_1, m_i)

    def _do_tile_main(kv_i32, m_i, l_i, o_accs,
                      s_lo_prev, s_hi_prev, corr_prev, v_lo_prev, v_hi_prev, m_i_prev):
        """Steady-state: has deferred PV, guaranteed valid odd + next."""
        # Cluster 0: mem tile 0
        rocdl.s_waitcnt(_WAIT_VMCNT_0)
        read_k_work(0)
        v_lo_0, v_hi_0 = read_v_slot[0]()
        _safe_load_kv(kv_i32 + fx.Int32(1), 1)
        rocdl.s_waitcnt(_WAIT_LGKMCNT_0)
        dualwave_cluster_sync(0)
        # Cluster 1: QK tile 0 + deferred PV from prev
        l_i, o_accs = _finish_pv(s_lo_prev, s_hi_prev, m_i_prev, l_i, o_accs, corr_prev, v_lo_prev, v_hi_prev)
        corr_0, s_lo_0, s_hi_0, m_new = _qk_softmax_start(0, kv_i32, m_i)
        m_i = [m_new]
        dualwave_cluster_sync(1)
        # Cluster 2: mem tile 1
        rocdl.s_waitcnt(_WAIT_VMCNT_0)
        read_k_work(1)
        v_lo_1, v_hi_1 = read_v_slot[1]()
        _safe_load_kv(kv_i32 + fx.Int32(2), 0)
        rocdl.s_waitcnt(_WAIT_LGKMCNT_0)
        dualwave_cluster_sync(2)
        # Cluster 3: QK tile 1 + PV from tile 0
        l_i, o_accs = _finish_pv(s_lo_0, s_hi_0, m_i, l_i, o_accs, corr_0, v_lo_0, v_hi_0)
        corr_1, s_lo_1, s_hi_1, m_new = _qk_softmax_start(1, kv_i32 + fx.Int32(1), m_i)
        m_i = [m_new]
        dualwave_cluster_sync(3)
        return (m_i, l_i, o_accs,
                s_lo_1, s_hi_1, corr_1, v_lo_1, v_hi_1, m_i)

    def _do_tile_epilogue(kv_i32, m_i, l_i, o_accs,
                          s_lo_prev, s_hi_prev, corr_prev, v_lo_prev, v_hi_prev, m_i_prev):
        """Last pair: has deferred PV, odd may be invalid, no next DMA."""
        odd_valid = (kv_i32 + fx.Int32(1)) < _kv_hi
        has_next = (kv_i32 + fx.Int32(2)) < _kv_hi
        # Cluster 0: mem tile 0
        rocdl.s_waitcnt(_WAIT_VMCNT_0)
        rocdl.s_barrier()
        read_k_work(0)
        v_lo_0, v_hi_0 = read_v_slot[0]()
        if odd_valid:
            _safe_load_kv(kv_i32 + fx.Int32(1), 1)
        rocdl.s_waitcnt(_WAIT_LGKMCNT_0)
        dualwave_cluster_sync(0)
        # Cluster 1: QK tile 0 + deferred PV from prev
        l_i, o_accs = _finish_pv(s_lo_prev, s_hi_prev, m_i_prev, l_i, o_accs, corr_prev, v_lo_prev, v_hi_prev)
        corr_0, s_lo_0, s_hi_0, m_new = _qk_softmax_start(0, kv_i32, m_i)
        m_i = [m_new]
        dualwave_cluster_sync(1)
        # Cluster 2: mem tile 1
        rocdl.s_waitcnt(_WAIT_VMCNT_0)
        rocdl.s_barrier()
        read_k_work(1)
        v_lo_1, v_hi_1 = read_v_slot[1]()
        if has_next:
            _safe_load_kv(kv_i32 + fx.Int32(2), 0)
        rocdl.s_waitcnt(_WAIT_LGKMCNT_0)
        dualwave_cluster_sync(2)
        # Cluster 3: QK tile 1 + PV from tile 0 + final deferred
        l_i, o_accs = _finish_pv(s_lo_0, s_hi_0, m_i, l_i, o_accs, corr_0, v_lo_0, v_hi_0)
        corr_1, s_lo_1, s_hi_1, m_new = _qk_softmax_start_masked(1, kv_i32 + fx.Int32(1), m_i, odd_valid)
        m_i = [m_new]
        # Flush final deferred PV
        l_i, o_accs = _finish_pv(s_lo_1, s_hi_1, m_i, l_i, o_accs, corr_1, v_lo_1, v_hi_1)
        dualwave_cluster_sync(3)
        return m_i, l_i, o_accs

    # ── KV loop with overlapping softmax ──────────────────────────────────
    _kv_range = _kv_hi - _kv_lo
    _kv_pairs = (_kv_range + fx.Int32(1)) // fx.Int32(2)
    _o = 2 * npair
    _n_s = 16  # 8 s_lo + 8 s_hi

    # Prologue: first pair
    (m_i, l_i, o_accs,
     s_lo_prev, s_hi_prev, corr_prev, v_lo_prev, v_hi_prev, m_i_prev) = (
        _do_tile_prologue(_kv_lo, m_i, l_i, o_accs)
    )

    # Main loop: pairs 1 .. _kv_pairs-2
    _main_count = _kv_pairs - fx.Int32(2)
    _sm_base = _o + _n_d_chunks
    _v_base = _sm_base + _n_s + 1
    _mi_prev_base = _v_base + 2 * _n_d_chunks

    init_args = (
        [m_i[r] for r in range_constexpr(npair)]
        + [l_i[r] for r in range_constexpr(npair)]
        + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
        + s_lo_prev + s_hi_prev + [corr_prev]
        + [v_lo_prev[dc] for dc in range_constexpr(_n_d_chunks)]
        + [v_hi_prev[dc] for dc in range_constexpr(_n_d_chunks)]
        + [m_i_prev[r] for r in range_constexpr(npair)]
    )
    loop_results = init_args

    for kv_mid, loop_args in range(
        fx.Int32(0),
        _main_count,
        fx.Int32(1),
        init=init_args,
    ):
        m_i = [loop_args[r] for r in range_constexpr(npair)]
        l_i = [loop_args[npair + r] for r in range_constexpr(npair)]
        o_accs = [loop_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
        s_lo_prev = [loop_args[_sm_base + i] for i in range_constexpr(8)]
        s_hi_prev = [loop_args[_sm_base + 8 + i] for i in range_constexpr(8)]
        corr_prev = loop_args[_sm_base + _n_s]
        v_lo_prev = [loop_args[_v_base + dc] for dc in range_constexpr(_n_d_chunks)]
        v_hi_prev = [loop_args[_v_base + _n_d_chunks + dc] for dc in range_constexpr(_n_d_chunks)]
        m_i_prev = [loop_args[_mi_prev_base + r] for r in range_constexpr(npair)]

        kv_even = _kv_lo + (fx.Int32(arith.index_cast(T.i32, kv_mid)) + fx.Int32(1)) * fx.Int32(2)
        (m_i, l_i, o_accs,
         s_lo_prev, s_hi_prev, corr_prev, v_lo_prev, v_hi_prev, m_i_prev) = (
            _do_tile_main(
                kv_even, m_i, l_i, o_accs,
                s_lo_prev, s_hi_prev, corr_prev, v_lo_prev, v_hi_prev, m_i_prev,
            )
        )

        loop_results = yield (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + s_lo_prev + s_hi_prev + [corr_prev]
            + [v_lo_prev[dc] for dc in range_constexpr(_n_d_chunks)]
            + [v_hi_prev[dc] for dc in range_constexpr(_n_d_chunks)]
            + [m_i_prev[r] for r in range_constexpr(npair)]
        )

    m_i = [loop_results[r] for r in range_constexpr(npair)]
    l_i = [loop_results[npair + r] for r in range_constexpr(npair)]
    o_accs = [loop_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]
    s_lo_prev = [loop_results[_sm_base + i] for i in range_constexpr(8)]
    s_hi_prev = [loop_results[_sm_base + 8 + i] for i in range_constexpr(8)]
    corr_prev = loop_results[_sm_base + _n_s]
    v_lo_prev = [loop_results[_v_base + dc] for dc in range_constexpr(_n_d_chunks)]
    v_hi_prev = [loop_results[_v_base + _n_d_chunks + dc] for dc in range_constexpr(_n_d_chunks)]
    m_i_prev = [loop_results[_mi_prev_base + r] for r in range_constexpr(npair)]

    # Epilogue: last pair, flush deferred PV
    _kv_last_pair = _kv_lo + (_kv_pairs - fx.Int32(1)) * fx.Int32(2)
    m_i, l_i, o_accs = _do_tile_epilogue(
        _kv_last_pair, m_i, l_i, o_accs,
        s_lo_prev, s_hi_prev, corr_prev, v_lo_prev, v_hi_prev, m_i_prev,
    )

    if const_expr(_enable_stagger):
        _stagger_extra_barrier_if_zero(stagger_i32)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()

    # ── O normalization ───────────────────────────────────────────────────
    if const_expr(flex_mod.needs_safe_norm):
        _safe_l = l_i[0].maximumf(fx.Float32(1e-12))
        inv_l = fx.Float32(1.0) / _safe_l
    else:
        inv_l = fx.Float32(1.0) / l_i[0]
    inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)
    for dc in range_constexpr(_n_d_chunks):
        o_accs[dc] = (Vec(o_accs[dc]) * inv_l_vec).ir_value()

    # ── O store ───────────────────────────────────────────────────────────
    # PV C layout (32x32): query = t%32, D mapped via value index.
    # Same O store as 32x32 kernel.
    _qrow = fx.Int32(local_tid % 32)
    _group_d_base = fx.Int32((local_tid // 32) * 4)
    _o_row_stride = hq * head_dim
    _out_elem_dtype = elem_dtype

    _o_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), _out_elem_dtype)
    o_store_reg = fx.make_rmem_tensor(fx.make_layout(4, 1), _out_elem_dtype)
    o_div = fx.logical_divide(
        fx.rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.recast_iter(_out_elem_dtype, fx.get_iter(o)),
                                   fx.make_layout(0x7FFFFFFF, 1))),
            max_size=True,
        ),
        fx.make_layout(1, 1),
    )
    o_base = o_off + _qrow * _o_row_stride
    for dc in range_constexpr(_n_d_chunks):
        o_vec = Vec(o_accs[dc])
        for k in range_constexpr(4):
            col = dc * 32 + _group_d_base + fx.Int32(k * 8)
            elems = [o_vec[k * 4 + e] for e in range_constexpr(4)]
            vbf = Vec.from_elements(elems, fx.Float32).to(_out_elem_dtype)
            off = o_base + col
            fx.memref_store_vec(vbf, o_store_reg)
            fx.copy(_o_store_atom, o_store_reg, fx.slice(o_div, (None, fx.Int32(off))))


# ── Launch wrapper ────────────────────────────────────────────────────────

@flyc.jit
def launch_flex_attn_16x16(
    o: fx.Tensor,
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    scale: fx.Float32,
    param: FlexAttnParam16,
    stream: fx.Stream = fx.Stream(None),
    ws_o: fx.Tensor = fx.Tensor,
    ws_ml: fx.Tensor = fx.Tensor,
):
    b = fx.Int32(fx.get_scalar(q.shape[0]))
    seqlen_q = fx.Int32(fx.get_scalar(q.shape[1]))
    hq = fx.Int32(fx.get_scalar(q.shape[2]))
    seqlen_kv = fx.Int32(fx.get_scalar(k.shape[1]))

    elem_dtype = _elem_dtype(param.dtype_id)

    # QK uses 16x16x16 for the MFMA atoms but we create tiled_mma with 32x32
    # for fragment layout compatibility with PV and O store.
    wave_layout = fx.make_layout((1, 1, 1), (1, 1, 0))
    mma_atom_pv = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, elem_dtype))
    tiled_mma_qk = fx.make_tiled_mma(mma_atom_pv, wave_layout)
    tiled_mma_pv = fx.make_tiled_mma(mma_atom_pv, wave_layout)

    rows_per_wg = param.block_m * param.num_groups
    num_q_tiles = (seqlen_q + rows_per_wg - 1) // rows_per_wg

    flex_attn_16x16_kernel._known_block_size = [param.block_threads, 1, 1]
    flex_attn_16x16_kernel._func.__name__ = (
        f"flex_attn_16x16_bf16_m{param.block_m}n{param.block_n}d{param.head_dim}"
        f"_g{param.num_groups}"
    )

    flex_attn_16x16_kernel(
        o, q, k, v, seqlen_q, seqlen_kv, b, scale, tiled_mma_qk, tiled_mma_pv, param,
        value_attrs={
            "rocdl.waves_per_eu": 2,
            "rocdl.flat_work_group_size": f"{param.block_threads},{param.block_threads}",
        },
    ).launch(
        grid=(num_q_tiles, hq, b),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


_flex_attn_16x16_compile_hints = {
    "waves_per_eu": 2,
    "unsafe_fp_math": True,
    "llvm_options": {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
    },
}
launch_flex_attn_16x16.compile_hints = dict(_flex_attn_16x16_compile_hints)


# ── Host-side entry point ─────────────────────────────────────────────────

def flydsl_flex_attention_16x16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    num_groups: int = 8,
    mask_type: int = MASK_NONE,
    score_type: int = SCORE_NONE,
    mask_window: int = 0,
    score_alibi_slope: float = 0.0,
    num_kv_splits: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Flash-attention forward using 4x MFMA 16x16x16 QK GEMM (gfx950).

    q/k/v: [B, S, H, D] (BSHD), bf16/f16. Returns [B, Sq, Hq, D].
    """
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"flex_attention_16x16 targets gfx950; got {arch!r}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q/k/v must be CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q/k/v must share dtype")
    if q.dim() != 4:
        raise ValueError(f"q must be 4D [B,S,H,D], got {q.dim()}D")

    dtype_id = FLEX_DTYPE_FP16 if q.dtype is torch.float16 else FLEX_DTYPE_BF16
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {q.dtype}")

    B, Sq, Hq, D = q.shape
    Skv, Hkv = k.shape[1], k.shape[2]
    if num_kv_heads is not None and num_kv_heads != Hkv:
        raise ValueError(f"num_kv_heads {num_kv_heads} != k head count {Hkv}")

    block_m, block_n = 32, 32
    rows_per_wg = block_m * num_groups
    if Sq % rows_per_wg != 0:
        raise ValueError(
            f"seqlen_q ({Sq}) must be a multiple of block_m*num_groups ({rows_per_wg})"
        )
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    if stream is None:
        stream = torch.cuda.current_stream()
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    param = make_flex_attn_param_16x16(
        seqlen_kv=Skv,
        dtype_id=dtype_id,
        head_dim=D,
        num_heads_q=Hq,
        num_heads_kv=Hkv,
        num_groups=num_groups,
        mask_type=mask_type,
        score_type=score_type,
        mask_window=mask_window,
        score_alibi_slope=score_alibi_slope,
        num_kv_splits=num_kv_splits,
    )

    if num_kv_splits > 1:
        ws_o = torch.zeros(num_kv_splits, B, Hq, Sq, D, dtype=torch.float32, device=q.device)
        ws_ml = torch.full((num_kv_splits, B, Hq, Sq, 2), -1e30, dtype=torch.float32, device=q.device)
    else:
        ws_o = torch.empty(1, dtype=torch.float32, device=q.device)
        ws_ml = torch.empty(1, dtype=torch.float32, device=q.device)

    launch_flex_attn_16x16(
        out.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(),
        fx.Float32(scale), param, stream,
        ws_o=ws_o, ws_ml=ws_ml,
    )
    return out
