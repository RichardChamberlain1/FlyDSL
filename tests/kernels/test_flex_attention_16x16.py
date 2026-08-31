#!/usr/bin/env python3
"""Correctness test for the 16x16x16 MFMA flex attention forward (gfx950)."""
import math
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch not available")
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA/ROCm not available")
    sys.exit(1)

import pytest

from flydsl.runtime.device import get_rocm_arch
from kernels.attention.flex_attention_16x16_gfx950 import (
    flydsl_flex_attention_16x16,
    MASK_NONE,
    MASK_CAUSAL,
)

_requires_gfx950 = pytest.mark.skipif(
    not get_rocm_arch().startswith("gfx950"),
    reason="16x16 attention kernel targets gfx950",
)


def _sdpa_ref(q, k, v, scale, *, is_causal=False):
    B, Sq, Hq, D = q.shape
    _, Skv, Hkv, _ = k.shape
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    if Hq != Hkv:
        rep = Hq // Hkv
        kh = kh.repeat_interleave(rep, dim=1)
        vh = vh.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(
        qh, kh, vh, scale=scale, is_causal=is_causal,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def _run(B, Sq, Skv, H, D, *, num_groups=8, Hkv=None, mask_type=MASK_NONE):
    if Hkv is None:
        Hkv = H
    dtype = torch.bfloat16
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.empty(B, Sq, H, D, dtype=dtype, device=dev).uniform_(-1, 1)
    k = torch.empty(B, Skv, Hkv, D, dtype=dtype, device=dev).uniform_(-1, 1)
    v = torch.empty(B, Skv, Hkv, D, dtype=dtype, device=dev).uniform_(-1, 1)
    scale = 1.0 / math.sqrt(D)

    out = flydsl_flex_attention_16x16(
        q, k, v, scale=scale, num_groups=num_groups,
        mask_type=mask_type,
    ).float()
    ref = _sdpa_ref(q, k, v, scale, is_causal=(mask_type == MASK_CAUSAL)).float()
    max_err = (out - ref).abs().max().item()
    cos = F.cosine_similarity(out.reshape(-1), ref.reshape(-1), dim=0).item()
    return max_err, cos


_SHAPES = [
    (1, 256, 256, 8, 128),
]


@_requires_gfx950
@pytest.mark.parametrize("B,Sq,Skv,H,D", _SHAPES)
def test_flex_attention_16x16_dense(B, Sq, Skv, H, D):
    max_err, cos = _run(B, Sq, Skv, H, D, num_groups=8)
    print(f"B{B} Sq{Sq} Skv{Skv} H{H} D{D}: max_err={max_err:.4f} cos={cos:.6f}")
    assert max_err < 0.03 and cos > 0.99, (
        f"B{B} Sq{Sq} Skv{Skv} H{H} D{D}: max_err={max_err} cos={cos}"
    )


if __name__ == "__main__":
    print("Running correctness tests...")
    shapes = [
        (1, 256, 32, 8, 128),
        (1, 256, 64, 8, 128),
        (1, 256, 128, 8, 128),
        (1, 256, 256, 8, 128),
        (2, 256, 256, 8, 128),
        (1, 256, 512, 8, 128),
        (2, 512, 1024, 32, 128),
        (2, 256, 2048, 32, 128),
    ]
    all_pass = True
    for B, Sq, Skv, H, D in shapes:
        Hkv = 8 if H > 8 else H
        max_err, cos = _run(B, Sq, Skv, H, D, num_groups=8, Hkv=Hkv)
        ok = "PASS" if max_err < 0.03 and cos > 0.99 else "FAIL"
        if ok == "FAIL":
            all_pass = False
        print(f"B={B} Sq={Sq} Skv={Skv} H={H} Hkv={Hkv} D={D}: max_err={max_err:.4f} cos={cos:.6f} {ok}")
    print("ALL PASS" if all_pass else "SOME FAILED")
