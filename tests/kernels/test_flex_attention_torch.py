#!/usr/bin/env python3
"""Tests for the BHSD PyTorch adapter around gfx950 FlyDSL flex attention."""

import importlib
import math
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

try:
    import pytest
    import torch
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention
except ImportError:
    print("PyTorch/pytest not available")
    sys.exit(1)

if not torch.cuda.is_available():
    print("ROCm not available")
    sys.exit(1)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

torch_wrapper = importlib.import_module("kernels.attention.flex_attention_torch")

_requires_gfx950 = pytest.mark.skipif(
    not get_rocm_arch().startswith("gfx950"),
    reason="layout-API attention kernel targets gfx950",
)


def _causal_mask(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


def _make_qkv(*, B=1, Hq=4, Hkv=None, Sq=256, Skv=256, D=128):
    if Hkv is None:
        Hkv = Hq
    torch.manual_seed(0)
    make = lambda *shape: torch.empty(  # noqa: E731
        *shape, dtype=torch.bfloat16, device="cuda"
    ).uniform_(-1, 1)
    return (
        make(B, Hq, Sq, D),
        make(B, Hkv, Skv, D),
        make(B, Hkv, Skv, D),
    )


def _check(out, ref, *, max_err_tol=8e-2, cos_tol=0.98):
    assert out.shape == ref.shape
    max_err = (out.float() - ref.float()).abs().max().item()
    cos = F.cosine_similarity(
        out.float().reshape(-1), ref.float().reshape(-1), dim=0
    ).item()
    assert max_err < max_err_tol and cos > cos_tol, f"max_err={max_err} cos={cos}"


@_requires_gfx950
def test_torch_wrapper_dense_bhsd(monkeypatch):
    q, k, v = _make_qkv()
    scale = 1.0 / math.sqrt(q.shape[-1])
    seen = {}
    raw_layout = torch_wrapper.flydsl_flex_attention_layout

    def record_layout(q_bshd, k_bshd, v_bshd, **kwargs):
        seen["q_shape"] = q_bshd.shape
        return raw_layout(q_bshd, k_bshd, v_bshd, **kwargs)

    monkeypatch.setattr(torch_wrapper, "flydsl_flex_attention_layout", record_layout)
    out = torch_wrapper.flex_attention(q, k, v, scale=scale)
    torch.compiler.reset()
    ref = torch_flex_attention(q, k, v, scale=scale)

    assert seen["q_shape"] == (1, 256, 4, 128)
    assert out.shape == q.shape
    assert out.is_contiguous()
    _check(out, ref)


@_requires_gfx950
def test_torch_wrapper_block_mask_and_explicit_mask_mod():
    q, k, v = _make_qkv()
    scale = 1.0 / math.sqrt(q.shape[-1])
    block_mask = create_block_mask(
        _causal_mask,
        B=q.shape[0],
        H=q.shape[1],
        Q_LEN=q.shape[2],
        KV_LEN=k.shape[2],
        device=q.device,
    )

    from_block_mask = torch_wrapper.flex_attention(
        q, k, v, block_mask=block_mask, scale=scale
    )
    from_mask_mod = torch_wrapper.flex_attention(
        q, k, v, mask_mod=_causal_mask, scale=scale
    )
    torch.compiler.reset()
    ref = torch_flex_attention(q, k, v, block_mask=block_mask, scale=scale)

    _check(from_block_mask, ref)
    _check(from_mask_mod, ref)
    _check(from_block_mask, from_mask_mod, max_err_tol=1e-3, cos_tol=0.9999)


@_requires_gfx950
def test_torch_wrapper_gqa():
    q, k, v = _make_qkv(Hq=4, Hkv=2)
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = torch_wrapper.flex_attention(q, k, v, scale=scale, enable_gqa=True)
    torch.compiler.reset()
    ref = torch_flex_attention(q, k, v, scale=scale, enable_gqa=True)
    _check(out, ref)


@_requires_gfx950
@pytest.mark.parametrize("Sq,D", [(256, 64), (255, 128)])
def test_torch_wrapper_falls_back_for_unsupported_shape(monkeypatch, Sq, D):
    q, k, v = _make_qkv(Sq=Sq, Skv=256, D=D)
    scale = 1.0 / math.sqrt(D)

    def unexpected_flydsl_call(*args, **kwargs):
        raise AssertionError("unsupported shape must use the PyTorch fallback")

    monkeypatch.setattr(
        torch_wrapper, "flydsl_flex_attention_layout", unexpected_flydsl_call
    )
    out = torch_wrapper.flex_attention(q, k, v, scale=scale)
    torch.compiler.reset()
    ref = torch_flex_attention(q, k, v, scale=scale)
    _check(out, ref, max_err_tol=1e-3, cos_tol=0.9999)


@_requires_gfx950
def test_torch_wrapper_fallback_preserves_explicit_mask_mod(monkeypatch):
    q, k, v = _make_qkv(D=64)
    scale = 1.0 / math.sqrt(q.shape[-1])

    def unexpected_flydsl_call(*args, **kwargs):
        raise AssertionError("unsupported shape must use the PyTorch fallback")

    monkeypatch.setattr(
        torch_wrapper, "flydsl_flex_attention_layout", unexpected_flydsl_call
    )
    out = torch_wrapper.flex_attention(
        q, k, v, scale=scale, mask_mod=_causal_mask
    )
    block_mask = create_block_mask(
        _causal_mask,
        B=q.shape[0],
        H=q.shape[1],
        Q_LEN=q.shape[2],
        KV_LEN=k.shape[2],
        device=q.device,
    )
    torch.compiler.reset()
    ref = torch_flex_attention(
        q, k, v, scale=scale, block_mask=block_mask
    )
    _check(out, ref, max_err_tol=1e-3, cos_tol=0.9999)
