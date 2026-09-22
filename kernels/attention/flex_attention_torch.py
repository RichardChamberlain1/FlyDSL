# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""PyTorch-compatible wrapper for the gfx950 FlyDSL flex-attention forward.

The public layout matches :func:`torch.nn.attention.flex_attention`:
``query`` is ``[B, Hq, Sq, D]`` and ``key``/``value`` are
``[B, Hkv, Skv, D]``.  Supported inference calls are converted to BSHD and
dispatched to :func:`flydsl_flex_attention_layout`; unsupported calls fall
back to PyTorch.

FlyDSL score and mask callables are constexpr-traced and therefore cannot
close over tensors or perform arbitrary tensor/global loads.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch

from flydsl.runtime.device import get_rocm_arch

from .flex_attention_gfx950 import (
    _prepare_constexpr_callable,
    flydsl_flex_attention_layout,
)


def _pytorch_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Optional[Callable],
    block_mask,
    scale: Optional[float],
    enable_gqa: bool,
    return_lse: bool,
    kernel_options,
    return_aux,
    mask_mod: Optional[Callable],
):
    """Call PyTorch, materializing a BlockMask for the FlyDSL-only mask_mod."""
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

    if mask_mod is not None and query.dim() == 4 and key.dim() == 4:
        block_mask = create_block_mask(
            mask_mod,
            B=query.shape[0],
            H=query.shape[1],
            Q_LEN=query.shape[2],
            KV_LEN=key.shape[2],
            device=query.device,
        )

    return torch_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=kernel_options,
        return_aux=return_aux,
    )


def _block_mask_callable(block_mask):
    if block_mask is None:
        return None
    mask_mod = getattr(block_mask, "mask_mod", None)
    if not callable(mask_mod):
        return None
    # PyTorch deliberately replaces mask_mod with this stub after slicing.
    if getattr(mask_mod, "__name__", "") == "_sliced_mask_mod_error":
        return None
    return mask_mod


def _can_use_flydsl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    score_mod: Optional[Callable],
    mask_mod: Optional[Callable],
    block_mask,
    enable_gqa: bool,
    return_lse: bool,
    return_aux,
    kernel_options,
) -> bool:
    if os.environ.get("FLYDSL_FLEX_TORCH", "1") == "0":
        return False
    if return_lse or return_aux is not None or kernel_options is not None:
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)):
        return False
    if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in (query, key, value)):
        return False
    if not all(t.dim() == 4 for t in (query, key, value)):
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return False
    if query.device != key.device or query.device != value.device:
        return False

    Bq, Hq, Sq, Dq = query.shape
    Bk, Hkv, Skv, Dk = key.shape
    Bv, Hv, Sv, Dv = value.shape
    if (Bq, Hkv, Skv) != (Bk, Hv, Sv) or Dq != Dk or Dv != Dq:
        return False
    if Dq != 128 or Sq % (32 * 8) != 0 or Skv % 64 != 0:
        return False
    if Hkv == 0 or Hq % Hkv != 0:
        return False
    if Hq != Hkv and not enable_gqa:
        return False
    if block_mask is not None and mask_mod is None:
        return False

    try:
        if not get_rocm_arch().startswith("gfx950"):
            return False
        _prepare_constexpr_callable(score_mod, lambda score, b, h, q, kv: score)
        _prepare_constexpr_callable(mask_mod, lambda b, h, q, kv: q == q)
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


def flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Optional[Callable] = None,
    block_mask=None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options=None,
    *,
    return_aux=None,
    mask_mod: Optional[Callable] = None,
):
    """Run gfx950 FlyDSL flex attention through a PyTorch-compatible API.

    Inputs and outputs use PyTorch's BHSD layout.  ``mask_mod`` is a FlyDSL
    convenience that avoids constructing a PyTorch ``BlockMask``.  Calls not
    supported by the n64/d128 gfx950 forward automatically use PyTorch's
    implementation instead.
    """
    block_mask_mod = _block_mask_callable(block_mask)
    chosen_mask_mod = mask_mod if mask_mod is not None else block_mask_mod

    if not _can_use_flydsl(
        query,
        key,
        value,
        score_mod=score_mod,
        mask_mod=chosen_mask_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        return_aux=return_aux,
        kernel_options=kernel_options,
    ):
        return _pytorch_flex_attention(
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            enable_gqa,
            return_lse,
            kernel_options,
            return_aux,
            mask_mod,
        )

    q_bshd = query.permute(0, 2, 1, 3).contiguous()
    k_bshd = key.permute(0, 2, 1, 3).contiguous()
    v_bshd = value.permute(0, 2, 1, 3).contiguous()
    out_bshd = flydsl_flex_attention_layout(
        q_bshd,
        k_bshd,
        v_bshd,
        scale=scale,
        num_kv_heads=key.shape[1],
        score_mod=score_mod,
        mask_mod=chosen_mask_mod,
    )
    return out_bshd.permute(0, 2, 1, 3).contiguous()
