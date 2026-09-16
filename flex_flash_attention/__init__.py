"""Flex flash attention for PPU 1.5 (SM89).

Public surface: hand in a boolean mask, get attention back.  Decomposing that
mask into slices, choosing tile sizes and building the kernel's layout tables
are internal concerns and are NOT re-exported here.

    from flex_flash_attention import flash_attn_flex_flash, flash_attn_flex_flash_bwd

    out, lse = flash_attn_flex_flash(q, k, v, mask)
    dq, dk, dv = flash_attn_flex_flash_bwd(q, k, v, out, lse, dout, mask)

`MaskNotSupportedError` is part of the DECOMPOSER contract only: rows whose
visible columns split into several intervals (holes) are decomposed layer by
layer exactly, and masks whose per-row interval count exceeds the layer cap
(unstructured sparsity) raise from the decomposer.  The user-facing entry
points catch that and fall back to an exact element-wise bitmask kernel path,
so flash_attn_flex_flash / _bwd accept EVERY boolean mask.

Callers that already hold a slice decomposition (custom geometry, or a layout
reused across steps) can reach the advanced entry points and the slice
primitives through the submodules directly:

    from flex_flash_attention.interface import flash_attn_flex_flash_precomputed
    from flex_flash_attention.mask_decomp import SliceInfo, build_merge_layout
"""

from .mask_decomp import MaskNotSupportedError

from .interface import (
    flash_attn_flex_flash,
    flash_attn_flex_flash_bwd,
    flash_attn_flex_flash_cached,
    flash_attn_flex_flash_bwd_cached,
    flash_attn_flex_flash_cache_clear,
    decompose_mask_fused,
    mask_supported,
    flash_attn_flex_flash_desc,
    flex_flash_attention_sdpa_backend,
)

__all__ = [
    "flash_attn_flex_flash",
    "flash_attn_flex_flash_bwd",
    "flash_attn_flex_flash_cached",
    "flash_attn_flex_flash_bwd_cached",
    "flash_attn_flex_flash_cache_clear",
    "MaskNotSupportedError",
    "decompose_mask_fused",
    "mask_supported",
    "flash_attn_flex_flash_desc",
    "flex_flash_attention_sdpa_backend",
]
