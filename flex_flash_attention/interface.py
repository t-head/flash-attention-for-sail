"""Python interface for flex flash attention.

Usage:
    from flex_flash_attention.interface import flash_attn_flex_flash

    # Create an arbitrary boolean mask
    mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool, device='cuda')
    mask[:10, :20] = True          # first 10 Q rows attend to first 20 K cols
    mask[10:20, 10:30] = True      # next 10 Q rows attend to K[10:30]

    # Run flex flash attention
    out, lse = flash_attn_flex_flash(q, k, v, mask)

    # Backward
    dq, dk, dv = flash_attn_flex_flash_bwd(q, k, v, out, lse, dout, mask)

    # Cached variant: decomposition is computed once per distinct mask
    # content and reused across calls; a full-content check (~0.4ms at
    # S=25k) re-validates the mask on every call and re-decomposes
    # automatically when it changed.
    out, lse = flash_attn_flex_flash_cached(q, k, v, mask)
    dq, dk, dv = flash_attn_flex_flash_bwd_cached(q, k, v, out, lse, dout, mask)
"""

import torch
import weakref
from typing import Optional, Tuple

try:
    import flash_attn_3._C  # triggers TORCH_LIBRARY(flex_flash_attention) registration
    _flex_flash_attention = torch.ops.flex_flash_attention
except ImportError:
    _flex_flash_attention = None

from .mask_decomp import (decompose_mask, decompose_mask_optimized,
                          slices_to_tensors, build_merge_layout,
                          MaskNotSupportedError, SliceInfo, pack_mask_bits,
                          build_bh_layout,
                          SLICE_FULL, SLICE_CAUSAL, SLICE_BICAUSAL,
                          SLICE_BITMASK)

# kBlockM=128 dispatch hint.
# Measured on PPU SM89 (diag_kspan/diag_kspan2 sweeps, b=4 h=16):
# kBlockM=128 wins for large-span slices at head_dim 128:
#   - FULL slices with K-span >= 2048 (full s=2048: 1.11ms vs 1.48ms kbm64;
#     s=4096: 3.34ms vs 5.28ms)
#   - diagonal slices (CAUSAL/INVCAUSAL/BICAUSAL) once their K-spans are
#     long enough — after the no-mask-zone fix (aligned K ranges skip
#     mask.apply() on fully-visible blocks, crossing blocks are pruned to
#     the diagonal) pure causal s=24576: kbm128 8.94 vs kbm64 12.94ms;
#     frame-stair (FULL+CAUSAL mix): 6.46 vs 9.59ms
# Everything else is faster on kBlockM=64:
#   - FULL slices with K-span <= 1024 (s=1024 dense: 0.41 vs 0.44ms)
#   - block-diagonal document masks (ALL slices FULL but each K-span is
#     s/n_docs <= 1024): 0.10ms vs 0.47ms — the old "all slices FULL" rule
#     mis-routed these to kbm128 and lost ~5x (the avg-span guard below
#     catches them)
# kbm64 is never pathological, so unverified shapes (head_dim != 128)
# default to it.
_KBM128_MIN_KSPAN = 2048      # retired: kept for reference, no longer used
_KBN64_MAX_KSPAN = 20480      # kbn64 win/lose boundary on span (see _kblockn64_hint)


# Identity-level layout cache: within one training iteration fwd and bwd
# receive the SAME mask tensor object, and re-decomposing it for bwd is
# pure waste (~5.7ms at S=25k).  Keyed on tensor identity: the weakref
# comparison guards against id reuse after deallocation, _version catches
# in-place mutation, the shape check guards resized views.  Entries hold
# the layout tensors (tens of MB at S=25k), so keep the LRU tiny.
_id_layout_cache = {}
_ID_LAYOUT_CACHE_MAX = 4

# Bwd hdim128 Q-tile is 48 rows; slice bands must be a MULTIPLE of 48 to
# avoid misaligned block over-coverage (~25% at the legacy 128-row bands),
# yet coarse enough that each slice spans several m_blocks so the Q/dO
# staging pipeline overlaps (48-row bands measured slower despite fewer
# tiles).  Stair S=25286 bwd sweep (ms): 96=27.29 144=26.29 192=25.64
# 240=25.30 288=24.96 384=24.57 480=24.45 576=24.30 768=24.14 960=24.17
# 1440=24.21 1920=24.23 — plateau from ~576; 768 picked (16 tiles per
# band).  Period boundaries not divisible by 768 (e.g. stair P=2048 leaves
# 512-row tails) still over-cover only <=47 rows per boundary — negligible.
_BWD_KBLOCK_M = 768


def _mask_layout(mask: torch.Tensor, seqlen_q: int, seqlen_k: int,
                 device: torch.device, kblock_m: int = 128):
    """(slice_tensors, slices) for the e2e entry points.

    Prefers the slice decomposition (algebraic masks, all optimizations on).
    When the mask is outside the layered-interval envelope (decompose raises
    MaskNotSupportedError), falls back to ONE SLICE_BITMASK slice covering
    the full rectangle plus the packed element-wise bitmask — exact for ANY
    boolean mask, at the cost of per-element bit tests and no tile pruning.
    slice_tensors['mask_bits'] is None on the slice path.

    kblock_m must match the consuming kernel's Q-tile: fwd runs 128-row
    tiles; the hdim128 bwd kernel runs 48-row tiles, and feeding it
    128-aligned bands wastes ~25% of tile-pairs on misaligned coverage.
    Layouts are cached per (mask identity, kblock_m).
    """
    ent = _id_layout_cache.get((id(mask), kblock_m))
    if ent is not None:
        ref, version, shape, tensors, slices = ent
        if ref() is mask and version == mask._version and shape == mask.shape:
            return tensors, slices
        del _id_layout_cache[(id(mask), kblock_m)]

    try:
        slices = decompose_mask_optimized(mask, kBlockM=kblock_m, tile_align=True)
    except MaskNotSupportedError:
        slices = [SliceInfo(0, seqlen_q, 0, seqlen_k, SLICE_BITMASK, 0, 0)]
    tensors = build_merge_layout(slices, seqlen_q, kBlockM=kblock_m, device=device)
    tensors['mask_bits'] = (pack_mask_bits(mask)
                            if slices[0].mask_type == SLICE_BITMASK else None)

    _id_layout_cache[(id(mask), kblock_m)] = (weakref.ref(mask), mask._version,
                                              mask.shape, tensors, slices)
    if len(_id_layout_cache) > _ID_LAYOUT_CACHE_MAX:
        _id_layout_cache.pop(next(iter(_id_layout_cache)))
    return tensors, slices


def _kblockm128_hint(slices, headdim) -> bool:
    """True when kBlockM=128 should be allowed for this slice set.

    CURRENTLY ALWAYS FALSE.  A locked-clock (1300 MHz) A/B grid over stair
    (S=8192/25286/65536, H=12/40), causal (4096..24576), dense (4096..25286),
    sliding (bw=11264) and document masks found kBlockM=128 fwd to lose
    everywhere against kBlockM=64 in the current kernel build — including the
    shapes the original avg-K-span >= 2048 rule was calibrated on (stair
    s=4096 now measures kbm128 0.689 vs kbm64 0.443 ms; causal s=24576
    17.99 vs 13.38 ms).  The kernel's kBlockM=64 path has since improved
    enough that the historical wins no longer reproduce, so the hint stays
    off until a shape is found where kbm128 wins again.  bwd has no tile
    switch and is unaffected.
    """
    return False


def _kblockn64_hint(slices, headdim) -> bool:
    """True when the kBlockN=64 tile should be used.

    Locked-clock (1300 MHz) A/B at kBlockM=64, d=128, H=12: the narrow
    N-tile wins BIG on every slice family whose K-spans stay <=
    _KBN64_MAX_KSPAN — FULL sets (document n=4..32 span 4096..512:
    -32..-41%; dense s=16384 span 16384: -36%), diagonal sets (causal
    s=8192..20480: -25..-27%; frame-stair c=1..11 span 4096..24576*:
    -17..-20%), but LOSES above it (causal/dense s=25286 span 25286:
    +4..+5%) and on BICAUSAL (type-3) sets at any tested bandwidth
    (sliding bw=2048..11264: +4..+6%, bw=1024 only -2%).  (*) stair
    c=11 span 24576 > the cap would be excluded; at span 12288
    (production c=5) it wins -20%, and s=65536 c=5 (span 12288) -31%.

    The pre-kBlockM=64-rework rule (all-FULL, span <= 512) is obsolete:
    the kbn96 last-block padding cost now loses to kbn64's cheaper
    K-iteration up to span ~20480 for every non-BICAUSAL family.

    Guards: kbn64 only exists on the kBlockM=64 branch, so every slice
    needs a Q range >= 64 (high-step-count stair masks with b < 64
    Q-overlap and lose, 0.344 vs 0.292 ms).  d=64 keeps its own rule
    (some FULL span <= 64), unvalidated for the new wide-span regime.
    """
    if headdim not in (64, 128) or not slices:
        return False
    if min(s.q_end - s.q_start for s in slices) < 64:  # kbn64 runs on the kBlockM=64 branch
        return False
    spans = [s.k_end - s.k_start for s in slices]
    if headdim == 128:
        if any(s.mask_type == SLICE_BICAUSAL for s in slices):
            return False
        return max(spans) <= _KBN64_MAX_KSPAN
    if any(s.mask_type != SLICE_FULL for s in slices):
        return False
    return min(spans) <= 64


def flash_attn_flex_flash(
    q: torch.Tensor,           # (b, s_q, h, d)
    k: torch.Tensor,           # (b, s_k, h_k, d)
    v: torch.Tensor,           # (b, s_k, h_k, dv)
    mask: torch.Tensor,        # (s_q, s_k) bool — True = attend
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    inner_min_to_max: bool = False,
    persistent_scheduler: bool = False,
    dropout_p: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, ...]:
    """Compute flash attention with an arbitrary boolean mask.

    The mask is decomposed into slices and passed to the PPU-optimized kernel
    via the flex_flash_attention C++ extension.  The RangeMerge layout is always
    built (identity for non-overlapping decompositions), so Q-overlapping
    slice sets produced by custom decompositions are also supported.

    Supported mask envelope: EVERY boolean mask.  Masks whose rows decompose
    into few contiguous intervals run on the slice path (FULL / CAUSAL /
    INVCAUSAL / BICAUSAL, all optimizations on); masks beyond the layered
    envelope (per-row interval count over the cap: random sparsity, dilated
    patterns, KV eviction) transparently fall back to an exact element-wise
    bitmask kernel path (SLICE_BITMASK) — slower, but never wrong and never
    raises.

    Args:
        q: Query tensor (batch, seqlen_q, num_heads, head_dim).
        k: Key tensor (batch, seqlen_k, num_heads_k, head_dim).
        v: Value tensor (batch, seqlen_k, num_heads_k, head_dim_v).
        mask: Boolean mask (seqlen_q, seqlen_k). True = attend, False = mask.
            Shared across batch AND heads.
        softmax_scale: Scaling factor. Defaults to 1/sqrt(head_dim).
        out: Optional pre-allocated output tensor.
        softcap: Optional logit softcap value (Gemma-style attention).
        inner_min_to_max: iterate K blocks min→max instead of max→min.
        persistent_scheduler: use the dynamic persistent tile scheduler.
        dropout_p: dropout probability applied to the softmax probabilities
            (FA2 semantics: zeroed elements, output scaled by 1/(1-p)).
        generator: optional CUDA generator for the dropout Philox stream.

    Returns:
        out: Attention output (batch, seqlen_q, num_heads, head_dim_v).
        lse: Log-sum-exp (batch, num_heads, seqlen_q).
        rng_state (only when dropout_p > 0): int64 tensor [seed, offset];
            pass it to flash_attn_flex_flash_bwd to replay the same mask.
    """
    if _flex_flash_attention is None:
        raise RuntimeError(
            "flex_flash_attention C++ extension not loaded. "
            "Make sure the flex_flash_attention module is built and importable."
        )

    assert q.is_cuda, "q must be on CUDA"
    assert k.is_cuda, "k must be on CUDA"
    assert v.is_cuda, "v must be on CUDA"
    assert mask.dtype == torch.bool, "mask must be boolean"

    seqlen_q = q.size(1)
    seqlen_k = k.size(1)
    assert mask.shape == (seqlen_q, seqlen_k), \
        f"mask shape {mask.shape} != ({seqlen_q}, {seqlen_k})"

    # Decompose mask into slices (with P4 tile alignment optimization);
    # masks beyond the layered envelope fall back to the bitmask path.
    device = q.device
    slice_tensors, slices = _mask_layout(mask, seqlen_q, seqlen_k, device)

    # Call the C++ kernel
    result = _flex_flash_attention.fwd(
        q, k, v,
        slice_tensors['q_starts'],
        slice_tensors['q_ends'],
        slice_tensors['k_starts'],
        slice_tensors['k_ends'],
        slice_tensors['mask_types'],
        slice_tensors['row_to_slice'],
        slice_tensors['diagonal_offsets'],
        slice_tensors['band_widths'],
        slice_tensors['vbatch_to_slice'],
        slice_tensors['row_to_vbatch_start'],
        slice_tensors['row_to_vbatch_end'],
        out,
        softmax_scale,
        softcap,
        inner_min_to_max,
        persistent_scheduler,
        _kblockm128_hint(slices, q.shape[-1]),
        _kblockn64_hint(slices, q.shape[-1]),
        dropout_p,
        generator,
        slice_tensors['mask_bits'],
    )

    if dropout_p > 0.0:
        return result[0], result[1], result[2]  # (out, softmax_lse, rng_state)
    return result[0], result[1]  # (out, softmax_lse)


def flash_attn_flex_flash_precomputed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_starts: torch.Tensor,
    q_ends: torch.Tensor,
    k_starts: torch.Tensor,
    k_ends: torch.Tensor,
    mask_types: torch.Tensor,
    row_to_slice: torch.Tensor,
    diagonal_offsets: torch.Tensor,
    band_widths: torch.Tensor,
    vbatch_to_slice: Optional[torch.Tensor] = None,
    row_to_vbatch_start: Optional[torch.Tensor] = None,
    row_to_vbatch_end: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    inner_min_to_max: bool = False,
    persistent_scheduler: bool = False,
    use_kblockm128: Optional[bool] = None,
    kblockn64: bool = False,
    mask_bits: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Call the kernel directly with pre-computed slice tensors.

    Use this when you've already decomposed the mask (e.g., from a custom
    decomposition or when the slice structure is known ahead of time).

    For Q-overlapping slices, pass the RangeMerge layout tensors
    (vbatch_to_slice / row_to_vbatch_start / row_to_vbatch_end, see
    mask_decomp.build_merge_layout); the kernel merges the per-slice
    partial results via LSE.  Without them the legacy single-slice
    dispatch (row_to_slice) is used.

    use_kblockm128: kBlockM=128 dispatch hint.  None = kernel-side shape
    heuristic; True/False forces the tile size (subject to head_dim/seqlen
    guards).  Set False for masks with large-span diagonal slices.
    kblockn64: use the kBlockN=64 tile (short-K-span dense slice sets).
    mask_bits: packed element-wise bitmask (see mask_decomp.pack_mask_bits)
    for SLICE_BITMASK slices; None for the four algebraic mask types.
    """
    if _flex_flash_attention is None:
        raise RuntimeError("flex_flash_attention C++ extension not loaded.")

    result = _flex_flash_attention.fwd(
        q, k, v,
        q_starts, q_ends, k_starts, k_ends,
        mask_types, row_to_slice,
        diagonal_offsets, band_widths,
        vbatch_to_slice, row_to_vbatch_start, row_to_vbatch_end,
        out, softmax_scale,
        softcap,
        inner_min_to_max,
        persistent_scheduler,
        use_kblockm128,
        kblockn64,
        0.0,        # dropout_p
        None,       # gen
        mask_bits,
    )
    return result[0], result[1]


def flash_attn_flex_flash_bwd(
    q: torch.Tensor,           # (b, s_q, h, d)
    k: torch.Tensor,           # (b, s_k, h_k, d)
    v: torch.Tensor,           # (b, s_k, h_k, dv)
    out: torch.Tensor,         # (b, s_q, h, dv)  — from fwd
    softmax_lse: torch.Tensor, # (b, h, s_q)      — from fwd
    dout: torch.Tensor,        # (b, s_q, h, dv)
    mask: torch.Tensor,        # (s_q, s_k) bool — True = attend
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    softcap: Optional[float] = None,
    reduce_kv: bool = False,
    use_loop_k: bool = False,
    inner_min_to_max: Optional[bool] = None,
    persistent_scheduler: bool = False,
    dropout_p: float = 0.0,
    rng_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for flex flash attention.

    The mask is re-decomposed with the same settings used by the forward
    (decompose_mask_optimized, kBlockM=128, tile_align=True, bitmask
    fallback beyond the layered envelope), so the slice layout matches the
    fwd that produced `out` / `softmax_lse`.

    Args:
        q, k, v: inputs of the forward pass.
        out, softmax_lse: outputs of flash_attn_flex_flash.
        dout: upstream gradient of `out`.
        mask: the same boolean mask used in the forward pass.
        softmax_scale: must match the forward call.
        dq/dk/dv: optional pre-allocated output tensors.
        deterministic: enforce a fixed accumulation order (bitwise
            reproducible); requires the extension to be built with
            FLASH_ATTENTION_ARB_DETERMINISTIC=1.
        softcap: must match the forward call.
        reduce_kv: accumulate dK/dV into a separate fp32 workspace and
            reduce in a dedicated kernel; requires use_loop_k=True.
        use_loop_k: use the LoopK mainloop (inner loop over K blocks);
            requires FLASH_ATTENTION_ARB_LOOPK=1 at build time.
        inner_min_to_max: iterate the inner loop min→max instead of
            max→min.  None (default) resolves to True: on the 64×96 tile
            the min→max direction is 2.5-3% faster for bwd across all
            measured mask scenarios (dense/causal/stair/sliding_window/
            prefix_lm/document/blockwise); pass False for the legacy
            direction.  NOTE: this default applies to BWD only — the FWD
            kernels are faster max→min, so keep the fwd default.
        persistent_scheduler: use the dynamic persistent tile scheduler.
        dropout_p: dropout probability used in the forward pass (must match).
        rng_state: the int64 [seed, offset] tensor returned by the fwd call
            when dropout_p > 0; required to replay the identical Philox mask.

    Returns:
        dq, dk, dv: gradients w.r.t. q, k, v.
    """
    if _flex_flash_attention is None:
        raise RuntimeError(
            "flex_flash_attention C++ extension not loaded. "
            "Make sure the flex_flash_attention module is built and importable."
        )

    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert mask.dtype == torch.bool, "mask must be boolean"
    if inner_min_to_max is None:
        inner_min_to_max = True   # measured ~2.5-3% bwd win, all scenarios
    if dropout_p > 0.0:
        assert rng_state is not None, \
            "rng_state (returned by fwd) is required when dropout_p > 0"

    seqlen_q = q.size(1)
    seqlen_k = k.size(1)
    assert mask.shape == (seqlen_q, seqlen_k), \
        f"mask shape {mask.shape} != ({seqlen_q}, {seqlen_k})"

    # Bwd hdim128 runs 48-row Q tiles: decompose on the 48-row grid so
    # slice bands align with tiles (128-row bands over-cover ~25%).
    device = q.device
    slice_tensors, _slices = _mask_layout(mask, seqlen_q, seqlen_k, device,
                                          kblock_m=_BWD_KBLOCK_M)

    result = _flex_flash_attention.bwd(
        q, k, v, out, softmax_lse, dout,
        slice_tensors['q_starts'],
        slice_tensors['q_ends'],
        slice_tensors['k_starts'],
        slice_tensors['k_ends'],
        slice_tensors['mask_types'],
        slice_tensors['row_to_slice'],
        slice_tensors['diagonal_offsets'],
        slice_tensors['band_widths'],
        slice_tensors['vbatch_to_slice'],
        slice_tensors['row_to_vbatch_start'],
        slice_tensors['row_to_vbatch_end'],
        dq, dk, dv,
        softmax_scale,
        deterministic,
        softcap,
        reduce_kv,
        use_loop_k,
        inner_min_to_max,
        persistent_scheduler,
        dropout_p,
        rng_state,
        slice_tensors['mask_bits'],
    )
    return result[0], result[1], result[2]  # (dq, dk, dv)


def _check_f32_available():
    """The fp32 kernel set is a mandatory part of the library; this only
    verifies the C++ extension is loaded at all."""
    if _flex_flash_attention is None:
        raise RuntimeError(
            "flex_flash_attention C++ extension not loaded. "
            "Make sure the flex_flash_attention module is built and importable."
        )


def flash_attn_flex_flash_f32(
    q: torch.Tensor,           # (b, s_q, h, d)  float32
    k: torch.Tensor,           # (b, s_k, h_k, d) float32
    v: torch.Tensor,           # (b, s_k, h_k, dv) float32
    mask: torch.Tensor,        # (s_q, s_k) bool — True = attend
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    inner_min_to_max: bool = False,
    persistent_scheduler: bool = False,
    dropout_p: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, ...]:
    """fp32 variant of flash_attn_flex_flash — ISOLATED kernel set.

    Inputs MUST be float32; the compute runs on TF32 tensor cores (fp32
    accumulate, 10-bit mantissa inputs).  Expected accuracy: max abs error
    ~1e-3 (rel ~5e-3) vs an fp32 dense reference for unit-variance inputs —
    markedly better than bf16 kernels, below true-fp32 CUDA-core quality.
    The bf16/fp16 kernels, dispatch and API are untouched by this path.

    dropout_p: dropout probability applied to the softmax probabilities
        (FA2 semantics: zeroed elements, output scaled by 1/(1-p)); the
        rng_state returned when dropout_p > 0 must be passed to
        flash_attn_flex_flash_bwd_f32 to replay the identical Philox mask.
    generator: optional CUDA generator for the dropout Philox stream.
    The per-(b,h) heterogeneous layout has its own entry,
    flash_attn_flex_flash_bh_f32.

    Returns (out, softmax_lse), or (out, softmax_lse, rng_state) when
    dropout_p > 0.
    """
    _check_f32_available()

    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert q.dtype == torch.float32 and k.dtype == torch.float32 \
        and v.dtype == torch.float32, "fwd_f32 requires float32 q/k/v"
    assert mask.dtype == torch.bool, "mask must be boolean"

    seqlen_q = q.size(1)
    seqlen_k = k.size(1)
    assert mask.shape == (seqlen_q, seqlen_k), \
        f"mask shape {mask.shape} != ({seqlen_q}, {seqlen_k})"

    device = q.device
    slice_tensors, slices = _mask_layout(mask, seqlen_q, seqlen_k, device)

    result = _flex_flash_attention.fwd_f32(
        q, k, v,
        slice_tensors['q_starts'],
        slice_tensors['q_ends'],
        slice_tensors['k_starts'],
        slice_tensors['k_ends'],
        slice_tensors['mask_types'],
        slice_tensors['row_to_slice'],
        slice_tensors['diagonal_offsets'],
        slice_tensors['band_widths'],
        slice_tensors['vbatch_to_slice'],
        slice_tensors['row_to_vbatch_start'],
        slice_tensors['row_to_vbatch_end'],
        out,
        softmax_scale,
        softcap,
        inner_min_to_max,
        persistent_scheduler,
        _kblockm128_hint(slices, q.shape[-1]),
        _kblockn64_hint(slices, q.shape[-1]),
        dropout_p,
        generator,
        slice_tensors['mask_bits'],
    )
    if dropout_p > 0.0:
        return result[0], result[1], result[2]  # (out, softmax_lse, rng_state)
    return result[0], result[1]


def flash_attn_flex_flash_bwd_f32(
    q: torch.Tensor,           # (b, s_q, h, d)  float32
    k: torch.Tensor,           # (b, s_k, h_k, d) float32
    v: torch.Tensor,           # (b, s_k, h_k, dv) float32
    out: torch.Tensor,         # from flash_attn_flex_flash_f32
    softmax_lse: torch.Tensor, # from flash_attn_flex_flash_f32
    dout: torch.Tensor,
    mask: torch.Tensor,        # same mask as the forward pass
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    softcap: Optional[float] = None,
    reduce_kv: bool = False,
    use_loop_k: bool = False,
    inner_min_to_max: Optional[bool] = None,
    persistent_scheduler: bool = False,
    dropout_p: float = 0.0,
    rng_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass of the ISOLATED fp32 (TF32 compute) kernel set.

    All inputs/outputs float32; see flash_attn_flex_flash_f32 for the
    accuracy notes.  Feature parity with flash_attn_flex_flash_bwd (P6,
    2026-08): softcap / reduce_kv / use_loop_k / dropout replay all ride
    the same C++ plumbing (the loop-k / softcap axes are build-gated
    exactly as for 16-bit).  Returns (dq, dk, dv).
    """
    _check_f32_available()

    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert q.dtype == torch.float32 and k.dtype == torch.float32 \
        and v.dtype == torch.float32, "bwd_f32 requires float32 inputs"
    assert mask.dtype == torch.bool, "mask must be boolean"
    if inner_min_to_max is None:
        inner_min_to_max = True   # measured ~2.5-3% bwd win, all scenarios
    if dropout_p > 0.0:
        assert rng_state is not None, \
            "rng_state (returned by fwd) is required when dropout_p > 0"

    seqlen_q = q.size(1)
    seqlen_k = k.size(1)
    assert mask.shape == (seqlen_q, seqlen_k), \
        f"mask shape {mask.shape} != ({seqlen_q}, {seqlen_k})"

    device = q.device
    slice_tensors, _slices = _mask_layout(mask, seqlen_q, seqlen_k, device)

    result = _flex_flash_attention.bwd_f32(
        q, k, v, out, softmax_lse, dout,
        slice_tensors['q_starts'],
        slice_tensors['q_ends'],
        slice_tensors['k_starts'],
        slice_tensors['k_ends'],
        slice_tensors['mask_types'],
        slice_tensors['row_to_slice'],
        slice_tensors['diagonal_offsets'],
        slice_tensors['band_widths'],
        slice_tensors['vbatch_to_slice'],
        slice_tensors['row_to_vbatch_start'],
        slice_tensors['row_to_vbatch_end'],
        dq, dk, dv,
        softmax_scale,
        deterministic,
        softcap,
        reduce_kv,
        use_loop_k,
        inner_min_to_max,
        persistent_scheduler,
        dropout_p,
        rng_state,
        slice_tensors['mask_bits'],
    )
    return result[0], result[1], result[2]  # (dq, dk, dv)


def flash_attn_flex_flash_bwd_precomputed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dout: torch.Tensor,
    q_starts: torch.Tensor,
    q_ends: torch.Tensor,
    k_starts: torch.Tensor,
    k_ends: torch.Tensor,
    mask_types: torch.Tensor,
    row_to_slice: torch.Tensor,
    diagonal_offsets: torch.Tensor,
    band_widths: torch.Tensor,
    vbatch_to_slice: Optional[torch.Tensor] = None,
    row_to_vbatch_start: Optional[torch.Tensor] = None,
    row_to_vbatch_end: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    softcap: Optional[float] = None,
    reduce_kv: bool = False,
    use_loop_k: bool = False,
    inner_min_to_max: Optional[bool] = None,
    persistent_scheduler: bool = False,
    mask_bits: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass with pre-computed slice tensors (mirrors
    flash_attn_flex_flash_precomputed).  The slice layout must be the same
    one used by the forward pass that produced `out` / `softmax_lse`.

    See flash_attn_flex_flash_bwd for the feature flags (softcap / reduce_kv
    / use_loop_k / inner_min_to_max / persistent_scheduler)."""
    if _flex_flash_attention is None:
        raise RuntimeError("flex_flash_attention C++ extension not loaded.")
    if inner_min_to_max is None:
        inner_min_to_max = True   # measured ~2.5-3% bwd win, all scenarios

    result = _flex_flash_attention.bwd(
        q, k, v, out, softmax_lse, dout,
        q_starts, q_ends, k_starts, k_ends,
        mask_types, row_to_slice,
        diagonal_offsets, band_widths,
        vbatch_to_slice, row_to_vbatch_start, row_to_vbatch_end,
        dq, dk, dv,
        softmax_scale,
        deterministic,
        softcap,
        reduce_kv,
        use_loop_k,
        inner_min_to_max,
        persistent_scheduler,
        0.0,        # dropout_p
        None,       # rng_state
        mask_bits,
    )
    return result[0], result[1], result[2]


# ── layout cache ───────────────────────────────────────────────────────────
# One entry per (mask shape, device).  Each entry keeps a snapshot CLONE of
# the mask content, so changes are detected even when the caller mutates
# their tensor in place.  The per-call validation is a full equality check
# (one S×S read, ~0.4ms at S=25k) — far cheaper than re-decomposing (~5ms).
_arb_layout_cache = {}


def _get_cached_layout(mask: torch.Tensor, seqlen_q: int, kblock_m: int = 128):
    """Return (slice_tensors, slices, hit).  Re-decomposes only when the
    mask content differs from the cached snapshot.  slice_tensors carries
    the packed bitmask too when the mask falls back to the SLICE_BITMASK
    path (mask_bits key, None on the slice path).  kblock_m selects the
    band grid (fwd 128 vs bwd _BWD_KBLOCK_M), so cached bwd stays bitwise
    equal to the e2e bwd path."""
    key = (mask.shape, mask.device.index, kblock_m)
    ent = _arb_layout_cache.get(key)
    if ent is not None:
        snap, tensors, slices = ent
        if torch.equal(snap, mask):
            return tensors, slices, True
    seqlen_k = mask.shape[1]
    tensors, slices = _mask_layout(mask, seqlen_q, seqlen_k, mask.device,
                                   kblock_m=kblock_m)
    _arb_layout_cache[key] = (mask.detach().clone(), tensors, slices)
    return tensors, slices, False


def flash_attn_flex_flash_cache_clear():
    """Drop all cached layouts (e.g. before memory-sensitive phases)."""
    _arb_layout_cache.clear()
    _id_layout_cache.clear()


def flash_attn_flex_flash_cached(
    q: torch.Tensor,           # (b, s_q, h, d)
    k: torch.Tensor,           # (b, s_k, h_k, d)
    v: torch.Tensor,           # (b, s_k, h_k, dv)
    mask: torch.Tensor,        # (s_q, s_k) bool — True = attend
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    inner_min_to_max: bool = False,
    persistent_scheduler: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """flash_attn_flex_flash with a cached mask decomposition.

    Semantics are identical to flash_attn_flex_flash, but the decomposition
    (+ RangeMerge layout build) runs only when the mask CONTENT changed
    since the last call with the same (shape, device).  The per-call cost
    is one S×S equality check instead of the full decomposition.

    Note: only the latest mask per (shape, device) is cached — alternating
    between two different masks of the same shape thrashes the cache.
    The cached snapshot holds one bool S×S tensor (~639MB at S=25k).
    """
    if _flex_flash_attention is None:
        raise RuntimeError(
            "flex_flash_attention C++ extension not loaded. "
            "Make sure the flex_flash_attention module is built and importable."
        )

    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert mask.dtype == torch.bool, "mask must be boolean"
    seqlen_q, seqlen_k = q.size(1), k.size(1)
    assert mask.shape == (seqlen_q, seqlen_k), \
        f"mask shape {mask.shape} != ({seqlen_q}, {seqlen_k})"

    st, slices, _hit = _get_cached_layout(mask, seqlen_q)

    result = _flex_flash_attention.fwd(
        q, k, v,
        st['q_starts'], st['q_ends'], st['k_starts'], st['k_ends'],
        st['mask_types'], st['row_to_slice'],
        st['diagonal_offsets'], st['band_widths'],
        st['vbatch_to_slice'], st['row_to_vbatch_start'],
        st['row_to_vbatch_end'],
        out,
        softmax_scale,
        softcap,
        inner_min_to_max,
        persistent_scheduler,
        _kblockm128_hint(slices, q.shape[-1]),
        _kblockn64_hint(slices, q.shape[-1]),
        0.0,        # dropout_p
        None,       # gen
        st['mask_bits'],
    )
    return result[0], result[1]


def flash_attn_flex_flash_bwd_cached(
    q: torch.Tensor,           # (b, s_q, h, d)
    k: torch.Tensor,           # (b, s_k, h_k, d)
    v: torch.Tensor,           # (b, s_k, h_k, dv)
    out: torch.Tensor,         # (b, s_q, h, dv)  — from fwd
    softmax_lse: torch.Tensor, # (b, h, s_q)      — from fwd
    dout: torch.Tensor,        # (b, s_q, h, dv)
    mask: torch.Tensor,        # (s_q, s_k) bool — True = attend
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    softcap: Optional[float] = None,
    reduce_kv: bool = False,
    use_loop_k: bool = False,
    inner_min_to_max: Optional[bool] = None,
    persistent_scheduler: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward counterpart of flash_attn_flex_flash_cached.

    Shares the same layout cache, so a fwd+bwd pair on an unchanged mask
    pays the equality check twice but the decomposition zero times.
    Feature flags mirror flash_attn_flex_flash_bwd.  As with any cached
    path, out/softmax_lse must come from a fwd run with the SAME mask
    content (the check runs against the mask passed here).
    """
    if _flex_flash_attention is None:
        raise RuntimeError(
            "flex_flash_attention C++ extension not loaded. "
            "Make sure the flex_flash_attention module is built and importable."
        )

    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert mask.dtype == torch.bool, "mask must be boolean"
    seqlen_q, seqlen_k = q.size(1), k.size(1)
    assert mask.shape == (seqlen_q, seqlen_k), \
        f"mask shape {mask.shape} != ({seqlen_q}, {seqlen_k})"
    if inner_min_to_max is None:
        inner_min_to_max = True   # mirror flash_attn_flex_flash_bwd default

    st, _slices, _hit = _get_cached_layout(mask, seqlen_q,
                                           kblock_m=_BWD_KBLOCK_M)

    result = _flex_flash_attention.bwd(
        q, k, v, out, softmax_lse, dout,
        st['q_starts'], st['q_ends'], st['k_starts'], st['k_ends'],
        st['mask_types'], st['row_to_slice'],
        st['diagonal_offsets'], st['band_widths'],
        st['vbatch_to_slice'], st['row_to_vbatch_start'],
        st['row_to_vbatch_end'],
        dq, dk, dv,
        softmax_scale,
        deterministic,
        softcap,
        reduce_kv,
        use_loop_k,
        inner_min_to_max,
        persistent_scheduler,
        0.0,        # dropout_p
        None,       # rng_state
        st['mask_bits'],
    )
    return result[0], result[1], result[2]


# ── varlen / packed-sequence (nested-tensor semantics) ─────────────────────
# Packed layout (PyTorch NestedTensor / FA varlen style): q is (1, total_q,
# h, d) with per-sample row ranges from cu_seqlens_q; k/v are (1, total_k,
# h_k, *).  Each sample is one FULL block on the diagonal of the implicit
# (total_q, total_k) mask; is_causal adds the per-sample causal diagonal.
# No mask tensor is ever materialized — the slices ARE the mask, O(B) ints.

def _varlen_slices(cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
                   is_causal: bool) -> Tuple[list, int, int]:
    """Block-diagonal slices from cu_seqlens, plus (total_q, total_k).

    Per sample s with rows [q0, q0+Lq) and cols [k0, k0+Lk):
      is_causal=False → SliceInfo(q0, q0+Lq, k0, k0+Lk, FULL)
      is_causal=True  → SliceInfo(q0, q0+Lq, k0, k0+Lk, CAUSAL,
                          diagonal_offset = k0 - q0)
    CAUSAL keeps k <= q + offset in GLOBAL coords, i.e. local k' <= local q'
    (TopLeft alignment).  Zero-length samples contribute nothing.
    """
    cq = cu_seqlens_q.detach().to('cpu', torch.int32).tolist()
    ck = cu_seqlens_k.detach().to('cpu', torch.int32).tolist()
    assert len(cq) == len(ck), \
        f"cu_seqlens_q/k length mismatch: {len(cq)} vs {len(ck)}"
    slices = []
    for q0, q1, k0, k1 in zip(cq[:-1], cq[1:], ck[:-1], ck[1:]):
        if q1 <= q0 or k1 <= k0:
            continue
        if is_causal:
            slices.append(SliceInfo(q0, q1, k0, k1, SLICE_CAUSAL, k0 - q0, 0))
        else:
            slices.append(SliceInfo(q0, q1, k0, k1, SLICE_FULL, 0, 0))
    return slices, cq[-1], ck[-1]


def flash_attn_varlen_flex_flash(
    q: torch.Tensor,           # (1, total_q, h, d)
    k: torch.Tensor,           # (1, total_k, h_k, d)
    v: torch.Tensor,           # (1, total_k, h_k, dv)
    cu_seqlens_q: torch.Tensor,  # int32 [B+1], on CPU or CUDA
    cu_seqlens_k: torch.Tensor,  # int32 [B+1]
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    inner_min_to_max: bool = False,
    persistent_scheduler: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Variable-length packed attention (NestedTensor / cu_seqlens semantics).

    Each packed sample attends only to its own K range (block-diagonal
    mask), optionally causal per sample.  Equivalent to running
    flash_attn_flex_flash with the implicit block-diagonal bool mask, but the
    slices are built directly from cu_seqlens — O(B) metadata, no S² mask.
    Zero kernel changes: reuses the standard slice machinery end to end.

    Returns (out, softmax_lse) like flash_attn_flex_flash.
    """
    if _flex_flash_attention is None:
        raise RuntimeError("flex_flash_attention C++ extension not loaded.")

    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert q.size(0) == 1 and k.size(0) == 1 and v.size(0) == 1, \
        "varlen entry expects packed tensors with batch dimension 1"
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32, \
        "cu_seqlens must be int32"

    slices, total_q, total_k = _varlen_slices(cu_seqlens_q, cu_seqlens_k, is_causal)
    assert total_q == q.size(1), \
        f"cu_seqlens_q total {total_q} != q seqlen {q.size(1)}"
    assert total_k == k.size(1), \
        f"cu_seqlens_k total {total_k} != k seqlen {k.size(1)}"
    assert len(slices) > 0, "cu_seqlens describe no non-empty sample"

    tensors = build_merge_layout(slices, total_q, kBlockM=128, device=q.device)

    result = _flex_flash_attention.fwd(
        q, k, v,
        tensors['q_starts'], tensors['q_ends'], tensors['k_starts'],
        tensors['k_ends'], tensors['mask_types'], tensors['row_to_slice'],
        tensors['diagonal_offsets'], tensors['band_widths'],
        tensors['vbatch_to_slice'], tensors['row_to_vbatch_start'],
        tensors['row_to_vbatch_end'],
        out,
        softmax_scale,
        softcap,
        inner_min_to_max,
        persistent_scheduler,
        _kblockm128_hint(slices, q.shape[-1]),
        _kblockn64_hint(slices, q.shape[-1]),
        0.0,        # dropout_p
        None,       # gen
        None,       # mask_bits (block-diagonal slices are algebraic)
    )
    return result[0], result[1]


def flash_attn_varlen_flex_flash_bwd(
    q: torch.Tensor,           # (1, total_q, h, d)
    k: torch.Tensor,           # (1, total_k, h_k, d)
    v: torch.Tensor,           # (1, total_k, h_k, dv)
    out: torch.Tensor,         # (1, total_q, h, dv) — from fwd
    softmax_lse: torch.Tensor, # (1, h, total_q)     — from fwd
    dout: torch.Tensor,        # (1, total_q, h, dv)
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    softcap: Optional[float] = None,
    reduce_kv: bool = False,
    use_loop_k: bool = False,
    inner_min_to_max: bool = False,
    persistent_scheduler: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward for flash_attn_varlen_flex_flash.

    Rebuilds the same block-diagonal slice layout from cu_seqlens (is_causal
    must match the fwd call).  Feature flags mirror flash_attn_flex_flash_bwd.
    """
    if _flex_flash_attention is None:
        raise RuntimeError("flex_flash_attention C++ extension not loaded.")

    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert q.size(0) == 1 and k.size(0) == 1 and v.size(0) == 1, \
        "varlen entry expects packed tensors with batch dimension 1"
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32, \
        "cu_seqlens must be int32"

    slices, total_q, total_k = _varlen_slices(cu_seqlens_q, cu_seqlens_k, is_causal)
    assert total_q == q.size(1) and total_k == k.size(1), \
        "cu_seqlens totals must match packed q/k seqlens"
    assert len(slices) > 0, "cu_seqlens describe no non-empty sample"

    tensors = build_merge_layout(slices, total_q, kBlockM=128, device=q.device)

    result = _flex_flash_attention.bwd(
        q, k, v, out, softmax_lse, dout,
        tensors['q_starts'], tensors['q_ends'], tensors['k_starts'],
        tensors['k_ends'], tensors['mask_types'], tensors['row_to_slice'],
        tensors['diagonal_offsets'], tensors['band_widths'],
        tensors['vbatch_to_slice'], tensors['row_to_vbatch_start'],
        tensors['row_to_vbatch_end'],
        dq, dk, dv,
        softmax_scale,
        deterministic,
        softcap,
        reduce_kv,
        use_loop_k,
        inner_min_to_max,
        persistent_scheduler,
        0.0,        # dropout_p
        None,       # rng_state
        None,       # mask_bits
    )
    return result[0], result[1], result[2]


# ── per-(batch, head) heterogeneous masks (P3) ─────────────────────────────
# One bool mask per (batch, q-head) pair.  Masks with identical content are
# deduplicated into layout groups; all group layouts are flat-concatenated
# (build_bh_layout) and the kernel rebases its CTA-local slice params per
# work tile (rebase_slices in attn_slice.h).  Zero kernel changes beyond the
# rebase; shared-mask calls degrade to the ordinary single-layout path.

def _group_bh_masks(mask_bh: torch.Tensor):
    """Deduplicate (b, h) masks by content.

    Returns (unique_masks: list of (seqlen_q, seqlen_k) bool tensors,
             assignment: group id per flattened (b, h), length B*H).
    Masks are bucketed by visible-element count first, so torch.equal only
    runs between content-plausible pairs (B*H is small; the expensive case
    "all masks identical" resolves with one comparison per pair).
    """
    n = mask_bh.shape[0] * mask_bh.shape[1]
    flat = mask_bh.reshape(n, mask_bh.shape[2], mask_bh.shape[3])
    counts = flat.sum(dim=(1, 2)).tolist()
    unique, unique_counts, assignment = [], [], []
    for i in range(n):
        gid = None
        for j, uc in enumerate(unique_counts):
            if uc == counts[i] and torch.equal(unique[j], flat[i]):
                gid = j
                break
        if gid is None:
            gid = len(unique)
            unique.append(flat[i])
            unique_counts.append(counts[i])
        assignment.append(gid)
    return unique, assignment


def _bh_layout(mask_bh: torch.Tensor, device: torch.device):
    """(single_mask_or_None, flat_tensors_or_None, bh_to_group_or_None).

    With a single unique mask the ordinary single-layout path is used
    (bit-identical to flash_attn_flex_flash); otherwise the flat P3 layout
    is built from one decomposition per unique mask.
    """
    uniques, assignment = _group_bh_masks(mask_bh)
    if len(uniques) == 1:
        return uniques[0], None, None
    seqlen_q = mask_bh.shape[2]
    seqlen_k = mask_bh.shape[3]
    group_layouts = []
    for m in uniques:
        t, _s = _mask_layout(m, seqlen_q, seqlen_k, device)
        group_layouts.append(t)
    flat = build_bh_layout(group_layouts, seqlen_q, device)
    bh_to_group = torch.tensor(assignment, dtype=torch.int32, device=device)
    return None, flat, bh_to_group


def flash_attn_flex_flash_bh(
    q: torch.Tensor,           # (b, s_q, h, d)
    k: torch.Tensor,           # (b, s_k, h_k, d)
    v: torch.Tensor,           # (b, s_k, h_k, dv)
    mask_bh: torch.Tensor,     # (b, h, s_q, s_k) bool — per (batch, q-head)
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    inner_min_to_max: bool = False,
    persistent_scheduler: bool = False,
    dropout_p: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, ...]:
    """flash_attn_flex_flash with an independent mask per (batch, q-head).

    mask_bh[b, h] is the bool mask for batch b and QUERY head h (GQA:
    k-heads share nothing here — the mask keys attention rows, which are
    per q-head).  Masks that repeat across (b, h) pairs share one layout
    group; when ALL pairs share one mask this degrades exactly to
    flash_attn_flex_flash.

    Stage A restrictions (enforced by the C++ API): no deterministic bwd,
    no use_loop_k/reduce_kv bwd with heterogeneous layouts.  Heterogeneous
    calls also force the conservative kBlockM=64 / kBlockN>=96 dispatch
    (per-group tile hints would need per-CTA dispatch, a Stage B item).

    Returns (out, lse[, rng_state]) like flash_attn_flex_flash.
    """
    if _flex_flash_attention is None:
        raise RuntimeError("flex_flash_attention C++ extension not loaded.")

    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert mask_bh.is_cuda and mask_bh.dtype == torch.bool, \
        "mask_bh must be a CUDA bool tensor"
    batch, seqlen_q, num_heads = q.shape[0], q.shape[1], q.shape[2]
    seqlen_k = k.size(1)
    assert mask_bh.shape == (batch, num_heads, seqlen_q, seqlen_k), \
        f"mask_bh shape {mask_bh.shape} != ({batch}, {num_heads}, " \
        f"{seqlen_q}, {seqlen_k})"

    single, flat, bh_to_group = _bh_layout(mask_bh, q.device)
    if single is not None:
        return flash_attn_flex_flash(
            q, k, v, single, softmax_scale, out, softcap,
            inner_min_to_max, persistent_scheduler, dropout_p, generator)

    result = _flex_flash_attention.fwd(
        q, k, v,
        flat['q_starts'], flat['q_ends'], flat['k_starts'], flat['k_ends'],
        flat['mask_types'], flat['row_to_slice'],
        flat['diagonal_offsets'], flat['band_widths'],
        flat['vbatch_to_slice'], flat['row_to_vbatch_start'],
        flat['row_to_vbatch_end'],
        out,
        softmax_scale,
        softcap,
        inner_min_to_max,
        persistent_scheduler,
        False,      # use_kblockm128: conservative kbm64 for hetero layouts
        False,      # kblockn64
        dropout_p,
        generator,
        flat['mask_bits'],
        bh_to_group, flat['group_slice_offsets'], flat['group_vb_offsets'],
    )
    if dropout_p > 0.0:
        return result[0], result[1], result[2]
    return result[0], result[1]


def flash_attn_flex_flash_bh_bwd(
    q: torch.Tensor,           # (b, s_q, h, d)
    k: torch.Tensor,           # (b, s_k, h_k, d)
    v: torch.Tensor,           # (b, s_k, h_k, dv)
    out: torch.Tensor,         # (b, s_q, h, dv)  — from fwd
    softmax_lse: torch.Tensor, # (b, h, s_q)      — from fwd
    dout: torch.Tensor,        # (b, s_q, h, dv)
    mask_bh: torch.Tensor,     # (b, h, s_q, s_k) bool — same as fwd
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    inner_min_to_max: Optional[bool] = None,
    persistent_scheduler: bool = False,
    dropout_p: float = 0.0,
    rng_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward for flash_attn_flex_flash_bh.

    Rebuilds the same per-(b, h) layout grouping as the forward (mask_bh
    must have the same content).  Stage A: deterministic accumulation and
    use_loop_k/reduce_kv are rejected for heterogeneous layouts by the C++
    API; single-group calls delegate to flash_attn_flex_flash_bwd.
    """
    if _flex_flash_attention is None:
        raise RuntimeError("flex_flash_attention C++ extension not loaded.")
    if inner_min_to_max is None:
        inner_min_to_max = True   # measured ~2.5-3% bwd win, all scenarios

    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert mask_bh.is_cuda and mask_bh.dtype == torch.bool, \
        "mask_bh must be a CUDA bool tensor"
    if dropout_p > 0.0:
        assert rng_state is not None, \
            "rng_state (returned by fwd) is required when dropout_p > 0"
    batch, seqlen_q, num_heads = q.shape[0], q.shape[1], q.shape[2]
    seqlen_k = k.size(1)
    assert mask_bh.shape == (batch, num_heads, seqlen_q, seqlen_k), \
        f"mask_bh shape {mask_bh.shape} != ({batch}, {num_heads}, " \
        f"{seqlen_q}, {seqlen_k})"

    single, flat, bh_to_group = _bh_layout(mask_bh, q.device)
    if single is not None:
        return flash_attn_flex_flash_bwd(
            q, k, v, out, softmax_lse, dout, single, softmax_scale,
            dq, dk, dv, False, softcap, False, False,
            inner_min_to_max, persistent_scheduler, dropout_p, rng_state)

    result = _flex_flash_attention.bwd(
        q, k, v, out, softmax_lse, dout,
        flat['q_starts'], flat['q_ends'], flat['k_starts'], flat['k_ends'],
        flat['mask_types'], flat['row_to_slice'],
        flat['diagonal_offsets'], flat['band_widths'],
        flat['vbatch_to_slice'], flat['row_to_vbatch_start'],
        flat['row_to_vbatch_end'],
        dq, dk, dv,
        softmax_scale,
        False,      # deterministic (Stage A: rejected for hetero layouts)
        softcap,
        False,      # reduce_kv
        False,      # use_loop_k
        inner_min_to_max,
        persistent_scheduler,
        dropout_p,
        rng_state,
        flat['mask_bits'],
        bh_to_group, flat['group_slice_offsets'], flat['group_vb_offsets'],
    )
    return result[0], result[1], result[2]


def flash_attn_flex_flash_bh_f32(
    q: torch.Tensor,           # (b, s_q, h, d)  float32
    k: torch.Tensor,           # (b, s_k, h_k, d) float32
    v: torch.Tensor,           # (b, s_k, h_k, dv) float32
    mask_bh: torch.Tensor,     # (b, h, s_q, s_k) bool — per (batch, q-head)
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    inner_min_to_max: bool = False,
    persistent_scheduler: bool = False,
    dropout_p: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, ...]:
    """fp32 variant of flash_attn_flex_flash_bh — ISOLATED kernel set.

    One bool mask per (batch, q-head) pair, identical layout semantics to
    flash_attn_flex_flash_bh (content-dedup groups, flat-concatenated
    layouts, kernel-side rebase).  All inputs/outputs float32; TF32
    compute — see flash_attn_flex_flash_f32 for accuracy notes.

    Single-group calls degrade exactly to flash_attn_flex_flash_f32.
    Returns (out, lse[, rng_state]) like flash_attn_flex_flash_f32.
    """
    _check_f32_available()

    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert q.dtype == torch.float32 and k.dtype == torch.float32 \
        and v.dtype == torch.float32, "bh fwd_f32 requires float32 q/k/v"
    assert mask_bh.is_cuda and mask_bh.dtype == torch.bool, \
        "mask_bh must be a CUDA bool tensor"
    batch, seqlen_q, num_heads = q.shape[0], q.shape[1], q.shape[2]
    seqlen_k = k.size(1)
    assert mask_bh.shape == (batch, num_heads, seqlen_q, seqlen_k), \
        f"mask_bh shape {mask_bh.shape} != ({batch}, {num_heads}, " \
        f"{seqlen_q}, {seqlen_k})"

    single, flat, bh_to_group = _bh_layout(mask_bh, q.device)
    if single is not None:
        return flash_attn_flex_flash_f32(
            q, k, v, single, softmax_scale, out, softcap,
            inner_min_to_max, persistent_scheduler, dropout_p, generator)

    result = _flex_flash_attention.fwd_f32(
        q, k, v,
        flat['q_starts'], flat['q_ends'], flat['k_starts'], flat['k_ends'],
        flat['mask_types'], flat['row_to_slice'],
        flat['diagonal_offsets'], flat['band_widths'],
        flat['vbatch_to_slice'], flat['row_to_vbatch_start'],
        flat['row_to_vbatch_end'],
        out,
        softmax_scale,
        softcap,
        inner_min_to_max,
        persistent_scheduler,
        False,      # use_kblockm128: conservative kbm64 for hetero layouts
        False,      # kblockn64
        dropout_p,
        generator,
        flat['mask_bits'],
        bh_to_group, flat['group_slice_offsets'], flat['group_vb_offsets'],
    )
    if dropout_p > 0.0:
        return result[0], result[1], result[2]
    return result[0], result[1]


def flash_attn_flex_flash_bh_bwd_f32(
    q: torch.Tensor,           # (b, s_q, h, d)  float32
    k: torch.Tensor,           # (b, s_k, h_k, d) float32
    v: torch.Tensor,           # (b, s_k, h_k, dv) float32
    out: torch.Tensor,         # from flash_attn_flex_flash_bh_f32
    softmax_lse: torch.Tensor, # from flash_attn_flex_flash_bh_f32
    dout: torch.Tensor,
    mask_bh: torch.Tensor,     # (b, h, s_q, s_k) bool — same as fwd
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    inner_min_to_max: Optional[bool] = None,
    persistent_scheduler: bool = False,
    dropout_p: float = 0.0,
    rng_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward for flash_attn_flex_flash_bh_f32.

    Rebuilds the same per-(b, h) layout grouping as the forward (mask_bh
    must have the same content).  Stage A: deterministic accumulation and
    use_loop_k/reduce_kv are rejected for heterogeneous layouts by the
    C++ API; single-group calls delegate to flash_attn_flex_flash_bwd_f32.
    """
    _check_f32_available()
    if inner_min_to_max is None:
        inner_min_to_max = True   # measured ~2.5-3% bwd win, all scenarios

    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be on CUDA"
    assert q.dtype == torch.float32 and k.dtype == torch.float32 \
        and v.dtype == torch.float32, "bh bwd_f32 requires float32 inputs"
    assert mask_bh.is_cuda and mask_bh.dtype == torch.bool, \
        "mask_bh must be a CUDA bool tensor"
    if dropout_p > 0.0:
        assert rng_state is not None, \
            "rng_state (returned by fwd) is required when dropout_p > 0"
    batch, seqlen_q, num_heads = q.shape[0], q.shape[1], q.shape[2]
    seqlen_k = k.size(1)
    assert mask_bh.shape == (batch, num_heads, seqlen_q, seqlen_k), \
        f"mask_bh shape {mask_bh.shape} != ({batch}, {num_heads}, " \
        f"{seqlen_q}, {seqlen_k})"

    single, flat, bh_to_group = _bh_layout(mask_bh, q.device)
    if single is not None:
        return flash_attn_flex_flash_bwd_f32(
            q, k, v, out, softmax_lse, dout, single, softmax_scale,
            dq, dk, dv, False, softcap, False, False,
            inner_min_to_max, persistent_scheduler, dropout_p, rng_state)

    result = _flex_flash_attention.bwd_f32(
        q, k, v, out, softmax_lse, dout,
        flat['q_starts'], flat['q_ends'], flat['k_starts'], flat['k_ends'],
        flat['mask_types'], flat['row_to_slice'],
        flat['diagonal_offsets'], flat['band_widths'],
        flat['vbatch_to_slice'], flat['row_to_vbatch_start'],
        flat['row_to_vbatch_end'],
        dq, dk, dv,
        softmax_scale,
        False,      # deterministic (Stage A: rejected for hetero layouts)
        softcap,
        False,      # reduce_kv
        False,      # use_loop_k
        inner_min_to_max,
        persistent_scheduler,
        dropout_p,
        rng_state,
        flat['mask_bits'],
        bh_to_group, flat['group_slice_offsets'], flat['group_vb_offsets'],
    )
    return result[0], result[1], result[2]


# ---------------------------------------------------------------------------
# Fused decomposition layer (l5/l6): the whole mask decomposition now runs
# device-side as one op (flex_flash_attention.decompose_mask), producing a fixed-
# shape desc pack consumable by the fwd/bwd ops — no host syncs, no data-
# dependent shapes, so the compute path is torch.compile-capturable.
#
# desc pack = (slices int32[7,CAP], rows int32[3,sq], vbatch int32[CAP],
#              counters int32[4] = [num_slices, num_vbatches, supported,
#              n_layers]); CAP = sq*17+64.  Valid content lives in the
#              first counters[0] slices / counters[1] vbatches.
# ---------------------------------------------------------------------------

FWD_KBLOCK_M = 128          # fwd Q-tile; layout bands align to it

MaskDecomp = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def decompose_mask_fused(mask: torch.Tensor,
                         kblock_m: int = FWD_KBLOCK_M) -> MaskDecomp:
    """Decompose `mask` into the kernel's slice layout, entirely on device.

    Returns the desc pack (see module note).  Whether the mask fits the
    layered-interval envelope is reported by counters[2] on device — NO
    host sync happens here; use mask_supported() for a host-side answer.
    """
    if _flex_flash_attention is None:
        raise RuntimeError("flex_flash_attention C++ extension not loaded.")
    assert mask.dtype == torch.bool, "mask must be boolean"
    if not mask.is_cuda:
        mask = mask.cuda()
    if not mask.is_contiguous():
        mask = mask.contiguous()
    return _flex_flash_attention.decompose_mask(mask, kblock_m)


def mask_supported(mask: torch.Tensor,
                   kblock_m: int = FWD_KBLOCK_M) -> bool:
    """Host-side answer to 'is this mask decomposable?' (one small D2H).

    Strategy A contract: honest — masks beyond the layered envelope (per-row
    interval count over the 16-layer cap) report False; callers fall back to
    their own exact path (math SDPA here).  Hits the C++ identity cache, so
    a following decompose_mask_fused call is free.
    """
    return bool(decompose_mask_fused(mask, kblock_m)[3][2].item())


def _desc_args(desc: MaskDecomp):
    """desc pack -> the 11 positional layout args of fwd/bwd."""
    slices, rows, vbatch, counters = desc
    n, nvb = int(counters[0]), int(counters[1])
    return (slices[0, :n], slices[1, :n], slices[2, :n], slices[3, :n],
            slices[4, :n], rows[0], slices[5, :n], slices[6, :n],
            vbatch[:nvb], rows[1], rows[2])


def _desc_args_full(desc: MaskDecomp):
    """Like _desc_args but WITHOUT slicing (no host sync, no data-dependent
    shapes).  WARNING: the kernel treats q_starts.size(0) as the slice count
    and does NOT skip the zero-padded CAP tail — passing full-length descs
    costs ~25x on bwd (phantom-slice scheduling).  Kept for reference only;
    production paths must use _desc_args.
    """
    slices, rows, vbatch, _counters = desc
    return (slices[0], slices[1], slices[2], slices[3], slices[4],
            rows[0], slices[5], slices[6], vbatch, rows[1], rows[2])


class _FlexFlashAttentionAttnFunc(torch.autograd.Function):
    """Differentiable glue of the fwd/bwd ops.

    fwd consumes a desc decomposed on the 128-row fwd tile grid; bwd re-
    decomposes on the _BWD_KBLOCK_M grid (its kernel runs different Q-tile
    sizes, and aligned bands save ~25% tile-pair work — same convention as
    the legacy flash_attn_flex_flash_bwd).  Both decompositions hit the C++
    identity cache on repeat calls with the same mask tensor.
    """

    @staticmethod
    def forward(ctx, q, k, v, mask, desc_slices, desc_rows, desc_vbatch,
                desc_counters, softmax_scale):
        desc = (desc_slices, desc_rows, desc_vbatch, desc_counters)
        # Trimmed args (_desc_args): the kernel uses q_starts.size(0) as the
        # slice count, so the zero-padded CAP tail must not be passed through
        # (phantom-slice scheduling cost).  Costs one cached host sync.
        out, lse, rng = _flex_flash_attention.fwd(
            q, k, v, *_desc_args(desc),
            None,               # out
            softmax_scale,
            None,               # softcap
            False,              # inner_min_to_max (fwd: max->min is faster)
            False,              # persistent_scheduler
            None,               # use_kblockm128: kernel-side heuristic
            False,              # kblockn64
            0.0,                # dropout_p
            None,               # gen
            None,               # mask_bits (desc path has no bitmask slices)
        )
        ctx.save_for_backward(q, k, v, out, lse, rng, mask, *desc)
        ctx.softmax_scale = softmax_scale
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        (q, k, v, out, lse, rng, mask, *desc) = ctx.saved_tensors
        del desc  # fwd desc is 128-aligned; bwd wants its own band grid
        if dout.stride(-1) != 1:  # e.g. grad of a transposed (BHSD) view
            dout = dout.contiguous()
        desc_b = decompose_mask_fused(mask, _BWD_KBLOCK_M)
        # Trimmed args: see the note in forward() — full CAP tails cost ~25x
        # on bwd (phantom slices are scheduled as real work).
        dq, dk, dv = _flex_flash_attention.bwd(
            q, k, v, out, lse, dout, *_desc_args(desc_b),
            None, None, None,   # dq/dk/dv preallocation
            ctx.softmax_scale,
            False,              # deterministic
            None,               # softcap
            False,              # reduce_kv
            False,              # use_loop_k
            True,               # inner_min_to_max (measured bwd win)
            False,              # persistent_scheduler
            0.0,                # dropout_p
            rng,                # rng_state
            None,               # mask_bits
        )
        return dq, dk, dv, None, None, None, None, None, None


def flash_attn_flex_flash_desc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable flex flash attention over the fused desc path.

    Decomposes `mask` on device (128-row fwd tile grid), runs the fwd op
    through an autograd.Function; .backward() runs the bwd op with a bwd-
    tuned re-decomposition.  torch.compile traces through it via Compiled
    Autograd (the ops carry Meta impls).

    Raises MaskNotSupportedError when the mask is outside the layered
    envelope (counters[2] == 0) — check with mask_supported() first if you
    need a non-raising probe.
    """
    desc = decompose_mask_fused(mask, FWD_KBLOCK_M)
    if not bool(desc[3][2].item()):
        raise MaskNotSupportedError(
            "mask exceeds the 16-layer decomposition envelope")
    return _FlexFlashAttentionAttnFunc.apply(q, k, v, mask, *desc, softmax_scale)


class FlexFlashAttentionSDPABackend:
    """SDPA-style backend callable for torch's scaled_dot_product_attention.

    Accepts the SDPA argument order — (q, k, v, attn_mask=None, dropout_p=
    0.0, is_causal=False, scale=None) with (b, h, s, d) BHSD layout — and
    routes to the fused flex flash attention kernel.  attn_mask is a BOOLEAN
    (s_q, s_k) tensor shared across batch and heads (True = attend).

    Strategy A contract: masks outside the decomposition envelope do NOT
    silently degrade — the backend falls back to torch's math SDPA with an
    equivalent additive mask (exact, no bitmask path in the desc layer).

    Note: dropout_p > 0 and is_causal=True are rejected (the desc path is
    dropout-free; causal masks should be passed as an explicit mask).
    """

    def __call__(self, q, k, v, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, **_ignored):
        assert attn_mask is not None, \
            "flex flash attention backend requires an explicit boolean attn_mask"
        assert not is_causal, "pass causal masks as an explicit attn_mask"
        assert dropout_p == 0.0, "flex flash attention backend does not dropout"
        assert attn_mask.dtype == torch.bool, "attn_mask must be boolean"
        if mask_supported(attn_mask):
            # BHSD -> BSHD views; the kernels only require the last dim
            # contiguous, so no copies here.
            out, _lse = _FlexFlashAttentionAttnFunc.apply(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_mask,
                *decompose_mask_fused(attn_mask, FWD_KBLOCK_M),
                scale)
            return out.transpose(1, 2)
        # Envelope exceeded: exact math fallback (SDPA additive convention:
        # masked positions get -inf).
        additive = torch.zeros_like(attn_mask, dtype=q.dtype)
        additive.masked_fill_(~attn_mask, float("-inf"))
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=additive, dropout_p=dropout_p,
            is_causal=False, scale=scale)


flex_flash_attention_sdpa_backend = FlexFlashAttentionSDPABackend()
