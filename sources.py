"""Build-source manifest for the flex flash attention module.

Kept inside the module (rather than inlined into hopper/setup.py) so that the
whole flex flash attention feature touches the shared build script in exactly one
place — an import plus one call — which keeps upstream FA3 setup.py merges
conflict-free.

Layout: C++ headers and the API translation unit live in flex_flash_attention/csrc/;
the instantiation TUs stay in flex_flash_attention/instantiations/, mirroring
hopper/instantiations/.  One lightweight dispatch TU plus one heavy TU per head
dim, so the five head dims compile in parallel.
"""

_DIR = "instantiations"

# Head-dim buckets the kernels are instantiated for; must stay in sync with
# round_up_headdim_arb() in flex_flash_attention_api.cpp.
HEAD_DIMENSIONS = [64, 96, 128, 192, 256]


def _sources(kind, disabled_hdims):
    """`kind` is "fwd" or "bwd"; returns dispatch TU + enabled per-hdim TUs."""
    out = [f"{_DIR}/flex_flash_attention_{kind}_dispatch.cu"]
    # Narrow-headdim bucket ported from the fp32 set (validated there):
    # hdim32 for fwd AND bwd (no fwd hdim16 TU — the 16-bit fwd kernel at
    # kHeadDim=16 trips a PPU hardware exception, AIU_ld TSM out of range;
    # bwd's MMA fragment geometry rejects width 16 as well).  The API
    # rounds headdims <=32 here.  Not part of HEAD_DIMENSIONS so the
    # standard bucket list stays untouched.
    out.append(f"{_DIR}/flex_flash_attention_{kind}_sm89_hdim32.cu")
    out += [f"{_DIR}/flex_flash_attention_{kind}_sm89_hdim{h}.cu"
            for h in HEAD_DIMENSIONS if h not in disabled_hdims]
    return out


def _sources_f32(disabled_hdims):
    """ISOLATED fp32 (TF32 compute) kernel set: one shared dispatch TU plus
    one heavy TU per head dim, mirroring the bf16/fp16 layout."""
    out = [f"{_DIR}/flex_flash_attention_f32_dispatch.cu"]
    # EXPERIMENTAL narrow-headdim TUs (D<=16, minimal padding); not part of
    # HEAD_DIMENSIONS so the 16-bit set stays untouched.  hdim8 was tried
    # and produces silently wrong results — 16 is the minimum safe width.
    out.append(f"{_DIR}/flex_flash_attention_fwd_sm89_hdim16_f32.cu")
    # EXPERIMENTAL: fwd hdim32 bucket — D in (16, 32] previously padded to
    # the 64 bucket (up to 3.76x waste).  Validated numerically against the
    # hdim64 bucket.
    out.append(f"{_DIR}/flex_flash_attention_fwd_sm89_hdim32_f32.cu")
    # No bwd hdim16 TU: at width 16 the per-thread MMA fragment (8 elems)
    # can never match the 4-wide accum copy partitioning (probed 2026-08:
    # non-swap AtomLayout=8 and swapAB/kBlockN variants all hit the same
    # 2:1 static_assert).  bwd's narrow bucket is hdim32 instead.
    out.append(f"{_DIR}/flex_flash_attention_bwd_sm89_hdim32_f32.cu")
    out += [f"{_DIR}/flex_flash_attention_fwd_sm89_hdim{h}_f32.cu"
            for h in HEAD_DIMENSIONS if h not in disabled_hdims]
    out += [f"{_DIR}/flex_flash_attention_bwd_sm89_hdim{h}_f32.cu"
            for h in HEAD_DIMENSIONS if h not in disabled_hdims]
    return out


def flex_flash_sources(disabled_hdims=(), disable_backward=False,
                      disable_sm8x=False):
    """All translation units of the flex flash attention module (API + fwd + bwd).

    Args:
        disabled_hdims: head dims to skip, e.g. (96, 192) when the matching
            FLASHATTENTION_DISABLE_HDIM* flags are set.
        disable_backward: drop the backward TUs.
        disable_sm8x: drop the SM8x-only backward TUs (the forward dispatch TU
            is always built, matching the base setup.py behaviour).

    The fp32 (TF32 compute) kernel set is mandatory — the API entry points
    reference it unconditionally.
    """
    srcs = ["csrc/flex_flash_attention_api.cpp",
            # Note: the aten SDPA glue TU (flex_flash_attention/csrc/sdpa_glue.cpp,
            # implementing flex_flash_attention::sdpa_fwd/sdpa_bwd from
            # include/flex_flash_attention_sdpa.h) is deliberately NOT built into
            # this wheel: the pytorch fork builds it together with the kernels
            # into libflex_flash_attention.so via build_lib.py, driven from its
            # cmake/flex_flash_attention.cmake, and links it into torch_cuda.
            # Fused mask-decomposition ops: row_stats / peel (+ layered
            # flags), segmentation, classification, and the merge/align/
            # vbatch layout stage — lightweight standalone TUs, always
            # built; together they implement the decompose_mask op.
            "csrc/mask_decomp_kernels.cu",
            "csrc/mask_decomp_segment.cu",
            "csrc/mask_decomp_classify.cu",
            "csrc/mask_decomp_layout.cu"]
    srcs += _sources("fwd", disabled_hdims)
    if not disable_backward and not disable_sm8x:
        srcs += _sources("bwd", disabled_hdims)
    srcs += _sources_f32(disabled_hdims)
    return srcs
