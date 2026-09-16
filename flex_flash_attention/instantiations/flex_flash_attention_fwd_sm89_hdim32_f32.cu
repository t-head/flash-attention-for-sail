// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Explicit instantiations for the ISOLATED fp32 (TF32 compute) flex flash
// attention forward, hdim=32.  Compiled by hgcc (GPU compiler).
//
// EXPERIMENTAL narrow-headdim bucket: D in (16, 32] previously padded to
// the 64 bucket (up to 3.76x compute waste).  Shares the common fwd tile
// dispatch (no per-width config knob), validated numerically against the
// hdim64 bucket.

#include "flex_flash_attention/csrc/flex_flash_attention_fwd_launch_template_f32.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
template void run_mha_flex_flash_fwd_f32_<89, 32, 32, false>(FlexFlashAttentionFwdParams &params, hggcStream_t stream);
#endif
