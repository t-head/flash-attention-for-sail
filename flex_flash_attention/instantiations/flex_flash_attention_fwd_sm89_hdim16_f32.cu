// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Explicit instantiations for the ISOLATED fp32 (TF32 compute) flex flash
// attention forward, hdim=16.  Compiled by hgcc (GPU compiler).
//
// EXPERIMENTAL: narrow-headdim probe — D=8 previously padded to the 64
// bucket (8x smem/bandwidth waste).  TF32 MMA granularity (m16n8k8) and
// cp.async 16B chunks are both satisfied at kHeadDim=8, but this is below
// the upstream FA3 sm80 minimum (64), so CuTe smem layout divisibility is
// unproven territory.

#include "flex_flash_attention/csrc/flex_flash_attention_fwd_launch_template_f32.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM16
template void run_mha_flex_flash_fwd_f32_<89, 16, 16, false>(FlexFlashAttentionFwdParams &params, hggcStream_t stream);
#endif
#endif
