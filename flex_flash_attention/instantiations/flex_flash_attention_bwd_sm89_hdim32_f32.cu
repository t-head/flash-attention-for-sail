// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// ISOLATED fp32 (TF32 compute) flex flash attention backward, hdim=32.
// EXPERIMENTAL narrow-headdim bucket: 32 is the narrowest bwd width the
// generic (CVT-free) path can express — per-warp MMA N slice stays a
// multiple of 16 with the legacy atom layout (32/2), while 16 fails it.
#include "flex_flash_attention/csrc/flex_flash_attention_bwd_launch_template_f32.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
template void run_mha_flex_flash_bwd_<89, float, 32>(FlexFlashAttentionBwdParams &params, hggcStream_t stream);
#endif
