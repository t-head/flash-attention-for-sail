// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// ISOLATED fp32 (TF32 compute) flex flash attention backward, hdim=128.
#include "flex_flash_attention/csrc/flex_flash_attention_bwd_launch_template_f32.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM128
template void run_mha_flex_flash_bwd_<89, float, 128>(FlexFlashAttentionBwdParams &params, hggcStream_t stream);
#endif
#endif
