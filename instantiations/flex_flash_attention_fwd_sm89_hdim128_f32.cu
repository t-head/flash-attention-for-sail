// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Explicit instantiations for the ISOLATED fp32 (TF32 compute) flex flash
// attention forward, hdim=128.  Compiled by hgcc (GPU compiler).

#include "csrc/flex_flash_attention_fwd_launch_template_f32.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM128
template void run_mha_flex_flash_fwd_f32_<89, 128, 128, false>(FlexFlashAttentionFwdParams &params, hggcStream_t stream);
#endif
#endif
