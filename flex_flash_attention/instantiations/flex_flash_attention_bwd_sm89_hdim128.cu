// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Explicit instantiations for flex flash attention backward, hdim=128.
// Compiled by hgcc (GPU compiler), not by the host C++ compiler.

#include "flex_flash_attention/csrc/flex_flash_attention_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM128
template void run_mha_flex_flash_bwd_<89, cutlass::bfloat16_t, 128>(FlexFlashAttentionBwdParams &params, hggcStream_t stream);
template void run_mha_flex_flash_bwd_<89, cutlass::half_t, 128>(FlexFlashAttentionBwdParams &params, hggcStream_t stream);
#endif
#endif
