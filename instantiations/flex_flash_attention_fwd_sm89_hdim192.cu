// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Explicit instantiations for flex flash attention forward, hdim=192.
// Compiled by hgcc (GPU compiler), not by the host C++ compiler.

#include "csrc/flex_flash_attention_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM192
template void run_mha_flex_flash_fwd_<89, cutlass::bfloat16_t, 192, 192, false>(FlexFlashAttentionFwdParams &params, hggcStream_t stream);
template void run_mha_flex_flash_fwd_<89, cutlass::half_t, 192, 192, false>(FlexFlashAttentionFwdParams &params, hggcStream_t stream);
#endif
#endif
