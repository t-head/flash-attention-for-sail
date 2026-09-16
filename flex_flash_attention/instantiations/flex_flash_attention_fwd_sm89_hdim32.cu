// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Explicit instantiations for flex flash attention forward, hdim=32.
// Compiled by hgcc (GPU compiler), not by the host C++ compiler.
//
// Narrow-headdim bucket (ported from the fp32 set, validated there):
// D in (16, 32] previously padded to the 64 bucket (up to 2x compute
// waste).  Shares the common fwd tile dispatch (no per-width config knob).

#include "flex_flash_attention/csrc/flex_flash_attention_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM32
template void run_mha_flex_flash_fwd_<89, cutlass::bfloat16_t, 32, 32, false>(FlexFlashAttentionFwdParams &params, hggcStream_t stream);
template void run_mha_flex_flash_fwd_<89, cutlass::half_t, 32, 32, false>(FlexFlashAttentionFwdParams &params, hggcStream_t stream);
#endif
#endif
