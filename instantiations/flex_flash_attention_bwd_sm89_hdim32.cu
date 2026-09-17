// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Explicit instantiations for flex flash attention backward, hdim=32.
// Compiled by hgcc (GPU compiler), not by the host C++ compiler.
//
// Narrow-headdim bucket ported from the fp32 set: 32 is the narrowest bwd
// width the generic (CVT-free) path can express — per-warp MMA N slice
// stays a multiple of 16 with the legacy atom layout (32/2), while 16
// fails the 2:1 accum-copy static_assert (probed in the fp32 set).  The
// API rounds bwd headdims <=32 here (fwd keeps its own <=16 bucket).

#include "csrc/flex_flash_attention_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM32
template void run_mha_flex_flash_bwd_<89, cutlass::bfloat16_t, 32>(FlexFlashAttentionBwdParams &params, hggcStream_t stream);
template void run_mha_flex_flash_bwd_<89, cutlass::half_t, 32>(FlexFlashAttentionBwdParams &params, hggcStream_t stream);
#endif
#endif
