/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Dispatch entry for flex flash attention forward, split per (dtype,
// hdim) so each heavy instantiation compiles in its own translation unit.
//
// run_mha_flex_flash_fwd_ is defined in flex_flash_attention_fwd_launch_template.h
// and explicitly instantiated in flex_flash_attention_fwd_sm89_hdimXX.cu.

#pragma once

#include <hggc_runtime.h>

#include "cutlass/numeric_types.h"   // cutlass::half_t / bfloat16_t
#include "../hopper/flash.h"
#include "attn_slice.h"

template <int Arch, typename T, int kHeadDim, int kHeadDimV, bool PackGQA>
void run_mha_flex_flash_fwd_(FlexFlashAttentionFwdParams &params, hggcStream_t stream);

// fp32 (TF32 compute) kernel set — isolated instantiations in
// flex_flash_attention_fwd_sm89_hdim*_f32.cu.
template <int Arch, int kHeadDim, int kHeadDimV, bool PackGQA>
void run_mha_flex_flash_fwd_f32_(FlexFlashAttentionFwdParams &params, hggcStream_t stream);

// Implemented in flex_flash_attention_fwd_dispatch.cu (called from the C++ API).
void run_mha_flex_flash_fwd(FlexFlashAttentionFwdParams &params, hggcStream_t stream);

// Implemented in flex_flash_attention_f32_dispatch.cu (fp32 API entry only).
void run_mha_flex_flash_fwd_f32(FlexFlashAttentionFwdParams &params, hggcStream_t stream);
