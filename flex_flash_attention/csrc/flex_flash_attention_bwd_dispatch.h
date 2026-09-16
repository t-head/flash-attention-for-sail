/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Dispatch entry for flex flash attention backward, split per (dtype,
// hdim) so each heavy instantiation compiles in its own translation unit.
//
// run_mha_flex_flash_bwd_ is defined in flex_flash_attention_bwd_launch_template.h
// and explicitly instantiated in flex_flash_attention_bwd_sm89_hdimXX.cu.

#pragma once

#include <hggc_runtime.h>

#include "cutlass/numeric_types.h"   // cutlass::half_t / bfloat16_t
#include "../hopper/flash.h"
#include "attn_slice.h"

template <int Arch, typename T, int kHeadDim>
void run_mha_flex_flash_bwd_(FlexFlashAttentionBwdParams &params, hggcStream_t stream);

// Implemented in flex_flash_attention_bwd_dispatch.cu (called from the C++ API).
void run_mha_flex_flash_bwd(FlexFlashAttentionBwdParams &params, hggcStream_t stream);

// Implemented in flex_flash_attention_f32_dispatch.cu (fp32 API entry only).
void run_mha_flex_flash_bwd_f32(FlexFlashAttentionBwdParams &params, hggcStream_t stream);
