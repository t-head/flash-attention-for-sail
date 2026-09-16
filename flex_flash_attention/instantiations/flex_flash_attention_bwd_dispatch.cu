// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Dispatch function for flex flash attention backward.
// Compiled by hgcc (GPU compiler), not by the host C++ compiler.
//
// The heavy template machinery (run_flex_flash_attention_bwd) is NOT compiled in
// this translation unit: we only declare the explicit instantiations that
// live in flex_flash_attention_bwd_sm89_hdim{64,96,128,192,256}.cu, keeping this
// file trivial to compile while the per-hdim files build in parallel.

#include <cstdio>
#include <cstdlib>

#include "../hopper/cuda_check.h"
#include "flex_flash_attention/csrc/flex_flash_attention_bwd_dispatch.h"

// Debug-build helper: when some hdim TUs are compiled out (slim debug
// wheels), provide weak fallback definitions so the link succeeds; calling
// a compiled-out hdim aborts at runtime.
#define ARB_DBG_WEAK_FALLBACK(T, HD)                                              \
    template <>                                                                   \
    __attribute__((weak)) void run_mha_flex_flash_bwd_<89, T, HD>(                 \
        FlexFlashAttentionBwdParams &params, hggcStream_t stream) {                   \
        fprintf(stderr, "flex flash attention bwd hdim%d not compiled in this build\n", HD); \
        abort();                                                                    \
    }
#if defined(FLASHATTENTION_DISABLE_HDIM64) || defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::bfloat16_t, 64)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 64)
#endif
// Narrow-headdim bucket (ported from the fp32 set): bwd hdim32 TU.  bwd
// has NO hdim16 TU (width 16 fails the accum-copy fragment geometry), so
// 16 keeps an unconditional weak fallback — unreachable: the API rounds
// bwd headdims to >=32.
#if defined(FLASHATTENTION_DISABLE_HDIM32) || defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::bfloat16_t, 32)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 32)
#endif
ARB_DBG_WEAK_FALLBACK(cutlass::bfloat16_t, 16)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 16)
#if defined(FLASHATTENTION_DISABLE_HDIM96) || defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::bfloat16_t, 96)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 96)
#endif
#if defined(FLASHATTENTION_DISABLE_HDIM128) || defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::bfloat16_t, 128)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 128)
#endif
#if defined(FLASHATTENTION_DISABLE_HDIM192) || defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::bfloat16_t, 192)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 192)
#endif
#if defined(FLASHATTENTION_DISABLE_HDIM256) || defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::bfloat16_t, 256)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 256)
#endif
// fp16 bwd TUs are compiled out when fp16 is disabled: weak fallbacks for
// every hdim keep the link alive; bf16 builds never reach them.  Skip hdims
// whose fallback was already emitted by the per-hdim guards above.
#if defined(FLASHATTENTION_DISABLE_FP16)
#if !defined(FLASHATTENTION_DISABLE_HDIM64) && !defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 64)
#endif
#if !defined(FLASHATTENTION_DISABLE_HDIM96) && !defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 96)
#endif
#if !defined(FLASHATTENTION_DISABLE_HDIM128) && !defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 128)
#endif
#if !defined(FLASHATTENTION_DISABLE_HDIM192) && !defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 192)
#endif
#if !defined(FLASHATTENTION_DISABLE_HDIM256) && !defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 256)
#endif
#endif

#define HEADDIM_SWITCH_ARB(HEADDIM, ...)  \
    [&] {                                 \
        if (HEADDIM <= 32) {              \
            constexpr static int kHeadDim = 32;  \
            return __VA_ARGS__();         \
        } else if (HEADDIM <= 64) {       \
            constexpr static int kHeadDim = 64;  \
            return __VA_ARGS__();         \
        } else if (HEADDIM <= 96) {       \
            constexpr static int kHeadDim = 96;  \
            return __VA_ARGS__();         \
        } else if (HEADDIM <= 128) {      \
            constexpr static int kHeadDim = 128; \
            return __VA_ARGS__();         \
        } else if (HEADDIM <= 192) {      \
            constexpr static int kHeadDim = 192; \
            return __VA_ARGS__();         \
        } else {                          \
            constexpr static int kHeadDim = 256; \
            return __VA_ARGS__();         \
        }                                 \
    }()

// Dispatch function called from flex_flash_attention_api.cpp (host C++ code)
void run_mha_flex_flash_bwd(FlexFlashAttentionBwdParams &params, hggcStream_t stream) {
    constexpr int Arch = 89;
    HEADDIM_SWITCH_ARB(params.d, [&] {
        if (params.is_bf16) {
            run_mha_flex_flash_bwd_<Arch, cutlass::bfloat16_t, kHeadDim>(params, stream);
        } else {
            run_mha_flex_flash_bwd_<Arch, cutlass::half_t, kHeadDim>(params, stream);
        }
    });
}
