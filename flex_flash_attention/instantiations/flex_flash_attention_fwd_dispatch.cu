// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Dispatch function for flex flash attention forward.
// Compiled by hgcc (GPU compiler), not by the host C++ compiler.
//
// The heavy template machinery (run_flex_flash_attention_fwd) is NOT compiled in
// this translation unit: we only declare the explicit instantiations that
// live in flex_flash_attention_fwd_sm89_hdim{64,96,128,192,256}.cu, keeping this
// file trivial to compile while the per-hdim files build in parallel.

#include <cstdio>
#include <cstdlib>

#include "../hopper/cuda_check.h"
#include "flex_flash_attention/csrc/flex_flash_attention_fwd_dispatch.h"

// Debug-build helper: when some hdim TUs are compiled out (slim debug
// wheels), provide weak fallback definitions so the link succeeds; calling
// a compiled-out hdim aborts at runtime.
#define ARB_DBG_WEAK_FALLBACK(T, HD)                                              \
    template <>                                                                   \
    __attribute__((weak)) void run_mha_flex_flash_fwd_<89, T, HD, HD, false>(      \
        FlexFlashAttentionFwdParams &params, hggcStream_t stream) {                   \
        fprintf(stderr, "flex flash attention fwd hdim%d not compiled in this build\n", HD); \
        abort();                                                                    \
    }
#if defined(FLASHATTENTION_DISABLE_HDIM64) || defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::bfloat16_t, 64)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 64)
#endif
// Narrow-headdim bucket (ported from the fp32 set): fwd hdim32 TU; weak
// fallbacks keep slim/debug builds linking.  No hdim16 TU: the 16-bit fwd
// kernel at kHeadDim=16 trips a PPU hardware exception (AIU_ld TSM size
// out of range), so the API rounds fwd headdims to >=32 — the 16 arm
// below keeps an unconditional weak fallback for link completeness.
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
// fp16 fwd TUs are compiled out when fp16 is disabled: weak fallbacks for
// every hdim keep the link alive; bf16 builds never reach them.  Skip hdims
// whose fallback was already emitted by the per-hdim guards above.
#if defined(FLASHATTENTION_DISABLE_FP16)
#if !defined(FLASHATTENTION_DISABLE_HDIM32) && !defined(ARB_DBG)
ARB_DBG_WEAK_FALLBACK(cutlass::half_t, 32)
#endif
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
void run_mha_flex_flash_fwd(FlexFlashAttentionFwdParams &params, hggcStream_t stream) {
    constexpr int Arch = 89;
    constexpr bool PackGQA = false;
    HEADDIM_SWITCH_ARB(params.d, [&] {
        constexpr int kHeadDimV = kHeadDim;
        if (params.is_bf16) {
            run_mha_flex_flash_fwd_<Arch, cutlass::bfloat16_t, kHeadDim, kHeadDimV, PackGQA>(params, stream);
        } else {
            run_mha_flex_flash_fwd_<Arch, cutlass::half_t, kHeadDim, kHeadDimV, PackGQA>(params, stream);
        }
    });
}
