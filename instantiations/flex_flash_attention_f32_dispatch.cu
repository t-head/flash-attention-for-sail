// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Dispatch functions for the ISOLATED fp32 (TF32 compute) flex flash attention
// kernel set.  Compiled by hgcc (GPU compiler), not by the host C++ compiler.
//
// Kept in its own translation unit so the bf16/fp16 dispatch TUs stay
// untouched: only run_mha_flex_flash_fwd_f32 / run_mha_flex_flash_bwd_f32
// live here, reached exclusively from the fwd_f32/bwd_f32 API entries.

#include <cstdio>
#include <cstdlib>

#include "../hopper/cuda_check.h"
#include "csrc/flex_flash_attention_fwd_dispatch.h"
#include "csrc/flex_flash_attention_bwd_dispatch.h"

// Debug-build helper (same convention as the bf16/fp16 dispatch TUs): when
// some hdim TUs are compiled out (slim debug wheels), provide weak fallback
// definitions so the link succeeds; calling a compiled-out hdim aborts.
#define ARB_F32_WEAK_FALLBACK_FWD(HD)                                                \
    template <>                                                                      \
    __attribute__((weak)) void run_mha_flex_flash_fwd_f32_<89, HD, HD, false>(        \
        FlexFlashAttentionFwdParams &params, hggcStream_t stream) {                      \
        fprintf(stderr, "flex flash attention fwd f32 hdim%d not compiled in this build\n", HD); \
        abort();                                                                     \
    }
#define ARB_F32_WEAK_FALLBACK_BWD(HD)                                                \
    template <>                                                                      \
    __attribute__((weak)) void run_mha_flex_flash_bwd_<89, float, HD>(     \
        FlexFlashAttentionBwdParams &params, hggcStream_t stream) {                      \
        fprintf(stderr, "flex flash attention bwd f32 hdim%d not compiled in this build\n", HD); \
        abort();                                                                     \
    }

#if defined(FLASHATTENTION_DISABLE_HDIM64) || defined(ARB_DBG)
ARB_F32_WEAK_FALLBACK_FWD(64)
ARB_F32_WEAK_FALLBACK_BWD(64)
#endif
// EXPERIMENTAL narrow-headdim bucket: fwd hdim16 + bwd hdim32 TUs.  bwd has
// NO hdim16 TU (width 16 cannot satisfy the epilogue fragment geometry), so
// 16 keeps an unconditional weak fallback — unreachable: the API rounds bwd
// to >=32.
ARB_F32_WEAK_FALLBACK_BWD(16)
#if defined(FLASHATTENTION_DISABLE_HDIM16)
ARB_F32_WEAK_FALLBACK_FWD(16)
#endif
// EXPERIMENTAL fwd hdim32 bucket (D in (16, 32]).  No 16-bit TU exists for
// fp32-32, so the weak fallback guards slim builds that compile it out.
#if defined(FLASHATTENTION_DISABLE_HDIM32)
ARB_F32_WEAK_FALLBACK_FWD(32)
#endif
#if defined(FLASHATTENTION_DISABLE_HDIM96) || defined(ARB_DBG)
ARB_F32_WEAK_FALLBACK_FWD(96)
ARB_F32_WEAK_FALLBACK_BWD(96)
#endif
#if defined(FLASHATTENTION_DISABLE_HDIM128) || defined(ARB_DBG)
ARB_F32_WEAK_FALLBACK_FWD(128)
ARB_F32_WEAK_FALLBACK_BWD(128)
#endif
#if defined(FLASHATTENTION_DISABLE_HDIM192) || defined(ARB_DBG)
ARB_F32_WEAK_FALLBACK_FWD(192)
ARB_F32_WEAK_FALLBACK_BWD(192)
#endif
#if defined(FLASHATTENTION_DISABLE_HDIM256) || defined(ARB_DBG)
ARB_F32_WEAK_FALLBACK_FWD(256)
ARB_F32_WEAK_FALLBACK_BWD(256)
#endif

#define HEADDIM_SWITCH_ARB_F32(HEADDIM, ...)  \
    [&] {                                     \
        if (HEADDIM <= 16) {                  \
            constexpr static int kHeadDim = 16;  \
            return __VA_ARGS__();             \
        } else if (HEADDIM <= 32) {           \
            constexpr static int kHeadDim = 32;  \
            return __VA_ARGS__();             \
        } else if (HEADDIM <= 64) {           \
            constexpr static int kHeadDim = 64;  \
            return __VA_ARGS__();             \
        } else if (HEADDIM <= 96) {           \
            constexpr static int kHeadDim = 96;  \
            return __VA_ARGS__();             \
        } else if (HEADDIM <= 128) {          \
            constexpr static int kHeadDim = 128; \
            return __VA_ARGS__();             \
        } else if (HEADDIM <= 192) {          \
            constexpr static int kHeadDim = 192; \
            return __VA_ARGS__();             \
        } else {                              \
            constexpr static int kHeadDim = 256; \
            return __VA_ARGS__();             \
        }                                     \
    }()

// bwd dispatch switch: same buckets as fwd plus the EXPERIMENTAL hdim32
// narrow bucket (hdim32 bwd TU).  Kept separate from the fwd switch so the
// fwd path never references a fwd-32 symbol that does not exist.
#define HEADDIM_SWITCH_ARB_BWD_F32(HEADDIM, ...)  \
    [&] {                                     \
        if (HEADDIM <= 32) {                  \
            constexpr static int kHeadDim = 32;  \
            return __VA_ARGS__();             \
        } else if (HEADDIM <= 64) {           \
            constexpr static int kHeadDim = 64;  \
            return __VA_ARGS__();             \
        } else if (HEADDIM <= 96) {           \
            constexpr static int kHeadDim = 96;  \
            return __VA_ARGS__();             \
        } else if (HEADDIM <= 128) {          \
            constexpr static int kHeadDim = 128; \
            return __VA_ARGS__();             \
        } else if (HEADDIM <= 192) {          \
            constexpr static int kHeadDim = 192; \
            return __VA_ARGS__();             \
        } else {                              \
            constexpr static int kHeadDim = 256; \
            return __VA_ARGS__();             \
        }                                     \
    }()

// fwd dispatch — called from mha_flex_flash_fwd_f32 (host C++ code).
void run_mha_flex_flash_fwd_f32(FlexFlashAttentionFwdParams &params, hggcStream_t stream) {
    constexpr int Arch = 89;
    constexpr bool PackGQA = false;
    HEADDIM_SWITCH_ARB_F32(params.d, [&] {
        constexpr int kHeadDimV = kHeadDim;
        run_mha_flex_flash_fwd_f32_<Arch, kHeadDim, kHeadDimV, PackGQA>(params, stream);
    });
}

// bwd dispatch — called from mha_flex_flash_bwd_f32 (host C++ code).
void run_mha_flex_flash_bwd_f32(FlexFlashAttentionBwdParams &params, hggcStream_t stream) {
    constexpr int Arch = 89;
    HEADDIM_SWITCH_ARB_BWD_F32(params.d, [&] {
        run_mha_flex_flash_bwd_<Arch, float, kHeadDim>(params, stream);
    });
}
