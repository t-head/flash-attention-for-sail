#pragma once
#include "flash.h"

namespace flash {
struct QsaConfig {
    static constexpr int M = 0, N = 0, Warps = 0, Stages = 0, Group = 1;
    static constexpr int RowBytes = 0, KVStride = 0;
    static constexpr bool QRegs = false, TsmQ = false, TsmKV = false;
    static constexpr bool FixedGroup = false, DirectIndex = false, SingleTile = false;
};
}

namespace qsa {
hggcError_t cached_occupancy(int device, int* result, const void* function,
                           int block, size_t smem, unsigned flags);
}

inline bool qsa_config_supported(Flash_fwd_params const& p) {
#if defined(USE_PPU) && USE_AIU && !defined(FLASHATTENTION_DISABLE_SM8x) && !defined(FLASHATTENTION_DISABLE_HDIM256) && !defined(FLASHATTENTION_DISABLE_PAGEDKV) && !defined(FLASHATTENTION_DISABLE_PACKGQA) && !defined(FA3_HLLM_BUILD)
    return p.is_qsa && !p.qsa_allow_aiu && p.arch == 89 && p.is_bf16 && p.d == 256 && p.dv == 256
        && p.h_k > 0 && p.page_size > 0
        && p.k_batch_stride == int64_t(p.page_size) * p.k_row_stride
        && p.v_batch_stride == int64_t(p.page_size) * p.v_row_stride
        && p.is_causal && !p.is_local && p.softcap == 0.f && p.pack_gqa
        && p.cu_seqlens_q && p.page_table && !p.leftpad_k && !p.knew_ptr && !p.vnew_ptr
        && !p.qv_ptr && !p.rotary_cos_ptr && !p.rotary_sin_ptr && !p.s_aux_ptr
        && p.v_dim_stride == 1
        // The direct GQA8 grid places batch on z and (split, KV head) on y.
        && (p.h != 8 * p.h_k || p.k_row_stride != 512 || p.v_row_stride != 512
            || (p.b <= 65535 && int64_t(p.h_k) * p.num_splits <= 65535));
#else
    return false;
#endif
}

inline bool qsa_uses_long_pipeline(Flash_fwd_params const& p) {
    return p.num_splits > 1 && p.seqlen_q <= 16
        && ((p.seqlen_k >= 8192 && p.seqlen_k / p.num_splits >= 256)
            || (p.seqlen_q == 1 && p.seqlen_k >= 2048 && p.seqlen_k < 8192
                && p.seqlen_k / p.num_splits >= 64
                && int64_t(p.num_pages) * p.page_size * p.k_row_stride * 2
                   >= int64_t(p.b) * 128 * 1024 * 1024));
}

inline bool qsa_uses_small_wide_tile(Flash_fwd_params const& p) {
    return p.seqlen_q == 1 && p.total_q == p.b && int64_t(p.b) * p.h_k <= 4
        && p.seqlen_k >= 2048 && p.seqlen_k < 8192;
}

inline bool qsa_gqa32_uses_single_tile(Flash_fwd_params const& p) {
    return p.h == 32 * p.h_k && p.seqlen_q <= 16
        && (p.seqlen_k <= 128 || (p.num_splits > 1 && p.seqlen_k <= 2048))
        && !(int64_t(p.b) * p.seqlen_q >= 256 && p.seqlen_k > 128)
        && p.b <= 65535 && int64_t(p.h_k) * p.num_splits <= 65535;
}

inline bool qsa_uses_single_tile(Flash_fwd_params const& p) {
    return qsa_config_supported(p)
        && ((p.h == 8 * p.h_k && p.k_row_stride == 512 && p.v_row_stride == 512
             && (p.seqlen_k < 2048 || qsa_uses_long_pipeline(p) || qsa_uses_small_wide_tile(p)))
            || (p.h > 16 * p.h_k && p.h < 32 * p.h_k && p.seqlen_q <= 16
                && p.b <= 65535 && int64_t(p.h_k) * p.num_splits <= 65535)
            || qsa_gqa32_uses_single_tile(p));
}

// Returns false without launching when this is not an eligible QSA request.
bool run_qsa(Flash_fwd_params&, hggcStream_t);
