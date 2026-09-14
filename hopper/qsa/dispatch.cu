// Copyright (c) 2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
#include "qsa/dispatch.h"
#include "flash_fwd_launch_template.h"

#if defined(USE_PPU) && USE_AIU && !defined(FLASHATTENTION_DISABLE_SM8x) && !defined(FLASHATTENTION_DISABLE_HDIM256) && !defined(FLASHATTENTION_DISABLE_PAGEDKV) && !defined(FLASHATTENTION_DISABLE_PACKGQA) && !defined(FA3_HLLM_BUILD)

namespace qsa {
// The defaults describe GQA8. Each named configuration changes only its tuning
// choices; eligibility and automatic SplitKV selection remain in dispatch.h/API.
struct Gqa8 : flash::QsaConfig {
    static constexpr int Group = 8, M = 16, N = 16, Warps = 1, Stages = 1;
    static constexpr int RowBytes = 128, KVStride = 512;
    static constexpr bool QRegs = true, TsmQ = true, TsmKV = false, FixedGroup = true;
    static constexpr bool DirectIndex = false, SingleTile = false;
};
struct Gqa8Direct : Gqa8 {
    static constexpr bool DirectIndex = true, SingleTile = true;
};
struct Gqa8Varlen : Gqa8Direct { static constexpr bool SingleTile = false; };
struct Gqa8Wide : Gqa8Varlen { static constexpr int RowBytes = 512; };
struct Gqa8ShortWide : Gqa8Wide { static constexpr bool SingleTile = true; };
struct Gqa8Long : Gqa8Direct { static constexpr int Stages = 2, RowBytes = 256; };
struct Gqa16 : Gqa8 {
    static constexpr int Group = 16, RowBytes = 256, KVStride = 256;
    static constexpr bool TsmKV = true, DirectIndex = true;
};
struct Gqa32N16 : Gqa8 {
    static constexpr int Group = 32, M = 32, Warps = 2, RowBytes = 256;
    static constexpr bool TsmKV = true;
};
struct Gqa32N32 : Gqa32N16 { static constexpr int N = 32; };
struct Gqa32Single : Gqa32N16 { static constexpr bool SingleTile = true; };
struct Gqa32Large : Gqa32N32 { static constexpr int RowBytes = 128; };
struct Gqa32BatchedQregs : Gqa32Large { static constexpr bool FixedGroup = false; };
struct Gqa32Batched : Gqa32BatchedQregs { static constexpr bool QRegs = false; };
struct Partial : Gqa32N16 {
    static constexpr int RowBytes = 128;
    static constexpr bool FixedGroup = false;
};
struct PartialSingle : Partial { static constexpr bool SingleTile = true; };

// Cache occupancy only for direct, stride-512 QSA loads. The device is refreshed
// on every launch; this cache never interposes the runtime used by other kernels.
struct OccupancyEntry {
    int device = -1;
    const void* function = nullptr;
    int block = 0;
    size_t smem = 0;
    unsigned flags = 0;
    int blocks = 0;
};
hggcError_t cached_occupancy(int device, int* result, const void* function, int block,
                           size_t smem, unsigned flags) {
    static thread_local OccupancyEntry entries[8];
    static thread_local unsigned replacement = 0;
    for (const auto& e : entries) {
        if (e.device == device && e.function == function && e.block == block
            && e.smem == smem && e.flags == flags && e.blocks > 0) {
            *result = e.blocks;
            return hggcSuccess;
        }
    }
    const auto status = hggcOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        result, function, block, smem, flags);
    if (status == hggcSuccess && *result > 0) {
        entries[replacement++ % 8] = {device, function, block, smem, flags, *result};
    }
    return status;
}

template<class Config, bool Split>
void launch(Flash_fwd_params& p, hggcStream_t stream) {
    run_flash_fwd<89, 256, 256, 1, cutlass::bfloat16_t, cutlass::bfloat16_t,
        true, false, false, true, true, false, 16, false, false,
        true, Split, false, false, false, false, true, Config>(p, stream);
}
template<class Config>
void launch(Flash_fwd_params& p, hggcStream_t stream) {
    if (p.num_splits > 1) { launch<Config, true>(p, stream); }
    else { launch<Config, false>(p, stream); }
}
}  // namespace qsa

bool run_qsa(Flash_fwd_params &p, hggcStream_t stream) {
    using namespace qsa;
    if (!qsa_config_supported(p)) { return false; }
    const bool split = p.num_splits > 1;
    const bool uniform_q = p.total_q == int64_t(p.b) * p.seqlen_q;
    if (p.h == 8 * p.h_k) {
        if (p.k_row_stride != 512 || p.v_row_stride != 512) {
            launch<Gqa8>(p, stream);
        } else if (qsa_uses_long_pipeline(p)) {
            launch<Gqa8Long, true>(p, stream);
        } else if (qsa_uses_small_wide_tile(p)) {
            launch<Gqa8ShortWide>(p, stream);
        } else if (uniform_q && p.seqlen_k >= 2048) {
            launch<Gqa8Wide>(p, stream);
        } else if (p.seqlen_k >= 2048) {
            launch<Gqa8Varlen>(p, stream);
        } else {
            launch<Gqa8Direct>(p, stream);
        }
        return true;
    }
    if (p.h == 16 * p.h_k && p.k_row_stride == 256 && p.v_row_stride == 256) {
        launch<Gqa16>(p, stream);
        return true;
    }
    if (p.h > 16 * p.h_k && p.h < 32 * p.h_k) {
        if (qsa_uses_single_tile(p)) { launch<PartialSingle>(p, stream); }
        else { launch<Partial>(p, stream); }
        return true;
    }
    if (p.h != 32 * p.h_k) { return false; }
    if (qsa_gqa32_uses_single_tile(p)) {
        launch<Gqa32Single>(p, stream);
    } else if (p.seqlen_q <= 128 && int64_t(p.b) * p.seqlen_q >= 256 && p.seqlen_k > 128) {
        if (split || p.use_kblockm_16 || p.use_kblockm_128) { return false; }
        if (uniform_q) { launch<Gqa32BatchedQregs, false>(p, stream); }
        else { launch<Gqa32Batched, false>(p, stream); }
    } else if (p.seqlen_q > 128 && p.seqlen_k > 128 && p.k_row_stride == 256 && p.v_row_stride == 256) {
        launch<Gqa32Large>(p, stream);
    } else if (split ? (p.seqlen_q <= 128 && p.seqlen_k <= 2048) : p.seqlen_k <= 128) {
        launch<Gqa32N16>(p, stream);
    } else {
        launch<Gqa32N32>(p, stream);
    }
    return true;
}

#else
bool run_qsa(Flash_fwd_params&, hggcStream_t) { return false; }
#endif
