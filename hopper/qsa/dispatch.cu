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
// Qwen3.8 TP8/TP4/TP2/TP1 use groups 3/6/12/12 respectively.
// Keep the 16-row MMA tile but avoid computing a runtime head divmod, and
// address each selected token directly when the physical KV row stride fits.
struct GqaSharedDirect256 : Gqa8Direct { static constexpr int KVStride = 256; static constexpr bool FixedGroup = false, DirectGroupIndex = true; };
// Large unsplit query batches avoid predicated async copies for unused Q rows.
struct GqaSharedLong256 : GqaSharedDirect256 { static constexpr int RowBytes = 256; static constexpr bool PredicatedQ = false; };
struct GqaSharedVarlen256 : GqaSharedDirect256 { static constexpr bool SingleTile = false; };
struct GqaSharedDirect512 : Gqa8Direct { static constexpr bool FixedGroup = false, DirectGroupIndex = true; };
struct GqaSharedVarlen512 : GqaSharedDirect512 { static constexpr bool SingleTile = false; };
struct GqaSharedWarpN : GqaSharedDirect256 {
    static constexpr int Warps = 4, Stages = 2;
    static constexpr bool QRegs = false;
};
// Single-query decode can share a noncausal instance across groups 3, 6, and 12.
struct GqaSharedWarpNShort : GqaSharedWarpN { static constexpr int Stages = 1; };
struct Gqa3Direct : Gqa8Direct { static constexpr int Group = 3, KVStride = 256; };
struct Gqa3WarpN : Gqa3Direct {
    static constexpr int Warps = 4;
    static constexpr int Stages = 2;
    static constexpr bool QRegs = false;
};
struct Gqa3WarpNShort : Gqa3WarpN { static constexpr int Stages = 1; };
struct Gqa3WarpNQueryFirst : Gqa3WarpNShort { static constexpr bool QueryFirst = true; };
struct Gqa12Direct256 : Gqa8Direct { static constexpr int Group = 12, KVStride = 256; };
struct Gqa12WarpN : Gqa12Direct256 {
    static constexpr int Warps = 4;
    static constexpr int Stages = 2;
    static constexpr bool QRegs = false;
};
struct Gqa12WarpNShort : Gqa12WarpN { static constexpr int Stages = 1; };
struct Gqa12WarpN512 : Gqa12WarpNShort { static constexpr int KVStride = 512; };
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

template<class Config, bool Split, bool Causal=true>
void launch(Flash_fwd_params& p, hggcStream_t stream) {
    run_flash_fwd<89, 256, 256, 1, cutlass::bfloat16_t, cutlass::bfloat16_t,
        Causal, false, false, true, true, false, 16, false, false,
        true, Split, false, false, false, false, true, Config>(p, stream);
}
template<class Config>
void launch(Flash_fwd_params& p, hggcStream_t stream) {
    if (p.num_splits > 1) { launch<Config, true>(p, stream); }
    else { launch<Config, false>(p, stream); }
}

// Each CTA merges a 64-element output slice of one query head. Its warps
// read different SplitKV partitions, exposing parallelism in short decode.
template<int FixedSplits, int Warps, bool PrefetchOutput=false>
__global__ void decode_combine_kernel(Flash_fwd_params p) {
    constexpr int DimTile = 64;
    constexpr int ValuesPerLane = DimTile / 32;
    static_assert(!PrefetchOutput || (FixedSplits == 128 && Warps == 16)
                  || (FixedSplits == 64 && Warps == 8));
    constexpr int PrefetchIters = PrefetchOutput ? (FixedSplits - 1) / (Warps - 1) : 8;
    __shared__ float weights[128];
    __shared__ float partial[Warps][DimTile];

    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const row = blockIdx.x;
    int const head = blockIdx.y;
    int const num_splits = FixedSplits > 0 ? FixedSplits :
        (p.num_splits_dynamic_ptr ? p.num_splits_dynamic_ptr[row] : p.num_splits);
    auto const* lse = static_cast<float const*>(p.softmax_lseaccum_ptr);
    auto const* oaccum = static_cast<float const*>(p.oaccum_ptr);

    // Warp 0 normalizes the split weights while the other warps fetch their
    // full rounds of O partials. Remaining splits are read after the CTA barrier.
    float prefetched[PrefetchIters][ValuesPerLane];
    if constexpr (PrefetchOutput) {
        if (warp != 0) {
            #pragma unroll
            for (int j = 0; j < PrefetchIters; ++j) {
                int const split = warp - 1 + j * (Warps - 1);
                auto const* src = oaccum + int64_t(split) * p.oaccum_split_stride
                    + int64_t(head) * p.oaccum_head_stride + int64_t(row) * p.oaccum_row_stride
                    + int64_t(blockIdx.z) * DimTile;
                #pragma unroll
                for (int i = 0; i < ValuesPerLane; ++i) {
                    prefetched[j][i] = src[lane * ValuesPerLane + i];
                }
            }
        }
    }

    // The first warp computes the normalized split weights once per head.
    if (warp == 0) {
        float lse_values[4];
        float lse_max = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int const split = lane + i * 32;
            lse_values[i] = split < num_splits
                ? lse[int64_t(split) * p.lseaccum_split_stride + int64_t(head) * p.lseaccum_head_stride + row]
                : -INFINITY;
            lse_max = fmaxf(lse_max, lse_values[i]);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            lse_max = fmaxf(lse_max, __shfl_down_sync(0xffffffff, lse_max, offset));
        }
        lse_max = __shfl_sync(0xffffffff, lse_max, 0);
        float weight_values[4];
        float weight_sum = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            weight_values[i] = lse_values[i] == -INFINITY ? 0.f : expf(lse_values[i] - lse_max);
            weight_sum += weight_values[i];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            weight_sum += __shfl_down_sync(0xffffffff, weight_sum, offset);
        }
        weight_sum = __shfl_sync(0xffffffff, weight_sum, 0);
        float const inv_sum = weight_sum > 0.f ? 1.f / weight_sum : 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int const split = lane + i * 32;
            if (split < num_splits) { weights[split] = weight_values[i] * inv_sum; }
        }
        if (lane == 0 && blockIdx.z == 0) {
            static_cast<float*>(p.softmax_lse_ptr)[int64_t(head) * p.total_q + row] =
                weight_sum > 0.f ? lse_max + logf(weight_sum) : -INFINITY;
        }
    }
    __syncthreads();

    float accum[ValuesPerLane] = {};
    if constexpr (PrefetchOutput) {
        if (warp != 0) {
            #pragma unroll
            for (int j = 0; j < PrefetchIters; ++j) {
                int const split = warp - 1 + j * (Warps - 1);
                float const weight = weights[split];
                if (weight > 0.f) {
                    #pragma unroll
                    for (int i = 0; i < ValuesPerLane; ++i) {
                        accum[i] = fmaf(weight, prefetched[j][i], accum[i]);
                    }
                }
            }
            #pragma unroll 2
            for (int j = PrefetchIters; j < PrefetchIters + 1; ++j) {
                int const split = warp - 1 + j * (Warps - 1);
                if (split < num_splits) {
                    float const weight = weights[split];
                    if (weight > 0.f) {
                        auto const* src = oaccum + int64_t(split) * p.oaccum_split_stride
                            + int64_t(head) * p.oaccum_head_stride + int64_t(row) * p.oaccum_row_stride
                            + int64_t(blockIdx.z) * DimTile;
                        #pragma unroll
                        for (int i = 0; i < ValuesPerLane; ++i) {
                            accum[i] = fmaf(weight, src[lane * ValuesPerLane + i], accum[i]);
                        }
                    }
                }
            }
        }
    } else {
        #pragma unroll 4
        for (int split = warp; split < num_splits; split += Warps) {
            float const weight = weights[split];
            if (weight > 0.f) {
                auto const* src = oaccum + int64_t(split) * p.oaccum_split_stride
                    + int64_t(head) * p.oaccum_head_stride + int64_t(row) * p.oaccum_row_stride
                    + int64_t(blockIdx.z) * DimTile;
                #pragma unroll
                for (int i = 0; i < ValuesPerLane; ++i) {
                    accum[i] = fmaf(weight, src[lane * ValuesPerLane + i], accum[i]);
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < ValuesPerLane; ++i) { partial[warp][lane * ValuesPerLane + i] = accum[i]; }
    __syncthreads();
    if (warp == 0) {
        auto* output = static_cast<cutlass::bfloat16_t*>(p.o_ptr)
            + int64_t(row) * p.o_row_stride + int64_t(head) * p.o_head_stride
            + int64_t(blockIdx.z) * DimTile;
        #pragma unroll
        for (int i = 0; i < ValuesPerLane; ++i) {
            int const d = lane * ValuesPerLane + i;
            float value = 0.f;
            #pragma unroll
            for (int w = 0; w < Warps; ++w) { value += partial[w][d]; }
            output[d] = cutlass::bfloat16_t(value);
        }
    }
}
}  // namespace qsa

bool run_qsa_decode_combine(Flash_fwd_params const& p, hggcStream_t stream) {
    bool const small_decode_2048 = qsa_small_decode_2048(p);
    bool const group12_small = p.h == 12 * p.h_k
        && ((p.b <= 6 && p.h_k == 1 && p.k_row_stride == 256)
            || (p.b <= 2 && p.h_k == 2 && p.k_row_stride == 512))
        && p.k_row_stride == p.v_row_stride
        && small_decode_2048
        && (p.num_splits == 32 || p.num_splits == 64);
    if (!qsa_config_supported(p) || p.d != 256 || p.dv != 256
        || p.seqlen_q != 1 || p.total_q != p.b || p.b > 16
        || p.num_splits <= 1 || p.num_splits > 128
        || (!group12_small && (p.h_k != 1 || (p.h != 3 && p.h != 6)))
        || p.o_head_stride != 256 || p.oaccum_row_stride != 256) { return false; }
    dim3 grid(p.total_q, p.h, 256 / 64);
    if (p.b == 1 && (p.h == 3 || p.h == 6) && small_decode_2048
        && p.num_splits_dynamic_ptr == nullptr && p.num_splits == 128) {
        qsa::decode_combine_kernel<128, 16, true><<<grid, 512, 0, stream>>>(p);
    } else if (p.num_splits_dynamic_ptr == nullptr && p.num_splits == 128) {
        qsa::decode_combine_kernel<128, 16><<<grid, 512, 0, stream>>>(p);
    } else if (p.num_splits == 128) {
        qsa::decode_combine_kernel<0, 16><<<grid, 512, 0, stream>>>(p);
    } else if (((group12_small && p.b == 1 && p.h_k == 1)
                || (p.h == 3 && p.h_k == 1 && p.b == 2
                    && p.k_row_stride == 256 && p.v_row_stride == 256
                    && small_decode_2048))
               && p.num_splits_dynamic_ptr == nullptr && p.num_splits == 64) {
        qsa::decode_combine_kernel<64, 8, true><<<grid, 256, 0, stream>>>(p);
    } else if (p.num_splits_dynamic_ptr == nullptr && p.num_splits == 64) {
        qsa::decode_combine_kernel<64, 8><<<grid, 256, 0, stream>>>(p);
    } else if (p.num_splits == 64) {
        qsa::decode_combine_kernel<0, 8><<<grid, 256, 0, stream>>>(p);
    } else {
        qsa::decode_combine_kernel<0, 4><<<grid, 128, 0, stream>>>(p);
    }
    CHECK_CUDA_KERNEL_LAUNCH();
    return true;
}

bool run_qsa(Flash_fwd_params &p, hggcStream_t stream) {
    using namespace qsa;
    if (!qsa_config_supported(p)) { return false; }
    const bool split = p.num_splits > 1;
    const bool uniform_q = p.total_q == int64_t(p.b) * p.seqlen_q;
    const bool direct_grid = uniform_q && p.b <= 65535
        && int64_t(p.h_k) * p.num_splits <= 65535;
    const bool small_decode_2048 = qsa_small_decode_2048(p);
    if (p.h == 3 * p.h_k && p.k_row_stride == 256 && p.v_row_stride == 256) {
        if (direct_grid && split && p.b <= 7 && small_decode_2048) {
            // Nearby query CTAs improve shared-KV reuse in short MTP groups.
            if (p.b >= 3 && p.b <= 6) { launch<Gqa3WarpNQueryFirst, true>(p, stream); }
            else if (p.b == 2) { launch<GqaSharedWarpNShort, true, false>(p, stream); }
            else { launch<GqaSharedWarpN, true>(p, stream); }
        } else if (direct_grid && !split && p.total_q > 2048) { launch<GqaSharedLong256, false>(p, stream); }
        else if (direct_grid) { launch<GqaSharedDirect256>(p, stream); }
        else { launch<GqaSharedVarlen256>(p, stream); }
        return true;
    }
    if (p.h == 6 * p.h_k && p.k_row_stride == 256 && p.v_row_stride == 256) {
        // A single right-aligned query sees every selected KV position; the
        // length and topk-width masks still handle the tail.
        if (direct_grid && split && p.b <= 2 && small_decode_2048) {
            launch<GqaSharedWarpNShort, true, false>(p, stream);
        } else if (direct_grid && !split && p.total_q > 2048) { launch<GqaSharedLong256, false>(p, stream); }
        else if (direct_grid) { launch<GqaSharedDirect256>(p, stream); }
        else { launch<GqaSharedVarlen256>(p, stream); }
        return true;
    }
    if (p.h == 12 * p.h_k && p.k_row_stride == p.v_row_stride) {
        if (p.k_row_stride == 256) {
            if (direct_grid && split && p.b <= 4 && small_decode_2048) {
                if (p.b >= 2 && p.b <= 3) { launch<GqaSharedWarpNShort, true, false>(p, stream); }
                else { launch<GqaSharedWarpN, true>(p, stream); }
            } else if (direct_grid && !split && p.total_q <= 2048) {
                // Small GQA12 prefill benefits from its fixed-group Q addressing.
                launch<Gqa12Direct256, false>(p, stream);
            } else if (direct_grid && !split && p.total_q > 2048) { launch<GqaSharedLong256, false>(p, stream); }
            else if (direct_grid) { launch<GqaSharedDirect256>(p, stream); }
            else { launch<GqaSharedVarlen256>(p, stream); }
            return true;
        }
        if (p.k_row_stride == 512) {
            if (direct_grid && split && p.b <= 2 && p.h_k == 2
                && small_decode_2048) {
                launch<Gqa12WarpN512, true>(p, stream);
            } else if (direct_grid) { launch<GqaSharedDirect512>(p, stream); }
            else { launch<GqaSharedVarlen512>(p, stream); }
            return true;
        }
    }
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
            launch<GqaSharedVarlen512>(p, stream);
        } else {
            launch<GqaSharedDirect512>(p, stream);
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
bool run_qsa_decode_combine(Flash_fwd_params const&, hggcStream_t) { return false; }
#endif
