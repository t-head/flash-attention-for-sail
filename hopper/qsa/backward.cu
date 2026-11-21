// Copyright (c) 2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
#include <hggc_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cutlass/numeric_types.h>
#include <c10/cuda/CUDAException.h>
#include "dispatch.h"
#include "utils.h"

#if defined(USE_PPU) && defined(FLASHATTENTION_ENABLE_QSA) \
    && !defined(FLASHATTENTION_DISABLE_BACKWARD) \
    && !defined(FLASHATTENTION_DISABLE_HDIM256) \
    && !defined(FLASHATTENTION_DISABLE_SM8x) && !defined(FA3_HLLM_BUILD)

namespace {
using namespace cute;
constexpr int kHeadDim = 256;
constexpr int kTileCols = 64;
// Pack the physical K/V row in the low bits and a query-membership mask above it.
constexpr int kPairTokenBits = 13;
constexpr int kPairPool = 1 << kPairTokenBits;
constexpr int kPairWords = kPairPool / 32;
constexpr int kThreads = 128;
constexpr int64_t kMaxBlocks = 65535;

int launch_blocks(int64_t work, int threads_per_item = kThreads) {
    int64_t const blocks = (work + threads_per_item - 1) / threads_per_item;
    return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void validate_inputs(Flash_bwd_params p, int* error,
                                int64_t table_count, int64_t work_count, bool check_order) {
    int flags = 0;
    int64_t const pool_tokens = int64_t(p.num_pages) * p.page_size;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < work_count; i += int64_t(gridDim.x) * blockDim.x) {
        if (i < table_count) {
            int64_t const row = i / p.seqlen_k;
            int64_t const col = i % p.seqlen_k;
            int64_t const token = p.page_table[row * p.page_table_batch_stride + col];
            if (token < 0 || token >= pool_tokens) { flags |= 1; }
            // For singleton-query batches each causal cache length bounds one
            // row. Only active columns participate in a grouped gradient.
            if (check_order && col > 0 && (!p.seqused_k || col < p.seqused_k[row])
                && p.page_table[row * p.page_table_batch_stride + col - 1] >= token) {
                flags |= 16;
            }
        }
        if (i <= p.b) {
            int64_t const start = p.cu_seqlens_q[i];
            if (start < 0 || start > p.total_q
                || (i == 0 && start != 0) || (i == p.b && start != p.total_q)) {
                flags |= 2;
            }
            if (i < p.b) {
                int64_t const end = p.cu_seqlens_q[i + 1];
                int64_t const length = end - start;
                if (end < 0 || end > p.total_q || length < 0 || length > p.seqlen_q) {
                    flags |= 2;
                }
                if (p.seqused_k && p.seqused_k[i] < 0) { flags |= 4; }
            }
        }
    }
    if (flags) { atomicOr(error, flags); }
}

template <typename T>
__device__ __forceinline__ uint4 load_chunk(T const* source) {
    static_assert(sizeof(T) == 2);
    if ((reinterpret_cast<uintptr_t>(source) & 15) == 0) {
        return *reinterpret_cast<uint4 const*>(source);
    }
    auto const* bits = reinterpret_cast<uint16_t const*>(source);
    return make_uint4(uint32_t(bits[0]) | (uint32_t(bits[1]) << 16),
                      uint32_t(bits[2]) | (uint32_t(bits[3]) << 16),
                      uint32_t(bits[4]) | (uint32_t(bits[5]) << 16),
                      uint32_t(bits[6]) | (uint32_t(bits[7]) << 16));
}

bool query_group_enabled() {
    char const* setting = std::getenv("FA3_QSA_BWD_2Q");
    return setting == nullptr || std::strcmp(setting, "1") == 0;
}

bool supports_query_group(Flash_bwd_params const& p) {
    // Shorter singleton batches also amortize grouping and bitmap construction.
    bool const enough_work = (p.total_q >= 2048 && p.seqlen_k > 15 * kTileCols)
        || (p.b == p.total_q && p.seqlen_q == 1 && p.total_q >= 512 && p.seqlen_k >= 128);
    return p.is_bf16 && p.arch >= 89 && p.h_k > 0 && p.h == 12LL * p.h_k
        && enough_work
        && 2LL * p.seqlen_k <= kPairPool && int64_t(p.num_pages) * p.page_size <= kPairPool
        && int64_t(p.num_pages) * p.page_size <= INT_MAX / (int64_t(p.h_k) * kHeadDim)
        && p.k_head_stride == kHeadDim && p.v_head_stride == kHeadDim
        && p.k_row_stride == int64_t(p.h_k) * kHeadDim && p.v_row_stride == p.k_row_stride
        && p.k_batch_stride == int64_t(p.page_size) * p.k_row_stride && p.v_batch_stride == p.k_batch_stride
        && (reinterpret_cast<uintptr_t>(p.k_ptr) | reinterpret_cast<uintptr_t>(p.v_ptr)) % 16 == 0;
}

bool deterministic_group_candidate(Flash_bwd_params const& p) {
    char const* group = std::getenv("FA3_QSA_BWD_QUERY_GROUP");
    return p.deterministic && p.seqlen_q == 1 && p.b == p.total_q && supports_query_group(p) && query_group_enabled()
        && (group == nullptr || std::strcmp(group, "10") == 0);
}

// An MMA over the union may multiply an unselected NaN by zero. Reuse the
// existing validation readback to select single-query execution in that case.
__device__ bool nonfinite_bf16_chunk(void const* ptr, int64_t offset) {
    auto const v = load_chunk(static_cast<uint16_t const*>(ptr) + offset);
    auto has_nonfinite = [](uint32_t x) {
        return (x & 0x7f80u) == 0x7f80u || (x & 0x7f800000u) == 0x7f800000u;
    };
    return has_nonfinite(v.x) || has_nonfinite(v.y) || has_nonfinite(v.z) || has_nonfinite(v.w);
}

__global__ void check_pair_operands(Flash_bwd_params p, int* flags, int64_t q_chunks, int64_t kv_chunks) {
    bool nonfinite = false;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < q_chunks + kv_chunks; i += int64_t(gridDim.x) * blockDim.x) {
        if (i < q_chunks) {
            int64_t const head_row = i / (kHeadDim / 8);
            int const head = head_row % p.h, dim = (i % (kHeadDim / 8)) * 8;
            int64_t const q = head_row / p.h;
            nonfinite |= nonfinite_bf16_chunk(p.q_ptr, q * p.q_row_stride + head * p.q_head_stride + dim)
                || nonfinite_bf16_chunk(p.do_ptr, q * p.do_row_stride + head * p.do_head_stride + dim)
                || nonfinite_bf16_chunk(p.o_ptr, q * p.o_row_stride + head * p.o_head_stride + dim);
            if (!dim) {
                float const lse = static_cast<float const*>(p.softmax_lse_ptr)[int64_t(head) * p.total_q + q];
                nonfinite |= !isfinite(lse) && lse != -CUDART_INF_F;
            }
        } else {
            int64_t const offset = (i - q_chunks) * 8;
            nonfinite |= nonfinite_bf16_chunk(p.k_ptr, offset) || nonfinite_bf16_chunk(p.v_ptr, offset);
        }
    }
    if (__syncthreads_or(nonfinite) && !threadIdx.x) { atomicOr(flags, 8); }
}

template <typename T, int Arch, int M, int N = kTileCols, int Queries = 1>
struct BackwardMma {
    // The two-query tile has one 16-row warp group per query.
    static constexpr int kWarpRows = Queries > 2 ? 4 : Queries;
    static constexpr int kWarpCols = N / 16;
    static constexpr int kWarps = kWarpRows * kWarpCols;
    static constexpr int kMmaThreads = 32 * kWarps;
    using Atom = MMA_Atom<std::conditional_t<Arch >= 89,
        std::conditional_t<std::is_same_v<T, cutlass::half_t>,
            PPU0015_16x16x16_F32F16F16F32_TN, PPU0015_16x16x16_F32BF16BF16F32_TN>,
        std::conditional_t<std::is_same_v<T, cutlass::half_t>,
            PPU0010_16x16x16_F32F16F16F32_TN, PPU0010_16x16x16_F32BF16BF16F32_TN>>>;
    using MmaSdP = TiledMMA<Atom, Layout<Shape<Int<kWarpRows>, Int<kWarpCols>, _1>>,
                            Tile<Int<16 * kWarpRows>, Int<16 * kWarpCols>, _16>>;
    using MmaDQ = MmaSdP;
    using MmaDKV = TiledMMA<Atom, Layout<Shape<Int<kWarps>, _1, _1>>,
                            Tile<Int<N>, _16, _16>>;
    using Normal = Copy_Atom<PPU_U32x4_LDSM_N, T>;
    using Transpose = Copy_Atom<PPU_U16x8_LDSM_T, T>;
    using AtomQKV = decltype(composition(Swizzle<3, 3, 3>{},
        Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
    using LayoutQ = decltype(tile_to_shape(AtomQKV{}, Shape<Int<M>, _256>{}));
    using LayoutKV = decltype(tile_to_shape(AtomQKV{}, Shape<Int<N>, _256>{}));
    static constexpr int kPSAtomCols = N < 64 ? N : 64;
    using AtomPS = decltype(composition(Swizzle<kPSAtomCols == 64 ? 3 : 2, 3, 3>{},
        Layout<Shape<_8, Int<kPSAtomCols>>, Stride<Int<kPSAtomCols>, _1>>{}));
    using LayoutPS = decltype(tile_to_shape(AtomPS{}, Shape<Int<M>, Int<N>>{}));
    using LayoutQt = decltype(composition(LayoutQ{},
        Layout<Shape<_256, Int<M>>, Stride<Int<M>, _1>>{}));
    using LayoutKt = decltype(composition(LayoutKV{},
        Layout<Shape<_256, Int<N>>, Stride<Int<N>, _1>>{}));
    using LayoutPSt = decltype(composition(LayoutPS{},
        Layout<Shape<Int<N>, Int<M>>, Stride<Int<M>, _1>>{}));
    static constexpr int kVPanel = Queries > 2 ? 128 : (Queries == 2 ? 256 : 64);
    template <int VWidth>
    using LayoutV = decltype(tile_to_shape(AtomQKV{}, Shape<Int<N>, Int<VWidth>>{}));
    template <bool PrefetchKV>
    struct SharedStorage {
        alignas(128) T q[cosize_v<LayoutQ>];
        alignas(128) T dout[cosize_v<LayoutQ>];
        alignas(128) T k[cosize_v<LayoutKV>];
        alignas(128) T v_ps_ds[cosize_v<LayoutV<PrefetchKV ? kVPanel : kHeadDim>> + 2 * cosize_v<LayoutPS>];
        float lse[M], delta[M];
        // Packed fast path stores K/V element offsets; other layouts store token IDs.
        int kv_row[N], valid_count;
        int next_valid_count, next_job;
    };
    struct GroupStorage : SharedStorage<true> {
        using Entry = std::conditional_t<Queries == 2, uint16_t, uint32_t>;
        static_assert(kPairTokenBits + Queries <= sizeof(Entry) * 8);
        struct alignas(16) SoftmaxRow {
            float lse, delta;
            uint32_t eligible, differentiable;
        };
        // Once compacted, the query bitmaps can hold invariant softmax metadata.
        union {
            uint32_t selected[Queries][kPairWords];
            SoftmaxRow softmax_rows[M];
        };
        Entry entries[kPairPool];
        int counts[Queries], warp_prefix[kPairWords / 32], duplicates;
        int members[N], previous_row[N];
    };
};

template <typename CopyA, typename CopyB, bool PairSteps = false, typename Mma, typename Acc,
          typename TensorA, typename TensorB, typename Hook = std::nullptr_t>
__device__ __forceinline__ void shared_gemm(Mma mma, Acc& acc,
                                           TensorA const& a, TensorB const& b, Hook hook = nullptr) {
    auto thr_mma = mma.get_thread_slice(threadIdx.x);
    auto ra = thr_mma.partition_fragment_A(a);
    auto rb = thr_mma.partition_fragment_B(b);
    auto copy_a = make_tiled_copy_A(CopyA{}, mma);
    auto copy_b = make_tiled_copy_B(CopyB{}, mma);
    auto thr_a = copy_a.get_thread_slice(threadIdx.x);
    auto thr_b = copy_b.get_thread_slice(threadIdx.x);
    auto sa = thr_a.partition_S(a);
    auto sb = thr_b.partition_S(b);
    if constexpr (PairSteps) {
        // Load both K steps from shared memory before consuming either one.
        // Each hook still runs immediately before its corresponding MMA.
        auto va = thr_a.retile_D(ra);
        auto vb = thr_b.retile_D(rb);
        constexpr int steps = decltype(size<2>(ra))::value;
        static_assert(steps % 2 == 0);
#pragma hggc mmatiestrictly
        {
#pragma unroll
            for (int i = 0; i < steps; i += 2) {
#pragma unroll
                for (int j = i; j < i + 2; ++j) {
                    cute::copy(copy_a, sa(_, _, j), va(_, _, j));
                    cute::copy(copy_b, sb(_, _, j), vb(_, _, j));
                }
#pragma unroll
                for (int j = i; j < i + 2; ++j) {
                    if constexpr (!std::is_same_v<Hook, std::nullptr_t>) { hook(j, steps); }
                    cute::gemm(mma, ra(_, _, j), rb(_, _, j), acc);
                }
            }
        }
    } else {
        flash::gemm_sm80(acc, ra, rb, sa, sb, mma, copy_a, copy_b, thr_a, thr_b, hook);
    }
}

// Compact the query group into (token, membership) entries. Duplicate rows
// keep each occurrence to preserve its separate softmax contribution.
template <int Queries, class Storage>
__device__ void build_query_group(Storage& s, Flash_bwd_params const& p, int64_t query) {
    int const tid = threadIdx.x;
    for (int w = tid; w < kPairWords; w += blockDim.x) {
        for (int q = 0; q < Queries; ++q) { s.selected[q][w] = 0; }
    }
    if (tid == 0) { s.duplicates = 0; }
    __syncthreads();
    if constexpr (Queries == 10) {
        // Read five independent page-table rows before consuming any token ID.
        // Lanes within a warp still read adjacent entries in each row.
        constexpr int Batch = 5;
#pragma unroll
        for (int first = 0; first < Queries; first += Batch) {
            int counts[Batch];
            int max_count = 0;
#pragma unroll
            for (int q = 0; q < Batch; ++q) {
                counts[q] = s.counts[first + q];
                max_count = (max_count > counts[q] ? max_count : counts[q]);
            }
            for (int i = tid; i < max_count; i += blockDim.x) {
                int tokens[Batch];
#pragma unroll
                for (int q = 0; q < Batch; ++q) {
                    tokens[q] = i < counts[q]
                        ? p.page_table[(query + first + q) * p.page_table_batch_stride + i] : -1;
                }
#pragma unroll
                for (int q = 0; q < Batch; ++q) {
                    int const token = tokens[q];
                    if (token >= 0) {
                        unsigned const bit = 1u << (token % 32);
                        if (atomicOr(&s.selected[first + q][token / 32], bit) & bit) {
                            atomicExch(&s.duplicates, 1);
                        }
                    }
                }
            }
        }
    } else {
        for (int q = 0; q < Queries; ++q) {
            for (int i = tid; i < s.counts[q]; i += blockDim.x) {
                int const token = p.page_table[(query + q) * p.page_table_batch_stride + i];
                unsigned const bit = 1u << (token % 32);
                if (atomicOr(&s.selected[q][token / 32], bit) & bit) { atomicExch(&s.duplicates, 1); }
            }
        }
    }
    __syncthreads();
    if (s.duplicates) {
        if constexpr (Queries == 2) {
            // Two prefixes fit side by side in entries under the dispatch bound.
            for (int q = 0; q < Queries; ++q) {
                int const start = q ? s.counts[0] : 0;
                for (int i = tid; i < s.counts[q]; i += blockDim.x) {
                    int const token = p.page_table[(query + q) * p.page_table_batch_stride + i];
                    s.entries[start + i] = token | (1 << (kPairTokenBits + q));
                }
            }
        }
        if (tid == 0) {
            int count = 0;
            for (int q = 0; q < Queries; ++q) { count += s.counts[q]; }
            s.valid_count = count;
        }
    } else {
        // First scan the number of live tokens per 1024-token physical range.
        int const lane = tid % 32;
        for (int w = tid; w < kPairWords; w += blockDim.x) {
            unsigned live = 0;
            for (int q = 0; q < Queries; ++q) { live |= s.selected[q][w]; }
            int count = __popc(live);
            for (int delta = 16; delta; delta /= 2) { count += __shfl_xor_sync(0xffffffffu, count, delta); }
            if (!lane) { s.warp_prefix[w / 32] = count; }
        }
        __syncthreads();
        if (!tid) {
            int total = 0;
            for (int w = 0; w < kPairWords / 32; ++w) {
                int const n = s.warp_prefix[w];
                s.warp_prefix[w] = total;
                total += n;
            }
            s.valid_count = total;
        }
        __syncthreads();
        for (int w = tid; w < kPairWords; w += blockDim.x) {
            unsigned selected[Queries], live = 0;
#pragma unroll
            for (int q = 0; q < Queries; ++q) {
                selected[q] = s.selected[q][w];
                live |= selected[q];
            }
            int prefix = __popc(live);
            for (int delta = 1; delta < 32; delta *= 2) {
                int const prev = __shfl_up_sync(0xffffffffu, prefix, delta);
                if (lane >= delta) { prefix += prev; }
            }
            int out = s.warp_prefix[w / 32] + prefix - __popc(live);
            while (live) {
                int const bit = __ffs(live) - 1;
                int mask = 0;
#pragma unroll
                for (int q = 0; q < Queries; ++q) {
                    mask |= ((selected[q] >> bit) & 1u) << q;
                }
                s.entries[out++] = (w * 32 + bit) | (mask << kPairTokenBits);
                live &= live - 1;
            }
        }
    }
    __syncthreads();
}

template <typename T, int Arch, int M, int N, bool PrefetchKV, bool CacheOffsets,
          int Queries = 1, bool Deterministic = false, bool PrivateKV = false>
__global__ void accumulate_backward_mma(Flash_bwd_params p, int64_t tile_groups,
                                       int tile_group, int64_t work_count, QsaBackwardTileBatch batch) {
    constexpr bool GroupQueries = Queries > 1;
    static_assert(!Deterministic || (!GroupQueries && !PrefetchKV && !CacheOffsets
                                     && N == QsaBackwardTileBatch::kCols));
    static_assert(!PrivateKV || Queries == 10);
    using Config = BackwardMma<T, Arch, M, N, Queries>;
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    constexpr int kStripeDim = GroupQueries ? 256 : 64;
    using Storage = std::conditional_t<GroupQueries, typename Config::GroupStorage,
                                       typename Config::template SharedStorage<PrefetchKV>>;
    auto& shared = *reinterpret_cast<Storage*>(shared_bytes);
    // Pack twelve GQA heads per query in the larger group. Map padding to
    // an inactive head and a valid count index, including incomplete groups.
    auto row_query = [](int row) {
        if constexpr (Queries > 2) { return row < 12 * Queries ? row / 12 : 0; }
        return GroupQueries ? row / 16 : 0;
    };
    auto row_head = [](int row) {
        if constexpr (Queries > 2) { return row < 12 * Queries ? row % 12 : row; }
        return GroupQueries ? row % 16 : row;
    };
    auto row_count = [&](int row) {
        if constexpr (GroupQueries) { return shared.counts[row_query(row)]; }
        else { return shared.valid_count; }
    };
    auto sq = make_tensor(make_smem_ptr(shared.q), typename Config::LayoutQ{});
    auto sdo = make_tensor(make_smem_ptr(shared.dout), typename Config::LayoutQ{});
    auto sk = make_tensor(make_smem_ptr(shared.k), typename Config::LayoutKV{});
    constexpr int VWidth = PrefetchKV ? Config::kVPanel : kHeadDim;
    constexpr bool StageFullV = Queries == 10;
    constexpr int VSize = cosize_v<typename Config::template LayoutV<VWidth>>;
    constexpr int PSSize = cosize_v<typename Config::LayoutPS>;
    // V occupies the first panel and the inactive dS buffer. The P buffer
    // stays independent because the last GEMM still consumes it during copy.
    static_assert(!StageFullV || (VSize == PSSize && 2 * VWidth == kHeadDim));
    auto* p_buffer = shared.v_ps_ds + (StageFullV ? 2 * VSize : VSize);
    auto* ds_buffer = shared.v_ps_ds + (StageFullV ? VSize : VSize + PSSize);
    auto sv = make_tensor(make_smem_ptr(shared.v_ps_ds),
                          typename Config::template LayoutV<StageFullV ? kHeadDim : VWidth>{});
    auto sp = make_tensor(make_smem_ptr(p_buffer), typename Config::LayoutPS{});
    auto sds = make_tensor(make_smem_ptr(ds_buffer), typename Config::LayoutPS{});
    auto sqt = make_tensor(make_smem_ptr(shared.q), typename Config::LayoutQt{});
    auto sdot = make_tensor(make_smem_ptr(shared.dout), typename Config::LayoutQt{});
    auto skt = make_tensor(make_smem_ptr(shared.k), typename Config::LayoutKt{});
    auto spt = make_tensor(make_smem_ptr(p_buffer), typename Config::LayoutPSt{});
    auto sdst = make_tensor(make_smem_ptr(ds_buffer), typename Config::LayoutPSt{});
    auto const* q = static_cast<T const*>(p.q_ptr);
    auto const* k = static_cast<T const*>(p.k_ptr);
    auto const* v = static_cast<T const*>(p.v_ptr);
    auto const* o = static_cast<T const*>(p.o_ptr);
    auto const* dout = static_cast<T const*>(p.do_ptr);
    auto const* lse_data = static_cast<float const*>(p.softmax_lse_ptr);
    auto* dq_accum = static_cast<float*>(p.dq_accum_ptr);
    auto* dk_accum = static_cast<float*>(p.dk_accum_ptr);
    auto* dv_accum = static_cast<float*>(p.dv_accum_ptr);
    if constexpr (PrivateKV) {
        int const plane = blockIdx.x / p.h_k;
        int64_t const stride = int64_t(p.num_pages) * p.page_size * p.h_k * kHeadDim;
        dk_accum += plane * stride;
        dv_accum += plane * stride;
    }
    int const group = p.h / p.h_k;
    int const lane = threadIdx.x % 32;
    int const warp = threadIdx.x / 32;
    bool const flat_pages = p.k_batch_stride == int64_t(p.page_size) * p.k_row_stride
        && p.v_batch_stride == int64_t(p.page_size) * p.v_row_stride;

    auto valid_for_query = [&](int64_t query) {
        if constexpr (Queries == 10) {
            if (!p.is_causal || !p.seqused_k) { return p.seqlen_k; }
            // Validation bounds every length by one. With as many queries as
            // sequences, every sequence therefore contains exactly one query.
            if (p.seqlen_q == 1 && p.b == p.total_q) {
                int const count = p.seqused_k[query];
                return count > p.seqlen_k ? p.seqlen_k : count;
            }
        }
        int lo = 0, hi = p.b;
        while (lo < hi) {
            int const mid = lo + (hi - lo) / 2;
            if (int64_t(p.cu_seqlens_q[mid]) <= query) { lo = mid + 1; }
            else { hi = mid; }
        }
        int const batch = lo - 1;
        int64_t count = p.seqlen_k;
        if (p.is_causal && p.seqused_k) {
            int64_t const cache = p.seqused_k[batch];
            int64_t const qpos = query - p.cu_seqlens_q[batch];
            int64_t const qlen = int64_t(p.cu_seqlens_q[batch + 1]) - p.cu_seqlens_q[batch];
            int64_t const causal_end = qpos + cache - qlen + 1;
            count = cache < causal_end ? cache : causal_end;
            count = count < 0 ? 0 : count;
            count = count > p.seqlen_k ? p.seqlen_k : count;
        }
        return static_cast<int>(count);
    };

    constexpr int RowsPerWarp = N / Config::kWarps;
    int const token_col = warp * RowsPerWarp + lane % RowsPerWarp;
    auto load_offset = [&](int64_t query, int head, int begin, int count) {
        if (token_col >= count - begin) { return 0; }
        if constexpr (GroupQueries) {
            if constexpr (Queries > 2) {
                // A concatenation with duplicate rows can exceed the physical pool.
                // In that case, read each occurrence from its original query.
                if (shared.duplicates) {
                    int idx = begin + token_col;
#pragma unroll
                    for (int q = 0; q < Queries; ++q) {
                        if (idx < shared.counts[q]) {
                            int const token = p.page_table[(query + q) * p.page_table_batch_stride + idx];
                            return token | (1 << (kPairTokenBits + q));
                        }
                        idx -= shared.counts[q];
                    }
                    return 0;
                }
            }
            return int(shared.entries[begin + token_col]);
        }
        int const token = p.page_table[query * p.page_table_batch_stride + begin + token_col];
        return (token * p.h_k + head) * kHeadDim;
    };
    int cached_offset = 0;
    bool first_tile_ready = false;
    int64_t next_ordinal;
    for (int64_t ordinal = blockIdx.x; ordinal < work_count; ordinal = next_ordinal) {
        // Later queries in a causal sequence generally have more sparse tiles.
        int64_t const work = Deterministic ? batch.first + ordinal : (PrefetchKV ? work_count - 1 - ordinal : ordinal);
        int64_t const query_kv = work / tile_groups;
        int64_t const query = Queries * (query_kv / p.h_k);
        int64_t const kv_head = query_kv % p.h_k;
        int64_t const group_begin = (work % tile_groups) * N * tile_group;
        int64_t const head_begin = kv_head * group;
        if (threadIdx.x == 0) {
            if constexpr (PrefetchKV) {
                if (p.tile_count_semaphore) {
                    shared.next_job = atomicAdd(p.tile_count_semaphore, 1) + int(gridDim.x);
                }
            }
            if constexpr (Queries == 2) {
                for (int q = 0; q < Queries; ++q) {
                    shared.counts[q] = query + q < p.total_q ? valid_for_query(query + q) : 0;
                }
            } else if constexpr (!GroupQueries) { shared.valid_count = valid_for_query(query); }
        }
        if constexpr (Queries == 10) {
            int const q = threadIdx.x;
            if (q < Queries) { shared.counts[q] = query + q < p.total_q ? valid_for_query(query + q) : 0; }
        }
        __syncthreads();
        next_ordinal = PrefetchKV && p.tile_count_semaphore ? shared.next_job : ordinal + int64_t(gridDim.x);
        if constexpr (GroupQueries) { build_query_group<Queries>(shared, p, query); }
        if constexpr (Deterministic) {
            if (threadIdx.x < N) {
                int64_t const col = group_begin + threadIdx.x;
                int64_t key = -1;
                if (col < shared.valid_count) {
                    int64_t const token = p.page_table[query * p.page_table_batch_stride + col];
                    key = token * p.h_k + kv_head;
                }
                batch.keys[ordinal * N + threadIdx.x] = key;
            }
        }
        if (group_begin >= shared.valid_count) {
            if constexpr (Deterministic) {
                for (int i = threadIdx.x; i < group * kHeadDim; i += blockDim.x) {
                    batch.dq[ordinal * group * kHeadDim + i] = 0.f;
                }
            }
            __syncthreads();
            continue;
        }
        if constexpr (CacheOffsets) {
            if (!first_tile_ready) { cached_offset = load_offset(query, kv_head, 0, shared.valid_count); }
        }
        for (int row = threadIdx.x; row < M; row += Config::kMmaThreads) {
            shared.lse[row] = row_head(row) < group && (!GroupQueries || row_count(row) > 0)
                ? lse_data[(head_begin + row_head(row)) * int64_t(p.total_q) + query + row_query(row)] : -CUDART_INF_F;
        }
        __syncthreads();
        unsigned dq_row_masks[(M + 31) / 32];
#pragma unroll
        for (int i = 0; i < (M + 31) / 32; ++i) {
            int const row = 32 * i + lane;
            dq_row_masks[i] = __ballot_sync(0xffffffffu,
                row < M && row_head(row) < group && shared.lse[row] != -CUDART_INF_F
                && !(row_count(row) == 1 && isfinite(shared.lse[row])));
        }
        for (int i = threadIdx.x; i < M * (kHeadDim / 8); i += Config::kMmaThreads) {
            int const row = i / (kHeadDim / 8), dim = (i % (kHeadDim / 8)) * 8;
            bool const active = row_head(row) < group && shared.lse[row] != -CUDART_INF_F;
            bool const singleton = row_count(row) == 1 && isfinite(shared.lse[row]);
            int64_t const head = head_begin + row_head(row);
            uint4 qval = make_uint4(0, 0, 0, 0), doval = make_uint4(0, 0, 0, 0);
            if (active && !singleton) {
                qval = load_chunk(q + (query + row_query(row)) * p.q_row_stride + head * p.q_head_stride + dim);
            }
            if (active) {
                doval = load_chunk(dout + (query + row_query(row)) * p.do_row_stride + head * p.do_head_stride + dim);
            }
            if constexpr (Queries == 10) {
                // One warp stages the entire row. Consume its eight contiguous
                // dO values per lane while O is loaded in the same format.
                float delta = 0.f;
                if (active && !singleton) {
                    int64_t const offset = (query + row_query(row)) * p.o_row_stride + head * p.o_head_stride + dim;
                    uint4 oval = load_chunk(o + offset);
                    cutlass::Array<T, 8> o_values, do_values;
                    __builtin_memcpy(&o_values, &oval, sizeof(oval));
                    __builtin_memcpy(&do_values, &doval, sizeof(doval));
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        delta = fmaf(float(do_values[j]), float(o_values[j]), delta);
                    }
                }
                delta = warp_sum(delta);
                if (lane == 0) {
                    unsigned const eligible = active ? 1u << row_query(row) : 0u;
                    // A singleton stages Q=0 and uses LSE=0 so exp(score - LSE)=1.
                    shared.softmax_rows[row] = {singleton ? 0.f : shared.lse[row], delta,
                                                eligible, singleton ? 0u : eligible};
                }
            }
            *reinterpret_cast<uint4*>(&sq(row, dim)) = qval;
            *reinterpret_cast<uint4*>(&sdo(row, dim)) = doval;
        }
        if constexpr (Queries != 10) {
            __syncthreads();
            for (int row = warp; row < M; row += Config::kWarps) {
                float delta = 0.f;
                if (row_head(row) < group && shared.lse[row] != -CUDART_INF_F
                    && !(row_count(row) == 1 && isfinite(shared.lse[row]))) {
                    int64_t const head = head_begin + row_head(row);
                    int64_t const o_offset = (query + row_query(row)) * p.o_row_stride + head * p.o_head_stride;
#pragma unroll
                    for (int dim = lane; dim < kHeadDim; dim += 32) {
                        delta = fmaf(float(sdo(row, dim)), float(o[o_offset + dim]), delta);
                    }
                }
                delta = warp_sum(delta);
                if (lane == 0) { shared.delta[row] = delta; }
            }
        }
        // When a CTA owns all sparse tiles for a (query, KV head), it also owns
        // those query-head gradients. Keep the FP32 sums until the final store.
        typename Config::MmaDQ dq_mma;
        using DqFragment = decltype(partition_fragment_C(dq_mma, Shape<Int<M>, Int<kStripeDim>>{}));
        DqFragment dq_stripes[PrefetchKV ? kHeadDim / kStripeDim : 1];
        if constexpr (PrefetchKV) {
#pragma unroll
            for (int stripe = 0; stripe < kHeadDim / kStripeDim; ++stripe) { clear(dq_stripes[stripe]); }
        }
        using GradShape = Shape<Int<kHeadDim>, Int<N>>;
        typename Config::MmaSdP grad_mma;
        using GradFragment = decltype(partition_fragment_C(grad_mma, GradShape{}));
        auto scatter_part = [&](auto const& values, int const* rows, int count, auto* dest,
                                float scale, int step, int steps) {
            auto coords = grad_mma.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(GradShape{}));
            auto write = [&](auto full) {
                if constexpr (Queries == 10) {
                    auto rc = make_tensor(coords.data(), flash::convert_layout_acc_rowcol(coords.layout()));
                    auto grad = make_tensor(values.data(), flash::convert_layout_acc_rowcol(values.layout()));
                    constexpr int Rows = decltype(size<0>(rc))::value;
                    constexpr int Cols = decltype(size<1>(rc))::value;
                    if constexpr (PrivateKV && decltype(full)::value) {
                        // Each 16x16 MMA gives lanes l and l^2 two token columns
                        // four apart, with eight adjacent dimensions in each half.
                        // Exchange the inner pair to write two aligned 64-byte
                        // segments per warp instead of four 32-byte segments.
                        constexpr int Width = Rows * Cols;
                        static_assert(Arch == 89 && M == 128 && N == 64 && Config::kMmaThreads == 512);
                        static_assert(Rows % 2 == 0 && Width % 32 == 0);
                        int const first = step * Width / steps;
                        int const last = (step + 1) * Width / steps;
                        bool const upper = (threadIdx.x & 2) != 0;
#pragma unroll
                        for (int slot = first; slot < last; slot += 2) {
                            int const r = slot % Rows, c = slot / Rows;
                            int const col = get<1>(rc(0, c)) - (upper ? 4 : 0);
                            int const dim = get<0>(rc(r, c)) + (upper ? 8 : 0);
                            float const x = grad(r, c), y = grad(r + 1, c);
                            float const other = __shfl_xor_sync(0xffffffffu, upper ? x : y, 2);
                            atomicAdd(dest + uint32_t(rows[col]) + dim, scale * (upper ? other : x));
                            atomicAdd(dest + uint32_t(rows[col + 4]) + dim, scale * (upper ? y : other));
                        }
                        return;
                    }
                    // Partition a column-major scatter into equally sized hook steps.
                    int const first = step * Rows * Cols / steps;
                    int const last = (step + 1) * Rows * Cols / steps;
#pragma unroll
                    for (int c = first / Rows; c < (last + Rows - 1) / Rows; ++c) {
                        int const col = get<1>(rc(0, c));
                        if (decltype(full)::value || col < count) {
                            auto* physical = dest + uint32_t(rows[col]) + get<0>(rc(0, c));
#pragma unroll
                            for (int r = 0; r < Rows; ++r) {
                                int const slot = c * Rows + r;
                                if (slot >= first && slot < last) {
                                    int const dim_offset = get<0>(rc(r, c)) - get<0>(rc(0, c));
                                    // Private planes have one writer per address per query group;
                                    // a CTA barrier separates successive groups on that plane.
                                    atomicAdd(physical + dim_offset, scale * grad(r, c));
                                }
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int i = step * size(values) / steps; i < (step + 1) * size(values) / steps; ++i) {
                        int const col = get<1>(coords(i));
                        if (decltype(full)::value || col < count) {
                            uint32_t const offset = uint32_t(rows[col]) + get<0>(coords(i));
                            atomicAdd(dest + offset, scale * values(i));
                        }
                    }
                }
            };
            if (count >= N) { write(std::true_type{}); } else { write(std::false_type{}); }
        };
        GradFragment pending_v;
        auto prefetch_kv = [&](int tile, int64_t query, int64_t kv_head, int count) {
            int64_t const begin = group_begin + int64_t(tile) * N;
            for (int step = 0; step < RowsPerWarp; ++step) {
                int const col = CacheOffsets ? warp * RowsPerWarp + step : step * Config::kWarps + warp;
                int const dim = lane * 8;
                bool const valid = begin + col < count;
                T const *k_src, *v_src;
                if constexpr (CacheOffsets) {
                    uint32_t const entry = __shfl_sync(0xffffffffu, cached_offset, step);
                    uint32_t const offset = GroupQueries ? ((entry % kPairPool) * p.h_k + kv_head) * kHeadDim : entry;
                    k_src = k + (offset + dim);
                    v_src = v + (offset + dim);
                } else {
                    int64_t const token = valid ? p.page_table[query * p.page_table_batch_stride + begin + col] : 0;
                    k_src = k + token * p.k_row_stride + kv_head * p.k_head_stride + dim;
                    v_src = v + token * p.v_row_stride + kv_head * p.v_head_stride + dim;
                }
                using Copy = PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>;
                Copy::copy(*reinterpret_cast<uint4 const*>(k_src), *reinterpret_cast<uint4*>(&sk(col, dim)), valid);
                if (StageFullV || dim < VWidth) {
                    Copy::copy(*reinterpret_cast<uint4 const*>(v_src), *reinterpret_cast<uint4*>(&sv(col, dim)), valid);
                }
            }
            cute::cp_async_fence();
        };
        if constexpr (PrefetchKV) {
            if (!first_tile_ready) { prefetch_kv(0, query, kv_head, shared.valid_count); }
            first_tile_ready = false;
        }
        for (int tile = 0; tile < tile_group; ++tile) {
            int64_t const col_begin = group_begin + int64_t(tile) * N;
            if (col_begin >= shared.valid_count) { break; }
            if constexpr (CacheOffsets) {
                // Share this tile's rows for V and the dK/dV scatter before looking ahead.
                if (lane < RowsPerWarp) {
                    if constexpr (GroupQueries) {
                        shared.kv_row[token_col] = ((cached_offset % kPairPool) * p.h_k + kv_head) * kHeadDim;
                        shared.members[token_col] = cached_offset >> kPairTokenBits;
                    } else { shared.kv_row[token_col] = cached_offset; }
                }
            } else {
                for (int col = threadIdx.x; col < N; col += Config::kMmaThreads) {
                    shared.kv_row[col] = col_begin + col < shared.valid_count
                        ? p.page_table[query * p.page_table_batch_stride + col_begin + col] : 0;
                }
            }
            if constexpr (PrefetchKV) {
                flash::cp_async_wait<0>();
                __syncthreads();
                if constexpr (CacheOffsets) {
                    if (col_begin + N < shared.valid_count) {
                        cached_offset = load_offset(query, kv_head, static_cast<int>(col_begin) + N, shared.valid_count);
                    } else if (!GroupQueries && next_ordinal < work_count) {
                        int64_t const next_work = work_count - 1 - next_ordinal;
                        int64_t const next_query = next_work / p.h_k;
                        if (threadIdx.x == 0) { shared.next_valid_count = valid_for_query(next_query); }
                        __syncthreads();
                        cached_offset = load_offset(next_query, next_work % p.h_k, 0, shared.next_valid_count);
                    }
                }
            } else {
                __syncthreads();
                for (int i = threadIdx.x; i < N * (kHeadDim / 8); i += Config::kMmaThreads) {
                    int const col = i / (kHeadDim / 8), dim = (i % (kHeadDim / 8)) * 8;
                    uint4 kval = make_uint4(0, 0, 0, 0);
                    uint4 vval = make_uint4(0, 0, 0, 0);
                    if (col_begin + col < shared.valid_count) {
                        int64_t const token = shared.kv_row[col];
                        int64_t k_offset, v_offset;
                        if (flat_pages) {
                            k_offset = token * p.k_row_stride;
                            v_offset = token * p.v_row_stride;
                        } else {
                            int64_t const page = token / p.page_size, row = token % p.page_size;
                            k_offset = page * p.k_batch_stride + row * p.k_row_stride;
                            v_offset = page * p.v_batch_stride + row * p.v_row_stride;
                        }
                        kval = load_chunk(k + k_offset + kv_head * p.k_head_stride + dim);
                        vval = load_chunk(v + v_offset + kv_head * p.v_head_stride + dim);
                    }
                    *reinterpret_cast<uint4*>(&sk(col, dim)) = kval;
                    *reinterpret_cast<uint4*>(&sv(col, dim)) = vval;
                }
                __syncthreads();
            }
            {
                typename Config::MmaSdP mma;
                auto scores = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
                auto dp = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
                clear(scores);
                clear(dp);
                auto compute_scores = [&](auto hook) {
                    shared_gemm<typename Config::Normal, typename Config::Normal, Queries == 10>(mma, scores, sq, sk, hook);
                };
                if constexpr (GroupQueries) {
                    if (tile > 0) {
                        // Only full tiles have a successor. Keep their row offsets
                        // until this QK multiply has issued all pending dV updates.
                        auto write_previous_v = [&](int step, int steps) {
                            scatter_part(pending_v, shared.previous_row, N, dv_accum, 1.f, step, steps);
                        };
                        compute_scores(write_previous_v);
                    } else { compute_scores(nullptr); }
                } else { compute_scores(nullptr); }
                if constexpr (StageFullV) {
                    shared_gemm<typename Config::Normal, typename Config::Normal, Queries == 10>(mma, dp, sdo, sv);
                } else if constexpr (PrefetchKV) {
                    for (int panel = 0; panel < kHeadDim / VWidth; ++panel) {
                        auto do_panel = local_tile(sdo, Shape<Int<M>, Int<VWidth>>{}, make_coord(_0{}, panel));
                        shared_gemm<typename Config::Normal, typename Config::Normal>(mma, dp, do_panel, sv);
                        if (panel + 1 < kHeadDim / VWidth) {
                            __syncthreads();
                            for (int i = threadIdx.x; i < N * (VWidth / 8); i += Config::kMmaThreads) {
                                const int col = i / (VWidth / 8);
                                const int dim = (i % (VWidth / 8)) * 8;
                                int const row = shared.kv_row[col];
                                int const vdim = (panel + 1) * VWidth + dim;
                                auto const* src = CacheOffsets ? v + (uint32_t(row) + vdim)
                                    : v + int64_t(row) * p.v_row_stride + kv_head * p.v_head_stride + vdim;
                                PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>::copy(*reinterpret_cast<uint4 const*>(src),
                                    *reinterpret_cast<uint4*>(&sv(col, dim)), col_begin + col < shared.valid_count);
                            }
                            cute::cp_async_fence();
                            flash::cp_async_wait<0>();
                            __syncthreads();
                        }
                    }
                } else {
                    shared_gemm<typename Config::Normal, typename Config::Normal>(mma, dp, sdo, sv);
                }
                auto coords = mma.get_thread_slice(threadIdx.x).partition_C(
                    make_identity_tensor(Shape<Int<M>, Int<N>>{}));
                if constexpr (Queries == 10) {
                    // Each lane owns a rectangle of scores. Reuse a token mask
                    // across its rows, and a query's normalization across its columns.
                    auto rc = make_tensor(coords.data(), flash::convert_layout_acc_rowcol(coords.layout()));
                    auto prob = make_tensor(scores.data(), flash::convert_layout_acc_rowcol(scores.layout()));
                    auto grad = make_tensor(dp.data(), flash::convert_layout_acc_rowcol(dp.layout()));
                    int members[decltype(size<1>(rc))::value];
#pragma unroll
                    for (int c = 0; c < size<1>(rc); ++c) {
                        int const col = get<1>(rc(0, c));
                        // Padded columns have a zero membership tag from load_offset.
                        members[c] = shared.members[col];
                    }
                    constexpr int Rows = decltype(size<0>(rc))::value;
                    using RowInfo = typename Config::GroupStorage::SoftmaxRow;
                    RowInfo info[Rows];
#pragma unroll
                    for (int r = 0; r < Rows; ++r) {
                        int const row = get<0>(rc(r, 0));
                        info[r] = shared.softmax_rows[row];
                    }
#pragma unroll
                    for (int r = 0; r < size<0>(rc); ++r) {
                        auto const [lse, delta, eligible, differentiable] = info[r];
#pragma unroll
                        for (int c = 0; c < size<1>(rc); ++c) {
                            bool const selected = (members[c] & eligible) != 0;
                            float const weight = expf(p.scale_softmax * prob(r, c) - lse);
                            float const ds = weight * (grad(r, c) - delta);
                            prob(r, c) = selected ? weight : 0.f;
                            grad(r, c) = (members[c] & differentiable) ? ds : 0.f;
                        }
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < size(scores); ++i) {
                        int const row = get<0>(coords(i)), col = get<1>(coords(i));
                        float probability = 0.f, ds = 0.f;
                        bool selected = true;
                        if constexpr (GroupQueries) { selected = shared.members[col] & (1 << row_query(row)); }
                        if (selected && row_head(row) < group && col_begin + col < shared.valid_count
                            && shared.lse[row] != -CUDART_INF_F) {
                            if (row_count(row) == 1 && isfinite(shared.lse[row])) {
                                probability = 1.f;
                            } else {
                                probability = expf(p.scale_softmax * scores(i) - shared.lse[row]);
                                ds = probability * (dp(i) - shared.delta[row]);
                            }
                        }
                        scores(i) = probability;
                        dp(i) = ds;
                    }
                }
                auto rp = make_tensor_like<T>(scores);
                auto rds = make_tensor_like<T>(dp);
                flash::convert_type_out(scores, rp);
                flash::convert_type_out(dp, rds);
                auto store = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>{}, mma);
                auto thr_store = store.get_thread_slice(threadIdx.x);
                cute::copy(store, thr_store.retile_S(rp), thr_store.partition_D(sp));
                // All warps must finish reading V high before its storage becomes dS.
                if constexpr (StageFullV) { __syncthreads(); }
                cute::copy(store, thr_store.retile_S(rds), thr_store.partition_D(sds));
            }
            __syncthreads();
            auto compute_dq = [&](int stripe, auto hook) {
                typename Config::MmaDQ mma;
                auto acc = partition_fragment_C(mma, Shape<Int<M>, Int<kStripeDim>>{});
                clear(acc);
                auto kb = local_tile(skt, Shape<Int<kStripeDim>, Int<N>>{}, make_coord(stripe, _0{}));
                shared_gemm<typename Config::Normal, typename Config::Transpose>(mma, acc, sds, kb, hook);
                if constexpr (PrefetchKV) {
#pragma unroll
                    for (int i = 0; i < size(acc); ++i) {
                        dq_stripes[stripe](i) += p.scale_softmax * acc(i);
                    }
                } else {
                    auto coords = mma.get_thread_slice(threadIdx.x).partition_C(
                        make_identity_tensor(Shape<Int<M>, Int<kStripeDim>>{}));
#pragma unroll
                    for (int i = 0; i < size(acc); ++i) {
                        int const row = get<0>(coords(i)), dim = stripe * kStripeDim + get<1>(coords(i));
                        bool const active = (dq_row_masks[row / 32] >> (row % 32)) & 1u;
                        if constexpr (Deterministic) {
                            if (row < group) {
                                batch.dq[(ordinal * group + row) * kHeadDim + dim] =
                                    active ? p.scale_softmax * acc(i) : 0.f;
                            }
                        } else if (active) {
                            int64_t const offset = (query * p.h + head_begin + row) * kHeadDim + dim;
                            atomicAdd(dq_accum + offset, p.scale_softmax * acc(i));
                        }
                    }
                }
            };
            auto compute_dkv = [&](int stripe) {
                for (int gradient = 0; gradient < 2; ++gradient) {
                    // Transposing the dK/dV product changes which dimensions each lane
                    // scatters, while preserving the physical (token, head, dim) address.
                    using Mma = std::conditional_t<PrefetchKV, typename Config::MmaSdP, typename Config::MmaDKV>;
                    using OutputShape = std::conditional_t<PrefetchKV, Shape<Int<kStripeDim>, Int<N>>,
                                                            Shape<Int<N>, Int<kStripeDim>>>;
                    Mma mma;
                    auto acc = partition_fragment_C(mma, OutputShape{});
                    clear(acc);
                    auto multiply = [&](auto weights, auto operand) {
                        using Copy = typename Config::Transpose;
                        if constexpr (PrefetchKV) { shared_gemm<Copy, Copy>(mma, acc, operand, weights); }
                        else { shared_gemm<Copy, Copy>(mma, acc, weights, operand); }
                    };
                    if (gradient == 0) {
                        multiply(sdst, local_tile(sqt, Shape<Int<kStripeDim>, Int<M>>{}, make_coord(stripe, _0{})));
                    } else {
                        multiply(spt, local_tile(sdot, Shape<Int<kStripeDim>, Int<M>>{}, make_coord(stripe, _0{})));
                    }
                    auto coords = mma.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(OutputShape{}));
                    auto* accum = gradient == 0 ? dk_accum : dv_accum;
                    float const scale = gradient == 0 ? p.scale_softmax : 1.f;
                    auto scatter = [&](auto full_tile) {
                        if constexpr (Deterministic) {
                            auto* partial = gradient == 0 ? batch.dk : batch.dv;
#pragma unroll
                            for (int i = 0; i < size(acc); ++i) {
                                int const col = get<0>(coords(i));
                                int const dim = stripe * kStripeDim + get<1>(coords(i));
                                if (decltype(full_tile)::value || col_begin + col < shared.valid_count) {
                                    partial[(ordinal * N + col) * kHeadDim + dim] = scale * acc(i);
                                }
                            }
                        } else if constexpr (M == 32 && !GroupQueries) {
                            constexpr auto row_coords = typename Config::MmaDKV{}.get_thread_slice(Int<0>{}).partition_C(
                                make_identity_tensor(Shape<Int<N>, Int<kStripeDim>>{}));
                            cute::for_each(make_seq<size(acc)>{}, [&](auto i) {
                                constexpr int row = get<0>(row_coords(i));
                                constexpr bool first_row = [&] {
                                    for (int j = 0; j < int(i); ++j) {
                                        if (int(get<0>(row_coords(j))) == row) { return false; }
                                    }
                                    return true;
                                }();
                                if constexpr (first_row) {
                                    int const col = get<0>(coords(i));
                                    if (decltype(full_tile)::value || col_begin + col < shared.valid_count) {
                                        int64_t const offset = (int64_t(shared.kv_row[col]) * p.h_k + kv_head) * kHeadDim
                                            + stripe * kStripeDim;
                                        auto* destination = accum + offset;
                                        cute::for_each(make_seq<size(acc)>{}, [&](auto j) {
                                            if constexpr (get<0>(row_coords(j)) == row) {
                                                atomicAdd(destination + get<1>(coords(j)), scale * acc(j));
                                            }
                                        });
                                    }
                                }
                            });
                        } else {
#pragma unroll
                            for (int i = 0; i < size(acc); ++i) {
                                int const col = get<PrefetchKV ? 1 : 0>(coords(i));
                                int const dim = stripe * kStripeDim + get<PrefetchKV ? 0 : 1>(coords(i));
                                if (decltype(full_tile)::value || col_begin + col < shared.valid_count) {
                                    int const row = shared.kv_row[col];
                                    int64_t const offset = CacheOffsets ? uint32_t(row) + dim
                                        : (int64_t(row) * p.h_k + kv_head) * kHeadDim + dim;
                                    atomicAdd(accum + offset, scale * acc(i));
                                }
                            }
                        }
                    };
                    if (col_begin + N <= shared.valid_count) { scatter(std::true_type{}); }
                    else { scatter(std::false_type{}); }
                }
            };
            if constexpr (GroupQueries) {
                int const count = shared.valid_count - col_begin;
                bool const more = tile + 1 < tile_group && count > N;
                using Copy = typename Config::Transpose;
                GradFragment dk;
                clear(dk);
                shared_gemm<Copy, Copy>(grad_mma, dk, sqt, sdst);
                // dQ reads K for the last time. Spread this tile's dK writes
                // across those MMA steps before staging K/V for the next tile.
                auto write_k = [&](int step, int steps) {
                    scatter_part(dk, shared.kv_row, count, dk_accum, p.scale_softmax, step, steps);
                };
                compute_dq(0, write_k);
                if (more) {
                    __syncthreads();
                    prefetch_kv(tile + 1, query, kv_head, shared.valid_count);
                }
                clear(pending_v);
                shared_gemm<Copy, Copy>(grad_mma, pending_v, sdot, spt);
                if (more) {
                    if (threadIdx.x < N) { shared.previous_row[threadIdx.x] = shared.kv_row[threadIdx.x]; }
                } else {
                    scatter_part(pending_v, shared.kv_row, count, dv_accum, 1.f, 0, 1);
                }
            } else if constexpr (PrefetchKV) {
                for (int stripe = 0; stripe < kHeadDim / kStripeDim; ++stripe) { compute_dq(stripe, nullptr); }
                if (tile + 1 < tile_group && col_begin + N < shared.valid_count) {
                    // dQ is the last reader of K. Keep this tile's token IDs for the
                    // atomic scatter, and reuse its K/V storage while computing dK/dV.
                    __syncthreads();
                    prefetch_kv(tile + 1, query, kv_head, shared.valid_count);
                } else {
                    const int64_t next_work = work_count - 1 - next_ordinal;
                    if (!GroupQueries && next_work >= 0) {
                        if constexpr (!CacheOffsets) {
                            if (threadIdx.x == 0) { shared.next_valid_count = valid_for_query(next_work / p.h_k); }
                        }
                        __syncthreads();
                        if (shared.next_valid_count > 0) {
                            // K/V are dead for this task. Stage the next task's first tile
                            // while the current task still has dK/dV scatter and dQ writeback.
                            prefetch_kv(0, next_work / p.h_k, next_work % p.h_k, shared.next_valid_count);
                            first_tile_ready = true;
                        }
                    }
                }
                for (int stripe = 0; stripe < kHeadDim / kStripeDim; ++stripe) { compute_dkv(stripe); }
            } else {
                for (int stripe = 0; stripe < kHeadDim / kStripeDim; ++stripe) {
                    compute_dq(stripe, nullptr);
                    compute_dkv(stripe);
                }
            }
            __syncthreads();
        }
        if constexpr (PrefetchKV) {
            auto coords = dq_mma.get_thread_slice(threadIdx.x).partition_C(
                make_identity_tensor(Shape<Int<M>, Int<kStripeDim>>{}));
#pragma unroll
            for (int stripe = 0; stripe < kHeadDim / kStripeDim; ++stripe) {
#pragma unroll
                for (int i = 0; i < size(dq_stripes[stripe]); ++i) {
                    int const row = get<0>(coords(i)), dim = stripe * kStripeDim + get<1>(coords(i));
                    if ((dq_row_masks[row / 32] >> (row % 32)) & 1u) {
                        dq_accum[((query + row_query(row)) * p.h + head_begin + row_head(row)) * kHeadDim + dim] = dq_stripes[stripe](i);
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void convert_accumulators(Flash_bwd_params p, int64_t q_count,
                                     int64_t kv_count, int64_t work_count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < work_count; i += int64_t(gridDim.x) * blockDim.x) {
        if (i < q_count) {
            static_cast<T*>(p.dq_ptr)[i] = T(static_cast<float const*>(p.dq_accum_ptr)[i]);
        } else if (i - q_count < kv_count) {
            int64_t const j = i - q_count;
            static_cast<T*>(p.dk_ptr)[j] = T(static_cast<float const*>(p.dk_accum_ptr)[j]);
        } else {
            int64_t const j = i - q_count - kv_count;
            static_cast<T*>(p.dv_ptr)[j] = T(static_cast<float const*>(p.dv_accum_ptr)[j]);
        }
    }
}

// A single writer per query/head/dimension reduces sparse tiles in column order.
__global__ void reduce_dq_tiles(Flash_bwd_params p, QsaBackwardTileBatch batch, int64_t tiles_per_row) {
    int const group = p.h / p.h_k;
    int64_t const row = batch.first / tiles_per_row + blockIdx.x / group;
    int const head = blockIdx.x % group;
    int64_t const row_start = row * tiles_per_row;
    int64_t const first = row_start > batch.first ? row_start - batch.first : 0;
    int64_t const last = row_start + tiles_per_row - batch.first;
    int64_t const end = last < batch.count ? last : batch.count;
    auto* output = static_cast<float*>(p.dq_accum_ptr) + (row * group + head) * kHeadDim;
    int const dim = threadIdx.x;
    float value = output[dim];
    for (int64_t tile = first; tile < end; ++tile) {
        value += batch.dq[(tile * group + head) * kHeadDim + dim];
    }
    output[dim] = value;
}

// Only the first warp in each sorted segment writes this KV token/head. Chunks
// execute on one stream, so the global summation order is independent of CTA scheduling.
__global__ void reduce_dkv_tiles(Flash_bwd_params p, QsaBackwardTileBatch batch,
                                 int64_t const* keys, int64_t const* order) {
    int const slot = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const slots = batch.count * QsaBackwardTileBatch::kCols;
    if (slot >= slots) { return; }
    int64_t const key = keys[slot];
    if (key < 0 || (slot && keys[slot - 1] == key)) { return; }
    int const lane = threadIdx.x % 32;
    auto* out_k = static_cast<float*>(p.dk_accum_ptr) + key * kHeadDim + lane;
    auto* out_v = static_cast<float*>(p.dv_accum_ptr) + key * kHeadDim + lane;
    float dk[kHeadDim / 32], dv[kHeadDim / 32];
#pragma unroll
    for (int i = 0; i < kHeadDim / 32; ++i) { dk[i] = out_k[32 * i]; dv[i] = out_v[32 * i]; }
    for (int j = slot; j < slots && keys[j] == key; ++j) {
        int64_t const src = order[j] * kHeadDim + lane;
#pragma unroll
        for (int i = 0; i < kHeadDim / 32; ++i) {
            dk[i] += batch.dk[src + 32 * i];
            dv[i] += batch.dv[src + 32 * i];
        }
    }
#pragma unroll
    for (int i = 0; i < kHeadDim / 32; ++i) { out_k[32 * i] = dk[i]; out_v[32 * i] = dv[i]; }
}

__global__ void sum_qsa_kv_planes(Flash_bwd_params p, int planes, int64_t elements,
                                   float const* dk, float const* dv) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += int64_t(gridDim.x) * blockDim.x) {
        float k = 0.f, v = 0.f;
        for (int plane = 0; plane < planes; ++plane) {
            k += dk[plane * elements + i];
            v += dv[plane * elements + i];
        }
        static_cast<float*>(p.dk_accum_ptr)[i] = k;
        static_cast<float*>(p.dv_accum_ptr)[i] = v;
    }
}

template <typename T, int Arch, int M, int N = kTileCols>
void launch_mma(Flash_bwd_params& p, hggcStream_t stream) {
    using Config = BackwardMma<T, Arch, M, N>;
    int64_t const tiles = (int64_t(p.seqlen_k) + N - 1) / N;
    int64_t const grouped_work = int64_t(p.total_q) * p.h_k * ((tiles + 3) / 4);
    auto launch = [&](auto prefetch_kv, auto cache_offsets, int tile_group) {
        int64_t const tile_groups = (tiles + tile_group - 1) / tile_group;
        int64_t const work_count = int64_t(p.total_q) * p.h_k * tile_groups;
        if (work_count == 0) { return; }
        auto kernel = accumulate_backward_mma<T, Arch, M, N, decltype(prefetch_kv)::value, decltype(cache_offsets)::value>;
        constexpr int bytes = sizeof(typename Config::template SharedStorage<decltype(prefetch_kv)::value>);
        CHECK_CUDA(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int blocks = launch_blocks(work_count, 1);
        if constexpr (decltype(prefetch_kv)::value) {
            // The 64-column V panel permits four resident CTAs per CU. Leave enough
            // counter range for the final unsuccessful claim from each worker.
            int64_t const resident = 4LL * p.num_sm;
            if (resident > 0 && resident < blocks && work_count <= INT_MAX - resident) {
                blocks = static_cast<int>(resident);
            } else {
                p.tile_count_semaphore = nullptr;
            }
        }
        kernel<<<blocks, Config::kMmaThreads, bytes, stream>>>(p, tile_groups, tile_group, work_count, {});
        CHECK_CUDA_KERNEL_LAUNCH();
    };
    if constexpr (Arch == 89 && M == 16 && std::is_same_v<T, cutlass::bfloat16_t>) {
        if (p.tile_count_semaphore && supports_query_group(p) && query_group_enabled()) {
            int device, max_smem;
            CHECK_CUDA(hggcGetDevice(&device));
            CHECK_CUDA(hggcDeviceGetAttribute(&max_smem, hggcDevAttrMaxSharedMemoryPerBlockOptin, device));
            auto launch_group = [&](auto group_size, auto tile_cols) {
                constexpr int Queries = decltype(group_size)::value, Cols = decltype(tile_cols)::value;
                constexpr int Rows = Queries == 2 ? 32 : 128;
                using GroupConfig = BackwardMma<T, Arch, Rows, Cols, Queries>;
                constexpr int bytes = sizeof(typename GroupConfig::GroupStorage);
                if (bytes > max_smem) { return false; }
                auto kernel = accumulate_backward_mma<T, Arch, Rows, Cols, true, true, Queries>;
                CHECK_CUDA(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
                int64_t const jobs = ((int64_t(p.total_q) + Queries - 1) / Queries) * p.h_k;
                int blocks = launch_blocks(jobs, 1);
                if (p.num_sm > 0 && blocks > p.num_sm) { blocks = p.num_sm; }
                // The last claim advances past the job count. Use a static stride
                // when those final tickets would overflow the int32 semaphore.
                if (jobs > INT_MAX - blocks) { p.tile_count_semaphore = nullptr; }
                int const tile_group = (Queries * p.seqlen_k + Cols - 1) / Cols;
                kernel<<<blocks, GroupConfig::kMmaThreads, bytes, stream>>>(p, 1, tile_group, jobs, {});
                CHECK_CUDA_KERNEL_LAUNCH();
                return true;
            };
            // Prefer 10Q when it fits. An explicit group size of 2 selects the smaller tile.
            char const* group_size = std::getenv("FA3_QSA_BWD_QUERY_GROUP");
            if ((group_size == nullptr || std::strcmp(group_size, "10") == 0)
                && launch_group(Int<10>{}, Int<64>{})) { return; }
            if (launch_group(Int<2>{}, Int<128>{})) { return; }
        }
        auto const strides = p.k_row_stride | p.v_row_stride | p.k_head_stride | p.v_head_stride;
        auto const addresses = reinterpret_cast<uintptr_t>(p.k_ptr) | reinterpret_cast<uintptr_t>(p.v_ptr);
        bool const aligned_kv = addresses % 16 == 0 && strides % (16 / sizeof(T)) == 0;
        bool const flat_pages = p.k_batch_stride == int64_t(p.page_size) * p.k_row_stride
                             && p.v_batch_stride == int64_t(p.page_size) * p.v_row_stride;
        if (aligned_kv && flat_pages && p.h / p.h_k == 12 && p.total_q >= 2048 && tiles >= 16 && tiles <= 128) {
            int64_t const row_stride = int64_t(p.h_k) * kHeadDim;
            bool const packed = p.k_row_stride == row_stride && p.v_row_stride == row_stride
                && p.k_head_stride == kHeadDim && p.v_head_stride == kHeadDim
                && int64_t(p.num_pages) * p.page_size <= INT_MAX / row_stride;
            if (packed) { launch(std::true_type{}, std::true_type{}, static_cast<int>(tiles)); }
            else { launch(std::true_type{}, std::false_type{}, static_cast<int>(tiles)); }
            return;
        }
    }
    launch(std::false_type{}, std::false_type{}, tiles >= 16 && grouped_work >= 1024 ? 4 : 1);
}

template <typename T, int Arch, int M>
void launch_backward(Flash_bwd_params& p, QsaBackwardTileBatch batch, hggcStream_t stream) {
    if (!batch.count) { launch_mma<T, Arch, M>(p, stream); return; }
    constexpr int cols = QsaBackwardTileBatch::kCols;
    using Config = BackwardMma<T, Arch, M, cols>;
    auto kernel = accumulate_backward_mma<T, Arch, M, cols, false, false, 1, true>;
    constexpr int bytes = sizeof(typename Config::template SharedStorage<false>);
    CHECK_CUDA(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    int64_t const tiles = (int64_t(p.seqlen_k) + cols - 1) / cols;
    kernel<<<batch.count, Config::kMmaThreads, bytes, stream>>>(p, tiles, 1, batch.count, batch);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template <typename T, int Arch>
void launch_backward(Flash_bwd_params& p, QsaBackwardTileBatch batch, hggcStream_t stream) {
    if (p.h / p.h_k <= 16) { launch_backward<T, Arch, 16>(p, batch, stream); }
    else { launch_backward<T, Arch, 32>(p, batch, stream); }
}

template <typename T>
void launch_convert(Flash_bwd_params const& p, hggcStream_t stream) {
    int64_t const q_count = int64_t(p.total_q) * p.h * kHeadDim;
    int64_t const kv_count = int64_t(p.num_pages) * p.page_size * p.h_k * kHeadDim;
    int64_t const output_count = q_count + 2 * kv_count;
    if (output_count > 0) {
        convert_accumulators<T><<<launch_blocks(output_count), kThreads, 0, stream>>>(
            p, q_count, kv_count, output_count);
        CHECK_CUDA_KERNEL_LAUNCH();
    }
}
}  // namespace

void validate_qsa_bwd(Flash_bwd_params const& p, int* error, bool check_order, hggcStream_t stream) {
    int64_t const table_count = int64_t(p.total_q) * p.seqlen_k;
    int64_t const sequence_count = int64_t(p.b) + 1;
    int64_t const work_count = table_count > sequence_count ? table_count : sequence_count;
    validate_inputs<<<launch_blocks(work_count), kThreads, 0, stream>>>(
        p, error, table_count, work_count, check_order);
    CHECK_CUDA_KERNEL_LAUNCH();
    // Validate every shape that could be grouped, including when another host
    // thread changes the scheduling override before the launch.
    if (check_order || (!p.deterministic && supports_query_group(p))) {
        int64_t const q_chunks = int64_t(p.total_q) * p.h * (kHeadDim / 8);
        int64_t const kv_chunks = int64_t(p.num_pages) * p.page_size * p.h_k * (kHeadDim / 8);
        check_pair_operands<<<launch_blocks(q_chunks + kv_chunks), kThreads, 0, stream>>>(
            p, error, q_chunks, kv_chunks);
        CHECK_CUDA_KERNEL_LAUNCH();
    }
}

int qsa_deterministic_kv_planes(Flash_bwd_params const& p) {
    if (!deterministic_group_candidate(p)) { return 0; }
    using Config = BackwardMma<cutlass::bfloat16_t, 89, 128, kTileCols, 10>;
    int device, max_shared;
    CHECK_CUDA(hggcGetDevice(&device));
    CHECK_CUDA(hggcDeviceGetAttribute(&max_shared, hggcDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (sizeof(typename Config::GroupStorage) > static_cast<size_t>(max_shared)) { return 0; }
    constexpr int64_t max_bytes = 512LL << 20;
    int64_t const bytes_per_plane = 2LL * sizeof(float) * p.num_pages * p.page_size * p.h_k * kHeadDim;
    int64_t const groups = (int64_t(p.total_q) + 9) / 10;
    return std::min({max_bytes / bytes_per_plane, groups, int64_t(std::max(1, p.num_sm / p.h_k))});
}

void run_qsa_bwd_planes(Flash_bwd_params const& p, int planes, float* dk, float* dv, hggcStream_t stream) {
    using Config = BackwardMma<cutlass::bfloat16_t, 89, 128, kTileCols, 10>;
    auto kernel = accumulate_backward_mma<cutlass::bfloat16_t, 89, 128, kTileCols, true, true, 10, false, true>;
    constexpr int bytes = sizeof(typename Config::GroupStorage);
    CHECK_CUDA(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    // Both the number of jobs and the static grid stride are multiples of Hkv:
    // successive jobs on each CTA retain the same KV head.
    Flash_bwd_params private_params = p;
    private_params.dk_accum_ptr = dk;
    private_params.dv_accum_ptr = dv;
    private_params.tile_count_semaphore = nullptr;
    int64_t const jobs = ((int64_t(p.total_q) + 9) / 10) * p.h_k;
    int const group_tiles = (10 * p.seqlen_k + kTileCols - 1) / kTileCols;
    kernel<<<planes * p.h_k, Config::kMmaThreads, bytes, stream>>>(private_params, 1, group_tiles, jobs, {});
    CHECK_CUDA_KERNEL_LAUNCH();
    int64_t const elements = int64_t(p.num_pages) * p.page_size * p.h_k * kHeadDim;
    sum_qsa_kv_planes<<<launch_blocks(elements), kThreads, 0, stream>>>(p, planes, elements, dk, dv);
    CHECK_CUDA_KERNEL_LAUNCH();
}

static void dispatch_qsa_bwd(Flash_bwd_params& p, QsaBackwardTileBatch batch, hggcStream_t stream) {
    TORCH_CHECK(p.is_qsa && p.d == kHeadDim && p.dv == kHeadDim,
                "QSA backward requires QSA with Q/K/V head dimension 256");
    if (p.is_bf16) {
        if (p.arch >= 89) { launch_backward<cutlass::bfloat16_t, 89>(p, batch, stream); }
        else { launch_backward<cutlass::bfloat16_t, 80>(p, batch, stream); }
    } else {
#ifndef FLASHATTENTION_DISABLE_FP16
        if (p.arch >= 89) { launch_backward<cutlass::half_t, 89>(p, batch, stream); }
        else { launch_backward<cutlass::half_t, 80>(p, batch, stream); }
#else
        TORCH_CHECK(false, "QSA backward FP16 support is disabled");
#endif
    }
}

void run_qsa_bwd(Flash_bwd_params& p, hggcStream_t stream) {
    dispatch_qsa_bwd(p, {}, stream);
}

void compute_qsa_bwd_tiles(Flash_bwd_params& p, QsaBackwardTileBatch batch, hggcStream_t stream) {
    dispatch_qsa_bwd(p, batch, stream);
}

void reduce_qsa_bwd_tiles(Flash_bwd_params const& p, QsaBackwardTileBatch batch,
                          int64_t const* keys, int64_t const* order, hggcStream_t stream) {
    constexpr int cols = QsaBackwardTileBatch::kCols;
    int64_t const tiles = (int64_t(p.seqlen_k) + cols - 1) / cols;
    int const rows = (batch.first + batch.count - 1) / tiles - batch.first / tiles + 1;
    reduce_dq_tiles<<<rows * (p.h / p.h_k), kHeadDim, 0, stream>>>(p, batch, tiles);
    CHECK_CUDA_KERNEL_LAUNCH();
    reduce_dkv_tiles<<<launch_blocks(batch.count * cols, kThreads / 32), kThreads, 0, stream>>>(
        p, batch, keys, order);
    CHECK_CUDA_KERNEL_LAUNCH();
}

void convert_qsa_bwd(Flash_bwd_params const& p, hggcStream_t stream) {
    if (p.is_bf16) { launch_convert<cutlass::bfloat16_t>(p, stream); }
#ifndef FLASHATTENTION_DISABLE_FP16
    else { launch_convert<cutlass::half_t>(p, stream); }
#endif
}

#endif
