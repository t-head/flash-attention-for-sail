// Copyright (c) 2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
#include <hggc_runtime.h>
#include <math_constants.h>
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
constexpr int kThreads = 128;
constexpr int kStripeDim = 64;
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
                                int64_t table_count, int64_t work_count) {
    int flags = 0;
    int64_t const pool_tokens = int64_t(p.num_pages) * p.page_size;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < work_count; i += int64_t(gridDim.x) * blockDim.x) {
        if (i < table_count) {
            int64_t const row = i / p.seqlen_k;
            int64_t const col = i % p.seqlen_k;
            int64_t const token = p.page_table[row * p.page_table_batch_stride + col];
            if (token < 0 || token >= pool_tokens) { flags |= 1; }
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

template <typename T, int Arch, int M, int N = kTileCols>
struct BackwardMma {
    static constexpr int kWarps = N / 16;
    static constexpr int kMmaThreads = 32 * kWarps;
    using Atom = MMA_Atom<std::conditional_t<Arch >= 89,
        std::conditional_t<std::is_same_v<T, cutlass::half_t>,
            PPU0015_16x16x16_F32F16F16F32_TN, PPU0015_16x16x16_F32BF16BF16F32_TN>,
        std::conditional_t<std::is_same_v<T, cutlass::half_t>,
            PPU0010_16x16x16_F32F16F16F32_TN, PPU0010_16x16x16_F32BF16BF16F32_TN>>>;
    using MmaSdP = TiledMMA<Atom, Layout<Shape<_1, Int<kWarps>, _1>>,
                            Tile<_16, Int<N>, _16>>;
    using MmaDQ = MmaSdP;
    using MmaDKV = TiledMMA<Atom, Layout<Shape<Int<kWarps>, _1, _1>>,
                            Tile<Int<N>, _16, _16>>;
    using Normal = Copy_Atom<PPU_U32x4_LDSM_N, T>;
    using Transpose = Copy_Atom<PPU_U16x8_LDSM_T, T>;
    using AtomQKV = decltype(composition(Swizzle<3, 3, 3>{},
        Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
    using LayoutQ = decltype(tile_to_shape(AtomQKV{}, Shape<Int<M>, _256>{}));
    using LayoutKV = decltype(tile_to_shape(AtomQKV{}, Shape<Int<N>, _256>{}));
    using AtomPS = decltype(composition(Swizzle<N == 64 ? 3 : 2, 3, 3>{},
        Layout<Shape<_8, Int<N>>, Stride<Int<N>, _1>>{}));
    using LayoutPS = decltype(tile_to_shape(AtomPS{}, Shape<Int<M>, Int<N>>{}));
    using LayoutQt = decltype(composition(LayoutQ{},
        Layout<Shape<_256, Int<M>>, Stride<Int<M>, _1>>{}));
    using LayoutKt = decltype(composition(LayoutKV{},
        Layout<Shape<_256, Int<N>>, Stride<Int<N>, _1>>{}));
    using LayoutPSt = decltype(composition(LayoutPS{},
        Layout<Shape<Int<N>, Int<M>>, Stride<Int<M>, _1>>{}));
    struct SharedStorage {
        alignas(128) T q[cosize_v<LayoutQ>];
        alignas(128) T dout[cosize_v<LayoutQ>];
        alignas(128) T k[cosize_v<LayoutKV>];
        alignas(128) T v[cosize_v<LayoutKV>];
        alignas(128) T probability[cosize_v<LayoutPS>];
        alignas(128) T ds[cosize_v<LayoutPS>];
        float lse[M], delta[M];
        int token[N], valid_count;
    };
};

template <typename CopyA, typename CopyB, typename Mma, typename Acc,
          typename TensorA, typename TensorB>
__device__ __forceinline__ void shared_gemm(Mma mma, Acc& acc,
                                           TensorA const& a, TensorB const& b) {
    auto thr_mma = mma.get_thread_slice(threadIdx.x);
    auto ra = thr_mma.partition_fragment_A(a);
    auto rb = thr_mma.partition_fragment_B(b);
    auto copy_a = make_tiled_copy_A(CopyA{}, mma);
    auto copy_b = make_tiled_copy_B(CopyB{}, mma);
    auto thr_a = copy_a.get_thread_slice(threadIdx.x);
    auto thr_b = copy_b.get_thread_slice(threadIdx.x);
    auto sa = thr_a.partition_S(a);
    auto sb = thr_b.partition_S(b);
    flash::gemm_sm80(acc, ra, rb, sa, sb, mma, copy_a, copy_b, thr_a, thr_b, nullptr);
}

template <typename T, int Arch, int M, int N = kTileCols>
__global__ void accumulate_backward_mma(Flash_bwd_params p, int64_t tile_groups,
                                       int tile_group, int64_t work_count) {
    using Config = BackwardMma<T, Arch, M, N>;
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<typename Config::SharedStorage*>(shared_bytes);
    auto sq = make_tensor(make_smem_ptr(shared.q), typename Config::LayoutQ{});
    auto sdo = make_tensor(make_smem_ptr(shared.dout), typename Config::LayoutQ{});
    auto sk = make_tensor(make_smem_ptr(shared.k), typename Config::LayoutKV{});
    auto sv = make_tensor(make_smem_ptr(shared.v), typename Config::LayoutKV{});
    auto sp = make_tensor(make_smem_ptr(shared.probability), typename Config::LayoutPS{});
    auto sds = make_tensor(make_smem_ptr(shared.ds), typename Config::LayoutPS{});
    auto sqt = make_tensor(make_smem_ptr(shared.q), typename Config::LayoutQt{});
    auto sdot = make_tensor(make_smem_ptr(shared.dout), typename Config::LayoutQt{});
    auto skt = make_tensor(make_smem_ptr(shared.k), typename Config::LayoutKt{});
    auto spt = make_tensor(make_smem_ptr(shared.probability), typename Config::LayoutPSt{});
    auto sdst = make_tensor(make_smem_ptr(shared.ds), typename Config::LayoutPSt{});
    auto const* q = static_cast<T const*>(p.q_ptr);
    auto const* k = static_cast<T const*>(p.k_ptr);
    auto const* v = static_cast<T const*>(p.v_ptr);
    auto const* o = static_cast<T const*>(p.o_ptr);
    auto const* dout = static_cast<T const*>(p.do_ptr);
    auto const* lse_data = static_cast<float const*>(p.softmax_lse_ptr);
    auto* dq_accum = static_cast<float*>(p.dq_accum_ptr);
    auto* dk_accum = static_cast<float*>(p.dk_accum_ptr);
    auto* dv_accum = static_cast<float*>(p.dv_accum_ptr);
    int const group = p.h / p.h_k;
    int const lane = threadIdx.x % 32;
    int const warp = threadIdx.x / 32;
    bool const flat_pages = p.k_batch_stride == int64_t(p.page_size) * p.k_row_stride
        && p.v_batch_stride == int64_t(p.page_size) * p.v_row_stride;

    for (int64_t work = blockIdx.x; work < work_count; work += int64_t(gridDim.x)) {
        int64_t const query_kv = work / tile_groups;
        int64_t const query = query_kv / p.h_k;
        int64_t const kv_head = query_kv % p.h_k;
        int64_t const group_begin = (work % tile_groups) * N * tile_group;
        int64_t const head_begin = kv_head * group;
        if (threadIdx.x == 0) {
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
            shared.valid_count = static_cast<int>(count);
        }
        __syncthreads();
        if (group_begin >= shared.valid_count) {
            __syncthreads();
            continue;
        }
        for (int row = threadIdx.x; row < M; row += Config::kMmaThreads) {
            shared.lse[row] = row < group
                ? lse_data[(head_begin + row) * int64_t(p.total_q) + query] : -CUDART_INF_F;
        }
        __syncthreads();
        unsigned const dq_row_mask = __ballot_sync(0xffffffffu,
            lane < group && shared.lse[lane] != -CUDART_INF_F
            && !(shared.valid_count == 1 && isfinite(shared.lse[lane])));
        for (int i = threadIdx.x; i < M * (kHeadDim / 8); i += Config::kMmaThreads) {
            int const row = i / (kHeadDim / 8), dim = (i % (kHeadDim / 8)) * 8;
            bool const active = row < group && shared.lse[row] != -CUDART_INF_F;
            bool const singleton = shared.valid_count == 1 && isfinite(shared.lse[row]);
            int64_t const head = head_begin + row;
            uint4 qval = make_uint4(0, 0, 0, 0), doval = make_uint4(0, 0, 0, 0);
            if (active && !singleton) {
                qval = load_chunk(q + query * p.q_row_stride + head * p.q_head_stride + dim);
            }
            if (active) {
                doval = load_chunk(dout + query * p.do_row_stride + head * p.do_head_stride + dim);
            }
            *reinterpret_cast<uint4*>(&sq(row, dim)) = qval;
            *reinterpret_cast<uint4*>(&sdo(row, dim)) = doval;
        }
        __syncthreads();
        for (int row = warp; row < M; row += Config::kWarps) {
            float delta = 0.f;
            if (row < group && shared.lse[row] != -CUDART_INF_F
                && !(shared.valid_count == 1 && isfinite(shared.lse[row]))) {
                int64_t const head = head_begin + row;
                int64_t const o_offset = query * p.o_row_stride + head * p.o_head_stride;
#pragma unroll
                for (int dim = lane; dim < kHeadDim; dim += 32) {
                    delta = fmaf(float(sdo(row, dim)), float(o[o_offset + dim]), delta);
                }
            }
            delta = warp_sum(delta);
            if (lane == 0) { shared.delta[row] = delta; }
        }
        for (int tile = 0; tile < tile_group; ++tile) {
            int64_t const col_begin = group_begin + int64_t(tile) * N;
            if (col_begin >= shared.valid_count) { break; }
            for (int col = threadIdx.x; col < N; col += Config::kMmaThreads) {
                shared.token[col] = col_begin + col < shared.valid_count
                    ? p.page_table[query * p.page_table_batch_stride + col_begin + col] : 0;
            }
            __syncthreads();
            for (int i = threadIdx.x; i < N * (kHeadDim / 8); i += Config::kMmaThreads) {
                int const col = i / (kHeadDim / 8), dim = (i % (kHeadDim / 8)) * 8;
                uint4 kval = make_uint4(0, 0, 0, 0);
                uint4 vval = make_uint4(0, 0, 0, 0);
                if (col_begin + col < shared.valid_count) {
                    int64_t const token = shared.token[col];
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
            {
                typename Config::MmaSdP mma;
                auto scores = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
                auto dp = partition_fragment_C(mma, Shape<Int<M>, Int<N>>{});
                clear(scores);
                clear(dp);
                shared_gemm<typename Config::Normal, typename Config::Normal>(mma, scores, sq, sk);
                shared_gemm<typename Config::Normal, typename Config::Normal>(mma, dp, sdo, sv);
                auto coords = mma.get_thread_slice(threadIdx.x).partition_C(
                    make_identity_tensor(Shape<Int<M>, Int<N>>{}));
#pragma unroll
                for (int i = 0; i < size(scores); ++i) {
                    int const row = get<0>(coords(i)), col = get<1>(coords(i));
                    float probability = 0.f, ds = 0.f;
                    if (row < group && col_begin + col < shared.valid_count
                        && shared.lse[row] != -CUDART_INF_F) {
                        if (shared.valid_count == 1 && isfinite(shared.lse[row])) {
                            probability = 1.f;
                        } else {
                            probability = expf(p.scale_softmax * scores(i) - shared.lse[row]);
                            ds = probability * (dp(i) - shared.delta[row]);
                        }
                    }
                    scores(i) = probability;
                    dp(i) = ds;
                }
                auto rp = make_tensor_like<T>(scores);
                auto rds = make_tensor_like<T>(dp);
                flash::convert_type_out(scores, rp);
                flash::convert_type_out(dp, rds);
                auto store = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>{}, mma);
                auto thr_store = store.get_thread_slice(threadIdx.x);
                cute::copy(store, thr_store.retile_S(rp), thr_store.partition_D(sp));
                cute::copy(store, thr_store.retile_S(rds), thr_store.partition_D(sds));
            }
            __syncthreads();
            for (int stripe = 0; stripe < kHeadDim / kStripeDim; ++stripe) {
                {
                    typename Config::MmaDQ mma;
                    auto acc = partition_fragment_C(mma, Shape<Int<M>, Int<kStripeDim>>{});
                    clear(acc);
                    auto kb = local_tile(skt, Shape<Int<kStripeDim>, Int<N>>{}, make_coord(stripe, _0{}));
                    shared_gemm<typename Config::Normal, typename Config::Transpose>(mma, acc, sds, kb);
                    auto coords = mma.get_thread_slice(threadIdx.x).partition_C(
                        make_identity_tensor(Shape<Int<M>, Int<kStripeDim>>{}));
#pragma unroll
                    for (int i = 0; i < size(acc); ++i) {
                        int const row = get<0>(coords(i)), dim = stripe * kStripeDim + get<1>(coords(i));
                        if ((dq_row_mask >> row) & 1u) {
                            int64_t const offset = (query * p.h + head_begin + row) * kHeadDim + dim;
                            atomicAdd(dq_accum + offset, p.scale_softmax * acc(i));
                        }
                    }
                }
                for (int gradient = 0; gradient < 2; ++gradient) {
                    typename Config::MmaDKV mma;
                    auto acc = partition_fragment_C(mma, Shape<Int<N>, Int<kStripeDim>>{});
                    clear(acc);
                    if (gradient == 0) {
                        auto qb = local_tile(sqt, Shape<Int<kStripeDim>, Int<M>>{}, make_coord(stripe, _0{}));
                        shared_gemm<typename Config::Transpose, typename Config::Transpose>(mma, acc, sdst, qb);
                    } else {
                        auto dob = local_tile(sdot, Shape<Int<kStripeDim>, Int<M>>{}, make_coord(stripe, _0{}));
                        shared_gemm<typename Config::Transpose, typename Config::Transpose>(mma, acc, spt, dob);
                    }
                    auto coords = mma.get_thread_slice(threadIdx.x).partition_C(
                        make_identity_tensor(Shape<Int<N>, Int<kStripeDim>>{}));
                    auto* accum = gradient == 0 ? dk_accum : dv_accum;
                    float const scale = gradient == 0 ? p.scale_softmax : 1.f;
                    auto scatter = [&](auto full_tile) {
                        if constexpr (M == 32) {
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
                                        int64_t const offset = (int64_t(shared.token[col]) * p.h_k + kv_head) * kHeadDim
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
                                int const col = get<0>(coords(i)), dim = stripe * kStripeDim + get<1>(coords(i));
                                if (decltype(full_tile)::value || col_begin + col < shared.valid_count) {
                                    int64_t const offset = (int64_t(shared.token[col]) * p.h_k + kv_head) * kHeadDim + dim;
                                    atomicAdd(accum + offset, scale * acc(i));
                                }
                            }
                        }
                    };
                    if (col_begin + N <= shared.valid_count) { scatter(std::true_type{}); }
                    else { scatter(std::false_type{}); }
                }
            }
            __syncthreads();
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

template <typename T, int Arch, int M, int N = kTileCols>
void launch_mma(Flash_bwd_params& p, hggcStream_t stream) {
    using Config = BackwardMma<T, Arch, M, N>;
    int64_t const tiles = (int64_t(p.seqlen_k) + N - 1) / N;
    int64_t const grouped_work = int64_t(p.total_q) * p.h_k * ((tiles + 3) / 4);
    int const tile_group = tiles >= 16 && grouped_work >= 1024 ? 4 : 1;
    int64_t const tile_groups = (tiles + tile_group - 1) / tile_group;
    int64_t const work_count = int64_t(p.total_q) * p.h_k * tile_groups;
    if (work_count == 0) { return; }
    auto kernel = accumulate_backward_mma<T, Arch, M, N>;
    constexpr int bytes = sizeof(typename Config::SharedStorage);
    CHECK_CUDA(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<launch_blocks(work_count, 1), Config::kMmaThreads, bytes, stream>>>(p, tile_groups, tile_group, work_count);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template <typename T, int Arch>
void launch_backward(Flash_bwd_params& p, hggcStream_t stream) {
    if (p.h / p.h_k <= 16) { launch_mma<T, Arch, 16>(p, stream); }
    else { launch_mma<T, Arch, 32>(p, stream); }
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

void validate_qsa_bwd(Flash_bwd_params const& p, int* error, hggcStream_t stream) {
    int64_t const table_count = int64_t(p.total_q) * p.seqlen_k;
    int64_t const sequence_count = int64_t(p.b) + 1;
    int64_t const work_count = table_count > sequence_count ? table_count : sequence_count;
    validate_inputs<<<launch_blocks(work_count), kThreads, 0, stream>>>(
        p, error, table_count, work_count);
    CHECK_CUDA_KERNEL_LAUNCH();
}

void run_qsa_bwd(Flash_bwd_params& p, hggcStream_t stream) {
    TORCH_CHECK(p.is_qsa && p.d == kHeadDim && p.dv == kHeadDim,
                "QSA backward requires QSA with Q/K/V head dimension 256");
    if (p.is_bf16) {
        if (p.arch >= 89) { launch_backward<cutlass::bfloat16_t, 89>(p, stream); }
        else { launch_backward<cutlass::bfloat16_t, 80>(p, stream); }
    } else {
#ifndef FLASHATTENTION_DISABLE_FP16
        if (p.arch >= 89) { launch_backward<cutlass::half_t, 89>(p, stream); }
        else { launch_backward<cutlass::half_t, 80>(p, stream); }
#else
        TORCH_CHECK(false, "QSA backward FP16 support is disabled");
#endif
    }
}

#endif
