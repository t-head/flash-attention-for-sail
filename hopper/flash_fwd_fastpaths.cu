// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// Selected forward fast paths: BF16 WI/WS and FP8 WI with V prepacking.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <algorithm>
#include <type_traits>
#include <cute/arch/mma_ppu0015.hpp>
#include <cute/arch/copy_ppu0015_aiu.hpp>
#include <hggc_runtime.h>

#include "cuda_check.h"
#include "flash_fwd_fastpaths.h"
#include "utils.h"
#include "seqlen.h"

#ifdef FLASHATTENTION_MHA_WI_WS
namespace flash::mha {
__device__ __forceinline__ int uniform(int x) {
    return int(__ppu_read_firstlane(unsigned(x)));
}

// Match the AIU swizzle to the Tensor Cell operand width. Both attention
// mainloops and the producer/consumer path use these same native operations.
template <class Element> struct MhaOps {
    static constexpr int Bits = cutlass::sizeof_bits<Element>::value;
    template <bool Trans, int Lbo, int Sbo, int Mode = (Bits == 8 ? 1 : 0)>
    __device__ __forceinline__ static void ldm(int *v, int address) {
        address = uniform(address);
        if constexpr (Trans)
            asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans "
                         "{%0,%1,%2,%3}, [%4], %5,%6,%7;"
                         : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                         : "l"(address), "r"(Lbo), "r"(Sbo), "n"(Mode));
        else
            asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 "
                         "{%0,%1,%2,%3}, [%4], %5,%6,%7;"
                         : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                         : "l"(address), "r"(Lbo), "r"(Sbo), "n"(Mode));
    }
    __device__ __forceinline__ static void mma(float (&c)[8], const int *a, const int *b) {
        using Mma = std::conditional_t<Bits == 8, cute::PPU0015_16x16x32_F32E4M3E4M3F32_TN,
                                       std::conditional_t<std::is_same_v<Element, cutlass::half_t>,
                                                          cute::PPU0015_16x16x16_F32F16F16F32_TN,
                                                          cute::PPU0015_16x16x16_F32BF16BF16F32_TN>>;
        Mma::fma(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], a[0], a[1], a[2], a[3], b[0], b[1], b[2],
                 b[3], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
    }
    template <int Rows, int Cols, int Cube = 64>
    __device__ __forceinline__ static void copy_tile(Element *dst, const Element *src, int64_t stride,
                                                     int start, int length, int tid, int count) {
        int warp = uniform(tid / 32);
        if ((tid & 31) == 0 && warp < Cols / Cube && length) {
            cute::AiuDesc desc{};
            desc.gmem_ptr = reinterpret_cast<const uint8_t *>(src);
            desc.dim_h = length;
            desc.dim_w = Cols;
            desc.stride_w = stride;
            cute::PPU0015_AIU_LOAD_IMPL<cute::Int<Rows * Cube * Bits>, Bits, Rows, Cube>{}(
                dst + warp * Rows * Cube, src, desc, warp * Cube, start, 0);
        }
        if (!length)
            for (int i = tid; i < Rows * Cols; i += count)
                dst[i] = Element(0.f);
    }
    __device__ __forceinline__ static int pack4(float a, float b, float c, float d) {
        static_assert(Bits == 8);
        cutlass::Array<float, 4> input;
        input[0] = a; input[1] = b; input[2] = c; input[3] = d;
        auto packed = cutlass::NumericArrayConverter<Element, float, 4>::convert(input);
        return cutlass::platform::bit_cast<int>(packed);
    }
    __device__ __forceinline__ static int pack(float a, float b) {
        static_assert(Bits == 16);
        Element lo(a), hi(b);
        return int(uint32_t(lo.raw()) | (uint32_t(hi.raw()) << 16));
    }

};
// Keep the same cube-major shared layout for contiguous and paged KV.
// A KV tile spans one or more page64 allocations, which need not be adjacent.
// Reuse the native copy atom for each page/cube; fences remain owned by gather.
template <bool Paged, int Rows, int Cols, typename Element>
__device__ __forceinline__ void copy_kv_tile(
    Element *dst, const Element *src, int64_t row_stride, int64_t page_stride,
    Flash_fwd_params const& p, int batch, int start, int length, int tid, int count) {
    if constexpr (!Paged) {
        MhaOps<Element>::template copy_tile<Rows, Cols>(
            dst, src, row_stride, start, length, tid, count);
    } else {
        constexpr int PageRows = 64, Cube = 64;
        static_assert(Rows % PageRows == 0 && Cols % Cube == 0);
        int cube = uniform(tid / 32);
        #pragma unroll
        for (int part = 0; part < Rows / PageRows; ++part) {
            int row = start + part * PageRows;
            int valid = max(0, min(PageRows, length - row));
            if (valid) {
                // Only the issuing lane needs a physical page address. Keeping
                // these pointers out of the other lanes avoids vector spills.
                if ((tid & 31) == 0 && cube < Cols / Cube) {
                    int physical = p.page_table[
                        int64_t(batch) * p.page_table_batch_stride + row / PageRows];
                    auto page = src + int64_t(physical) * page_stride + cube * Cube;
                    MhaOps<Element>::template copy_tile<PageRows, Cube>(
                        dst + cube * Rows * Cube + part * PageRows * Cube,
                        page, row_stride, 0, valid, 0, 32);
                }
            } else {
                for (int i = tid; i < PageRows * Cols; i += count) {
                    int block = i / (PageRows * Cube), offset = i % (PageRows * Cube);
                    dst[block * Rows * Cube + part * PageRows * Cube + offset] = Element(0.f);
                }
            }
        }
    }
}

// Bound the compact retry workspace by one resident WS wave. Subdivide KV
// only while the affected 32-row blocks still fit that wave. The private
// parameter copy carries the original WI partition count in num_splits.
constexpr int MhaWsRetryMaxSplits = 8;
__host__ __device__ __forceinline__ int mha_ws_retry_capacity(Flash_fwd_params const& p) {
    constexpr int MinKeyTiles = 128;
    constexpr int QuerySubtiles = detail::MhaWiQueryRows / detail::MhaWsQueryRows;
    if (p.seqlen_k < MinKeyTiles * detail::MhaWsKeyColumns) return 0;
    return detail::MhaWsResidentBlocks * p.num_sm / (QuerySubtiles * p.num_splits);
}

__device__ __forceinline__ int mha_ws_retry_splits(Flash_fwd_params const& p, int const* flags) {
    int const affected = uniform(flags[0]);
    if (affected <= 0 || affected > mha_ws_retry_capacity(p)) return 0;
    constexpr int QuerySubtiles = detail::MhaWiQueryRows / detail::MhaWsQueryRows;
    int splits = p.num_splits;
    while (splits < MhaWsRetryMaxSplits
           && affected * QuerySubtiles * (2 * splits) <= detail::MhaWsResidentBlocks * p.num_sm) splits *= 2;
    return splits;
}

struct alignas(8) MhaRetryTile {
    int batch;
    int query_head;
};

__host__ __device__ __forceinline__ int64_t mha_retry_queue_offset(int64_t tiles) {
    constexpr int Words = sizeof(MhaRetryTile) / sizeof(int);
    return ((1 + tiles + Words - 1) / Words) * Words;
}

__device__ __forceinline__ int64_t mha_wi_retry_tiles(Flash_fwd_params const& p) {
    return ((int64_t(p.seqlen_q) + detail::MhaWiQueryRows - 1) / detail::MhaWiQueryRows) * p.h * p.b;
}

__device__ __forceinline__ MhaRetryTile const* mha_retry_queue(Flash_fwd_params const& p, int const* flags) {
    return reinterpret_cast<MhaRetryTile const*>(flags + mha_retry_queue_offset(mha_wi_retry_tiles(p)));
}

// Compact a bounded worklist after WI finishes. This keeps queue management
// and its register/control pressure out of the normal attention mainloop.
__global__ void prepare_mha_wi_retry(int* flags, int64_t tiles, int query_heads, int capacity) {
    auto* queue = reinterpret_cast<MhaRetryTile*>(flags + mha_retry_queue_offset(tiles));
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < tiles; index += int64_t(gridDim.x) * blockDim.x) {
        if (!flags[1 + index]) continue;
        int slot = atomicAdd(flags, 0);
        // capacity+1 is a dense-retry sentinel. Saturation avoids counter
        // overflow and avoids writing beyond the compact workspace/queue.
        while (slot <= capacity) {
            int const previous = atomicCAS(flags, slot, slot + 1);
            if (previous == slot) break;
            slot = previous;
        }
        if (slot >= capacity) return;
        queue[slot] = {int(index / query_heads), int(index % query_heads)};
    }
}

}  // namespace flash::mha

// Warp interleaving (WI): staggered groups share a three-stage KV ring.
// Register-resident Q/P with four staggered groups and three AIU KV stages.
// Barrier ownership: the first group copies the next KV tile; staggered QK
// consumers publish completion before the three-stage KV ring can be reused.
// Guarded and peeled causal BF16 schedules share the native MhaOps atoms.
// Their distinct loop schedules are retained to avoid regressions on short KV
// and on the masked tail of long causal prefills.
namespace flash::wi {
using mha::copy_kv_tile;
using mha::uniform;
using B16 = cutlass::bfloat16_t;
using Ops = mha::MhaOps<B16>;
constexpr int BM = detail::MhaWiQueryRows, N = detail::MhaWiKeyColumns, D = 192, V = 128, NT = 512, WG = 128, Stages = 3;
struct alignas(128) Storage {
    union {
        B16 q[BM * D];
        B16 v[Stages][N * V];
    } qv;
    B16 k[Stages][N * D];
};
static_assert(sizeof(Storage) == 245760);

// Bound each tile's probabilities by one before BF16 packing and PV. A
// deferred maximum can leave P finite while P * V overflows FP32.
__device__ __forceinline__ void update_bf16_row_max(
    float const (&score)[N / 16][8], float (&acc)[8][8], float scale,
    float& mx0, float& mx1, float& sum0, float& sum1) {
    MaxOp<float> max_op;
    float a = -INFINITY, b = -INFINITY;
#pragma unroll
    for (int n = 0; n < N / 16; n++) {
        a = max_op(a, max_op(max_op(score[n][0], score[n][1]), max_op(score[n][4], score[n][5])));
        b = max_op(b, max_op(max_op(score[n][2], score[n][3]), max_op(score[n][6], score[n][7])));
    }
    if (!__any_sync(0xffffffff, a * scale > mx0 || b * scale > mx1)) return;
#pragma unroll
    for (int delta = 1; delta <= 2; delta *= 2) {
        a = max_op(a, __shfl_xor_sync(0xffffffff, a, delta));
        b = max_op(b, __shfl_xor_sync(0xffffffff, b, delta));
    }
    a *= scale;
    b *= scale;
    a = max_op(mx0, a);
    b = max_op(mx1, b);
    float alpha0 = exp2f(mx0 - a), alpha1 = exp2f(mx1 - b);
    mx0 = a;
    mx1 = b;
    sum0 *= alpha0;
    sum1 *= alpha1;
#pragma unroll
    for (int n = 0; n < 8; n++) {
#pragma unroll
        for (int e = 0; e < 8; e++)
            acc[n][e] *= ((e >> 1) & 1) ? alpha1 : alpha0;
    }
}

template <bool Paged = false, bool Varlen = true, bool Retry = false>
__global__ __launch_bounds__(NT) void mha_wi_bf16_guarded(Flash_fwd_params p, int* retry_flags) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    if constexpr (Retry) {
        if (mha::mha_ws_retry_splits(p, retry_flags)) return;
        // One word per query tile/head/batch; each bit owns one KV partition.
        int const flags = uniform(retry_flags[1 + int64_t(blockIdx.y) * gridDim.x + blockIdx.x]);
        if (!(flags & (1 << blockIdx.z))) return;
    }
    extern __shared__ __align__(128) unsigned char bytes[];
    auto &s = *reinterpret_cast<Storage *>(bytes);
    int tid = threadIdx.x, lane = tid & 31, w = uniform(tid / 32), group = uniform(tid / WG), gt = tid % WG;
    int blocks = (p.seqlen_q + BM - 1) / BM;
    int head = blockIdx.x % p.h, batch = blockIdx.y;
    int qstart = (blocks - 1 - blockIdx.x / p.h) * BM;
    SeqlenInfoQK<Varlen, BM> const info(batch, p.seqlen_q, p.seqlen_k,
        p.cu_seqlens_q, Paged ? nullptr : p.cu_seqlens_k,
        nullptr, Paged ? p.seqused_k : nullptr);
    int qo = info.offset_q, ko = info.offset_k;
    int qlen = uniform(info.seqlen_q), klen = uniform(info.seqlen_k);
    if (qstart >= qlen)
        return;
    const B16 *q = static_cast<const B16 *>(p.q_ptr) + int64_t(qo) * p.q_row_stride + head * p.q_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.q_batch_stride);
    const B16 *k = static_cast<const B16 *>(p.k_ptr) + int64_t(ko) * p.k_row_stride + head * p.k_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.k_batch_stride);
    const B16 *v = static_cast<const B16 *>(p.v_ptr) + int64_t(ko) * p.v_row_stride + head * p.v_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.v_batch_stride);
    B16 *o = static_cast<B16 *>(p.o_ptr) + int64_t(qo) * p.o_row_stride + head * p.o_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.o_batch_stride);
    float *lse = static_cast<float *>(p.softmax_lse_ptr)
        + (Varlen ? int64_t(head) * p.total_q + qo
                  : (int64_t(batch) * p.h + head) * p.seqlen_q);
    Ops::copy_tile<BM, D>(s.qv.q, q, p.q_row_stride, qstart, qlen, tid, NT);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
    int rq[12][4], qt = int(reinterpret_cast<uintptr_t>(s.qv.q) / 16) + w * 128;
#pragma unroll
    for (int d = 0; d < 12; d++)
        Ops::ldm<false, 1, 64>(rq[d], qt + (d / 4) * BM * 8 + (d % 4) * 2);
    // Q is now resident in every warp; V may reuse its shared memory.
    __syncthreads();
    int last = klen;
    last = min(klen, max(0, qstart + BM + klen - qlen));
    int tiles = max(1, (last + N - 1) / N);
    auto gather = [&](int t) {
        copy_kv_tile<Paged, N, D>(s.k[t % Stages], k, p.k_row_stride, p.k_batch_stride,
                                  p, batch, t * N, klen, gt, WG);
        copy_kv_tile<Paged, N, V>(s.qv.v[t % Stages], v, p.v_row_stride, p.v_batch_stride,
                                  p, batch, t * N, klen, gt, WG);
        cute::cp_async_fence();
    };
    if (group == 0) {
        gather(0);
        cute::cp_async_wait<0>();
        __ppu_barrier_sync(13, WG, 15u);
    }

    float mx0 = 0, mx1 = 0, sum0 = 0, sum1 = 0, acc[8][8] = {};
    float scale = p.scale_softmax * 1.4426950408889634f;
    for (int t = 0; t < tiles; t++) {
        int slot = t % Stages;
        if (group == 0) {
            if (t >= 2)
                __ppu_barrier_sync(10 + (t - 2) % Stages, NT, 15u);
            if (t + 1 < tiles)
                gather(t + 1);
        } else {
            __ppu_barrier_sync(1 + (group - 1) * Stages + slot, 2 * WG, 15u);
        }
        int kt = int(reinterpret_cast<uintptr_t>(s.k[slot]) / 16);
        float score[N / 16][8] = {};
#pragma hggc mmatiestrictly
        {
#pragma unroll
            for (int d = 0; d < 12; d++) {
                {
#pragma unroll
                    for (int n = 0; n < N / 16; n++) {
                        int rk[4];
                        Ops::ldm<false, 64, 1>(rk, kt + n * 128 + (d / 4) * N * 8 + (d % 4) * 2);
                        Ops::mma(score[n], rq[d], rk);
                    }
                }
            }
        }
        if (group < NT / WG - 1) {
            __syncwarp();
            __ppu_sched_bound();
            __ppu_barrier_arrive(1 + group * Stages + slot, 2 * WG, 15u);
        }
        bool masked = (t + 1) * N > klen;
        masked = masked || (t + 1) * N > qstart + group * 64 + klen - qlen + 1;
        if (uniform(int(masked))) {
#pragma unroll
            for (int n = 0; n < N / 16; n++) {
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    int r = w * 16 + lane / 4 + ((e >> 1) & 1) * 8;
                    int c = t * N + n * 16 + (lane & 3) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
                    bool valid = c < klen;
                    valid = valid && c <= qstart + r + klen - qlen;
                    if (!valid)
                        score[n][e] = -INFINITY;
                }
            }
        }
        if (t == 0) {
            float a = -INFINITY, b = -INFINITY;
#pragma unroll
            for (int n = 0; n < N / 16; n++) {
                a = fmaxf(a, fmaxf(fmaxf(score[n][0], score[n][1]), fmaxf(score[n][4], score[n][5])));
                b = fmaxf(b, fmaxf(fmaxf(score[n][2], score[n][3]), fmaxf(score[n][6], score[n][7])));
            }
#pragma unroll
            for (int delta = 1; delta <= 2; delta *= 2) {
                a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, delta));
                b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, delta));
            }
            a *= scale;
            b *= scale;
            {
                mx0 = a == -INFINITY ? 0.f : a;
                mx1 = b == -INFINITY ? 0.f : b;
            }
        }
        else {
            update_bf16_row_max(score, acc, scale, mx0, mx1, sum0, sum1);
        }
        int rp[N / 16][4];
        float z0 = 0, z1 = 0;
#pragma unroll
        for (int n = 0; n < N / 16; n++) {
#pragma unroll
            for (int e = 0; e < 8; e++) {
                float z = exp2f(fmaf(score[n][e], scale, -(((e >> 1) & 1) ? mx1 : mx0)));
                score[n][e] = z;
            }
            z0 += (score[n][0] + score[n][1]) + (score[n][4] + score[n][5]);
            z1 += (score[n][2] + score[n][3]) + (score[n][6] + score[n][7]);
#pragma unroll
            for (int e = 0; e < 4; e++)
                rp[n][e] = Ops::pack(score[n][e * 2], score[n][e * 2 + 1]);
        }
        sum0 += z0;
        sum1 += z1;
        int vt = int(reinterpret_cast<uintptr_t>(s.qv.v[slot]) / 16);
#pragma hggc mmatiestrictly
        {
#pragma unroll
            for (int d = 0; d < N / 16; d++) {
                {
#pragma unroll
                    for (int n = 0; n < 8; n++) {
                        int rv[4];
                        Ops::ldm<true, 1, 64>(rv, vt + (n / 4) * N * 8 + (n % 4) * 2 + d * 128);
                        Ops::mma(acc[n], rp[d], rv);
                    }
                }
            }
        }
        {
            if (group > 0) {
                __syncwarp();
                __ppu_sched_bound();
                __ppu_barrier_arrive(10 + slot, NT, 15u);
            }
        }
        if (group == 0 && t + 1 < tiles) {
            cute::cp_async_wait<0>();
            __ppu_barrier_sync(13, WG, 15u);
        }

    }
    if (group == 0) {
        if (tiles >= 2)
            __ppu_barrier_sync(10 + (tiles - 2) % Stages, NT, 15u);
        __ppu_barrier_sync(10 + (tiles - 1) % Stages, NT, 15u);
    }
#pragma unroll
    for (int delta = 1; delta <= 2; delta *= 2) {
        sum0 += __shfl_xor_sync(0xffffffff, sum0, delta);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, delta);
    }
    float inv0 = sum0 > 0 ? __ppu_rcpf(sum0) : 0.f, inv1 = sum1 > 0 ? __ppu_rcpf(sum1) : 0.f;
#pragma unroll
    for (int n = 0; n < 8; n++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int r = w * 16 + lane / 4 + ((e >> 1) & 1) * 8;
            int c = n * 16 + (lane & 3) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
            if (qstart + r < qlen) {
                o[int64_t(qstart + r) * p.o_row_stride + c] = B16(acc[n][e] * (((e >> 1) & 1) ? inv1 : inv0));
                if (n == 0 && (lane & 3) == 0 && e < 4 && (e & 1) == 0) {
                    float sum = ((e >> 1) & 1) ? sum1 : sum0;
                    lse[qstart + r] = sum > 0
                                          ? ((((e >> 1) & 1) ? mx1 : mx0) + log2f(sum)) * 0.6931471805599453f
                                          : (p.seqlen_k == 0 ? INFINITY : -INFINITY);
                }
            }
        }
    }
#endif
}

template <bool Paged = false, bool Varlen = true, int Splits = 1, bool Retry = false>
__global__ __launch_bounds__(NT) void mha_wi_bf16_peeled(Flash_fwd_params p, int* retry_flags) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    if constexpr (Retry) {
        if (mha::mha_ws_retry_splits(p, retry_flags)) return;
        // One word per query tile/head/batch; each bit owns one KV partition.
        int const flags = uniform(retry_flags[1 + int64_t(blockIdx.y) * gridDim.x + blockIdx.x]);
        if (!(flags & (1 << blockIdx.z))) return;
    }
    extern __shared__ __align__(128) unsigned char bytes[];
    auto &s = *reinterpret_cast<Storage *>(bytes);
    int tid = threadIdx.x, lane = tid & 31, w = uniform(tid / 32), group = uniform(tid / WG), gt = tid % WG;
    int blocks = (p.seqlen_q + BM - 1) / BM;
    int head = blockIdx.x % p.h, batch = blockIdx.y;
    int qstart = (blocks - 1 - blockIdx.x / p.h) * BM;
    SeqlenInfoQK<Varlen, BM> const info(batch, p.seqlen_q, p.seqlen_k,
        p.cu_seqlens_q, Paged ? nullptr : p.cu_seqlens_k,
        nullptr, Paged ? p.seqused_k : nullptr);
    int qo = info.offset_q, ko = info.offset_k;
    int qlen = uniform(info.seqlen_q), klen = uniform(info.seqlen_k);
    if (qstart >= qlen)
        return;
    const B16 *q = static_cast<const B16 *>(p.q_ptr) + int64_t(qo) * p.q_row_stride + head * p.q_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.q_batch_stride);
    const B16 *k = static_cast<const B16 *>(p.k_ptr) + int64_t(ko) * p.k_row_stride + head * p.k_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.k_batch_stride);
    const B16 *v = static_cast<const B16 *>(p.v_ptr) + int64_t(ko) * p.v_row_stride + head * p.v_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.v_batch_stride);
    B16 *o = static_cast<B16 *>(p.o_ptr) + int64_t(qo) * p.o_row_stride + head * p.o_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.o_batch_stride);
    float *lse = static_cast<float *>(p.softmax_lse_ptr)
        + (Varlen ? int64_t(head) * p.total_q + qo
                  : (int64_t(batch) * p.h + head) * p.seqlen_q);
    float *partial = nullptr, *partial_lse = nullptr;
    if constexpr (Splits > 1) {
        int64_t offset = (int64_t(blockIdx.z) * p.h + head) * p.total_q
            + (Varlen ? qo : int64_t(batch) * p.seqlen_q);
        partial = static_cast<float *>(p.oaccum_ptr) + offset * V;
        partial_lse = static_cast<float *>(p.softmax_lseaccum_ptr) + offset;
    }
    Ops::copy_tile<BM, D>(s.qv.q, q, p.q_row_stride, qstart, qlen, tid, NT);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
    int rq[12][4], qt = int(reinterpret_cast<uintptr_t>(s.qv.q) / 16) + w * 128;
#pragma unroll
    for (int d = 0; d < 12; d++)
        Ops::ldm<false, 1, 64>(rq[d], qt + (d / 4) * BM * 8 + (d % 4) * 2);
    // Q is now resident in every warp; V may reuse its shared memory.
    __syncthreads();
    int last = klen;
    last = min(klen, max(0, qstart + BM + klen - qlen));
    int tiles = max(1, (last + N - 1) / N);
    int first = 0;
    if constexpr (Splits > 1) {
        int chunk = (tiles + Splits - 1) / Splits;
        first = min(int(blockIdx.z) * chunk, tiles);
        tiles = max(1, min(chunk, tiles - first));
    }
    auto gather = [&](int t) {
        copy_kv_tile<Paged, N, D>(s.k[t % Stages], k, p.k_row_stride, p.k_batch_stride,
                                  p, batch, (first + t) * N, klen, gt, WG);
        copy_kv_tile<Paged, N, V>(s.qv.v[t % Stages], v, p.v_row_stride, p.v_batch_stride,
                                  p, batch, (first + t) * N, klen, gt, WG);
        cute::cp_async_fence();
    };
    if (group == 0) {
        gather(0);
        cute::cp_async_wait<0>();
        __ppu_barrier_sync(13, WG, 15u);
    }

    float mx0 = 0, mx1 = 0, sum0 = 0, sum1 = 0, acc[8][8] = {};
    float scale = p.scale_softmax * 1.4426950408889634f;
    {
        const int t = 0;
        int slot = t % Stages;
        if (group == 0) {
            if (t + 1 < tiles)
                gather(t + 1);
        } else {
            __ppu_barrier_sync(1 + (group - 1) * Stages + slot, 2 * WG, 15u);
        }
        int kt = int(reinterpret_cast<uintptr_t>(s.k[slot]) / 16);
        float score[N / 16][8] = {};
#pragma hggc mmatiestrictly
        {
#pragma unroll
            for (int d = 0; d < 12; d++) {
                {
#pragma unroll
                    for (int n = 0; n < N / 16; n++) {
                        int rk[4];
                        Ops::ldm<false, 64, 1>(rk, kt + n * 128 + (d / 4) * N * 8 + (d % 4) * 2);
                        Ops::mma(score[n], rq[d], rk);
                    }
                }
            }
        }
        if (group < NT / WG - 1) {
            __syncwarp();
            __ppu_sched_bound();
            __ppu_barrier_arrive(1 + group * Stages + slot, 2 * WG, 15u);
        }
        bool masked = (first + t + 1) * N > klen;
        masked = masked || (first + t + 1) * N > qstart + group * 64 + klen - qlen + 1;
        if (uniform(int(masked))) {
#pragma unroll
            for (int n = 0; n < N / 16; n++) {
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    int r = w * 16 + lane / 4 + ((e >> 1) & 1) * 8;
                    int c = (first + t) * N + n * 16 + (lane & 3) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
                    bool valid = c < klen;
                    valid = valid && c <= qstart + r + klen - qlen;
                    if (!valid)
                        score[n][e] = -INFINITY;
                }
            }
        }
        {
            float a = -INFINITY, b = -INFINITY;
#pragma unroll
            for (int n = 0; n < N / 16; n++) {
                a = fmaxf(a, fmaxf(fmaxf(score[n][0], score[n][1]), fmaxf(score[n][4], score[n][5])));
                b = fmaxf(b, fmaxf(fmaxf(score[n][2], score[n][3]), fmaxf(score[n][6], score[n][7])));
            }
#pragma unroll
            for (int delta = 1; delta <= 2; delta *= 2) {
                a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, delta));
                b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, delta));
            }
            a *= scale;
            b *= scale;
            {
                mx0 = a == -INFINITY ? 0.f : a;
                mx1 = b == -INFINITY ? 0.f : b;
            }
        }
        int rp[N / 16][4];
        float z0 = 0, z1 = 0;
#pragma unroll
        for (int n = 0; n < N / 16; n++) {
#pragma unroll
            for (int e = 0; e < 8; e++) {
                float z = exp2f(fmaf(score[n][e], scale, -(((e >> 1) & 1) ? mx1 : mx0)));
                score[n][e] = z;
            }
            z0 += (score[n][0] + score[n][1]) + (score[n][4] + score[n][5]);
            z1 += (score[n][2] + score[n][3]) + (score[n][6] + score[n][7]);
#pragma unroll
            for (int e = 0; e < 4; e++)
                rp[n][e] = Ops::pack(score[n][e * 2], score[n][e * 2 + 1]);
        }
        sum0 += z0;
        sum1 += z1;
        int vt = int(reinterpret_cast<uintptr_t>(s.qv.v[slot]) / 16);
#pragma hggc mmatiestrictly
        {
#pragma unroll
            for (int d = 0; d < N / 16; d++) {
                {
#pragma unroll
                    for (int n = 0; n < 8; n++) {
                        int rv[4];
                        Ops::ldm<true, 1, 64>(rv, vt + (n / 4) * N * 8 + (n % 4) * 2 + d * 128);
                        Ops::mma(acc[n], rp[d], rv);
                    }
                }
            }
        }
        {
            if (group > 0) {
                __syncwarp();
                __ppu_sched_bound();
                __ppu_barrier_arrive(10 + slot, NT, 15u);
            }
        }
        if (group == 0 && t + 1 < tiles) {
            cute::cp_async_wait<0>();
            __ppu_barrier_sync(13, WG, 15u);
        }

    }
    for (int t = 1; t < tiles; t++) {
        int slot = t % Stages;
        if (group == 0) {
            if (t >= 2)
                __ppu_barrier_sync(10 + (t - 2) % Stages, NT, 15u);
            if (t + 1 < tiles)
                gather(t + 1);
        } else {
            __ppu_barrier_sync(1 + (group - 1) * Stages + slot, 2 * WG, 15u);
        }
        int kt = int(reinterpret_cast<uintptr_t>(s.k[slot]) / 16);
        float score[N / 16][8] = {};
#pragma hggc mmatiestrictly
        {
#pragma unroll
            for (int d = 0; d < 12; d++) {
                {
#pragma unroll
                    for (int n = 0; n < N / 16; n++) {
                        int rk[4];
                        Ops::ldm<false, 64, 1>(rk, kt + n * 128 + (d / 4) * N * 8 + (d % 4) * 2);
                        Ops::mma(score[n], rq[d], rk);
                    }
                }
            }
        }
        if (group < NT / WG - 1) {
            __syncwarp();
            __ppu_sched_bound();
            __ppu_barrier_arrive(1 + group * Stages + slot, 2 * WG, 15u);
        }
        bool masked = (first + t + 1) * N > klen;
        masked = masked || (first + t + 1) * N > qstart + group * 64 + klen - qlen + 1;
        if (uniform(int(masked))) {
#pragma unroll
            for (int n = 0; n < N / 16; n++) {
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    int r = w * 16 + lane / 4 + ((e >> 1) & 1) * 8;
                    int c = (first + t) * N + n * 16 + (lane & 3) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
                    bool valid = c < klen;
                    valid = valid && c <= qstart + r + klen - qlen;
                    if (!valid)
                        score[n][e] = -INFINITY;
                }
            }
        }
        int vt = int(reinterpret_cast<uintptr_t>(s.qv.v[slot]) / 16);
        int rv_first[4];
        Ops::ldm<true, 1, 64>(rv_first, vt);
        update_bf16_row_max(score, acc, scale, mx0, mx1, sum0, sum1);
        int rp[N / 16][4];
        float z0 = 0, z1 = 0;
#pragma unroll
        for (int n = 0; n < N / 16; n++) {
#pragma unroll
            for (int e = 0; e < 8; e++) {
                float z = exp2f(fmaf(score[n][e], scale, -(((e >> 1) & 1) ? mx1 : mx0)));
                score[n][e] = z;
            }
            z0 += (score[n][0] + score[n][1]) + (score[n][4] + score[n][5]);
            z1 += (score[n][2] + score[n][3]) + (score[n][6] + score[n][7]);
#pragma unroll
            for (int e = 0; e < 4; e++)
                rp[n][e] = Ops::pack(score[n][e * 2], score[n][e * 2 + 1]);
        }
        sum0 += z0;
        sum1 += z1;
#pragma hggc mmatiestrictly
        {
#pragma unroll
            for (int d = 0; d < N / 16; d++) {
                {
#pragma unroll
                    for (int n = 0; n < 8; n++) {
                        if (d == 0 && n == 0) {
                            Ops::mma(acc[n], rp[d], rv_first);
                        } else {
                            int rv[4];
                            Ops::ldm<true, 1, 64>(rv, vt + (n / 4) * N * 8 + (n % 4) * 2 + d * 128);
                            Ops::mma(acc[n], rp[d], rv);
                        }
                    }
                }
            }
        }
        {
            if (group > 0) {
                __syncwarp();
                __ppu_sched_bound();
                __ppu_barrier_arrive(10 + slot, NT, 15u);
            }
        }
        if (group == 0 && t + 1 < tiles) {
            cute::cp_async_wait<0>();
            __ppu_barrier_sync(13, WG, 15u);
        }

    }
    if (group == 0) {
        if (tiles >= 2)
            __ppu_barrier_sync(10 + (tiles - 2) % Stages, NT, 15u);
        __ppu_barrier_sync(10 + (tiles - 1) % Stages, NT, 15u);
    }
#pragma unroll
    for (int delta = 1; delta <= 2; delta *= 2) {
        sum0 += __shfl_xor_sync(0xffffffff, sum0, delta);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, delta);
    }
    float inv0 = sum0 > 0 ? __ppu_rcpf(sum0) : 0.f, inv1 = sum1 > 0 ? __ppu_rcpf(sum1) : 0.f;
#pragma unroll
    for (int n = 0; n < 8; n++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int r = w * 16 + lane / 4 + ((e >> 1) & 1) * 8;
            int c = n * 16 + (lane & 3) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
            if (qstart + r < qlen) {
                if constexpr (Splits > 1) {
                    partial[int64_t(qstart + r) * V + c] = acc[n][e] * (((e >> 1) & 1) ? inv1 : inv0);
                    if (n == 0 && (lane & 3) == 0 && e < 4 && (e & 1) == 0) {
                        float sum = ((e >> 1) & 1) ? sum1 : sum0;
                        partial_lse[qstart + r] = sum > 0
                            ? ((((e >> 1) & 1) ? mx1 : mx0) + log2f(sum)) * 0.6931471805599453f
                            : -INFINITY;
                    }
                } else {
                    o[int64_t(qstart + r) * p.o_row_stride + c] = B16(acc[n][e] * (((e >> 1) & 1) ? inv1 : inv0));
                    if (n == 0 && (lane & 3) == 0 && e < 4 && (e & 1) == 0) {
                        float sum = ((e >> 1) & 1) ? sum1 : sum0;
                        lse[qstart + r] = sum > 0
                                              ? ((((e >> 1) & 1) ? mx1 : mx0) + log2f(sum)) * 0.6931471805599453f
                                              : (p.seqlen_k == 0 ? INFINITY : -INFINITY);
                    }
                }
            }
        }
    }
#endif
}

template <bool Paged = false, bool Varlen = true, int Splits = 1, bool Retry = false>
inline hggcError_t launch_mha_wi(Flash_fwd_params &p, hggcStream_t stream, int* retry_flags) {
    auto k = mha_wi_bf16_peeled<Paged, Varlen, Splits, Retry>;
    if constexpr (Splits == 1) {
        if (p.seqlen_k < 32768) k = mha_wi_bf16_guarded<Paged, Varlen, Retry>;
    }
    auto status = hggcFuncSetAttribute(k, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage));
    if (status != hggcSuccess)
        return status;
    k<<<dim3(((p.seqlen_q + BM - 1) / BM) * p.h, p.b, Splits), NT, sizeof(Storage), stream>>>(p, retry_flags);
    return hggcGetLastError();
}
} // namespace flash::wi


#if !defined(FLASHATTENTION_DISABLE_FP8)
#include "softmax.h"

namespace flash::wi {

using Fp8 = cutlass::float_e4m3_t;
using Fp8Ops = mha::MhaOps<Fp8>;

template <int HeadDim, int ValueDim, int BlockN> struct alignas(128) Fp8MhaStorage {
    static constexpr int BlockM = 256;
    static constexpr int Threads = 512;
    static constexpr int GroupThreads = 128;
    static constexpr int Stages = 3;
    Fp8 value[Stages][ValueDim * BlockN];
    Fp8 key[Stages][BlockN * HeadDim];
    Fp8 query[BlockM * HeadDim];
};

template <int HeadDim, int ValueDim, int BlockN, bool Paged, bool Varlen>
__global__ __launch_bounds__(512)
void mha_wi_fp8(Flash_fwd_params p, const Fp8 *packed, float scale_log2_host) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    static_assert(HeadDim == 192 && ValueDim == 128 && BlockN == 128);
    using Storage = Fp8MhaStorage<HeadDim, ValueDim, BlockN>;
    constexpr int BM = Storage::BlockM, NT = Storage::Threads;
    constexpr int WG = Storage::GroupThreads, Stages = Storage::Stages, QSpan = 64;
    // 1..9 hand K ownership from group g to g+1. 10..12 reclaim KV stages
    // after all groups finish PV; 13 makes group 0's async copies visible.
    constexpr int ReadyBarrier = 1, ReclaimBarrier = 10, CopyBarrier = 13;
    extern __shared__ __align__(128) unsigned char bytes[];
    auto &s = *reinterpret_cast<Storage *>(bytes);
    int tid = threadIdx.x, lane = tid & 31, warp = uniform(tid / 32);
    int group = uniform(tid / WG), group_tid = tid % WG;
    int blocks = (p.seqlen_q + BM - 1) / BM, head = blockIdx.x % p.h, batch = blockIdx.y;
    int qstart = (blocks - 1 - blockIdx.x / p.h) * BM;
    SeqlenInfoQK<Varlen, BM> const info(batch, p.seqlen_q, p.seqlen_k,
        p.cu_seqlens_q, Paged ? nullptr : p.cu_seqlens_k,
        nullptr, Paged ? p.seqused_k : nullptr);
    int qo = info.offset_q, ko = info.offset_k;
    int qlen = uniform(info.seqlen_q), klen = uniform(info.seqlen_k);
    int kvhead = head / (p.h / p.h_k);
    if (qstart >= qlen) return;
    const Fp8 *q = static_cast<const Fp8 *>(p.q_ptr) + int64_t(qo) * p.q_row_stride + head * p.q_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.q_batch_stride);
    const Fp8 *k = static_cast<const Fp8 *>(p.k_ptr) + int64_t(ko) * p.k_row_stride + kvhead * p.k_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.k_batch_stride);
    B16 *o = static_cast<B16 *>(p.o_ptr) + int64_t(qo) * p.o_row_stride + head * p.o_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.o_batch_stride);
    float *lse = static_cast<float *>(p.softmax_lse_ptr)
        + (Varlen ? int64_t(head) * p.total_q + qo
                  : (int64_t(batch) * p.h + head) * p.seqlen_q);
    Fp8Ops::copy_tile<BM, HeadDim, QSpan>(s.query, q, p.q_row_stride, qstart, qlen, tid, NT);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
    int last = min(klen, max(0, qstart + BM + klen - qlen));
    int tiles = max(1, (last + BlockN - 1) / BlockN), pages = (p.seqlen_k + 63) / 64;
    auto gather = [&](int step) {
        int key_tile = tiles - 1 - step;
        copy_kv_tile<Paged, BlockN, HeadDim>(
            s.key[step % Stages], k, p.k_row_stride, p.k_batch_stride,
            p, batch, key_tile * BlockN, klen, group_tid, WG);
        #pragma unroll
        for (int j = 0; j < BlockN / 64; ++j) {
            bool valid = key_tile * BlockN + j * 64 < klen;
            int page = valid ? key_tile * (BlockN / 64) + j : 0;
            const Fp8 *v = packed + ((int64_t(batch) * pages + page) * p.h_k + kvhead) * ValueDim * 64;
            Fp8Ops::copy_tile<ValueDim, 64, 64>(
                s.value[step % Stages] + j * ValueDim * 64, v, 64, 0, valid ? ValueDim : 0, group_tid, WG);
        }
        cute::cp_async_fence();
    };
    if (group == 0) {
        gather(0);
        cute::cp_async_wait<0>();
        __ppu_barrier_sync(CopyBarrier, WG, 15u);
    }
    float qd = p.q_descale_ptr
        ? p.q_descale_ptr[batch * p.q_descale_batch_stride + kvhead * p.q_descale_head_stride] : 1.f;
    float kd = p.k_descale_ptr
        ? p.k_descale_ptr[batch * p.k_descale_batch_stride + kvhead * p.k_descale_head_stride] : 1.f;
    float vd = p.v_descale_ptr
        ? p.v_descale_ptr[batch * p.v_descale_batch_stride + kvhead * p.v_descale_head_stride] : 1.f;
    float scale = scale_log2_host * (qd * kd);
    scale = __uint_as_float(__ppu_read_firstlane(__float_as_uint(scale)));
    vd = __uint_as_float(__ppu_read_firstlane(__float_as_uint(vd)));
    int scale_mode = scale > 0.f ? 0 : (scale == 0.f ? 2 : 1);
    if (scale_mode) scale = scale_mode == 2 ? 1.f : -scale;
    float acc[ValueDim / 16][8] = {};
    flash::Softmax<2, 8> softmax(scale);
    using OutputLayout = cute::Layout<
        cute::Shape<cute::Shape<cute::_2, cute::_2, cute::_2>, cute::_1, cute::Int<ValueDim / 16>>,
        cute::Stride<cute::Stride<cute::_1, cute::_2, cute::_4>, cute::_0, cute::_8>>;
    auto output = cute::make_tensor(cute::make_rmem_ptr(&acc[0][0]), OutputLayout{});
    for (int step = 0; step < tiles; ++step) {
        int slot = step % Stages, key_tile = tiles - 1 - step;
        if (group == 0) {
            // Three stages keep the current tile and one prefetched tile live.
            // Refill only after every group releases the overwritten stage.
            if (step >= 2) __ppu_barrier_sync(ReclaimBarrier + (step - 2) % Stages, NT, 15u);
            if (step + 1 < tiles) gather(step + 1);
        } else {
            __ppu_barrier_sync(ReadyBarrier + (group - 1) * Stages + slot, 2 * WG, 15u);
        }
        int kt = int(reinterpret_cast<uintptr_t>(s.key[slot]) / 16);
        float score[BlockN / 16][8] = {};
        int qt = int(reinterpret_cast<uintptr_t>(s.query) / 16) + warp * QSpan;
        #pragma unroll
        for (int di = 0; di < HeadDim / 32; ++di) {
            int qa[4];
            Fp8Ops::ldm<false, 1, QSpan / 2, 1>(
                qa, qt + (di / (QSpan / 32)) * BM * (QSpan / 16) + (di % (QSpan / 32)) * 2);
            int rk[2][4];
            Fp8Ops::ldm<false, QSpan / 2, 1, 1>(
                rk[0], kt + (di / (QSpan / 32)) * BlockN * (QSpan / 16) + (di % (QSpan / 32)) * 2);
            #pragma unroll
            for (int n = 0; n < BlockN / 16; ++n) {
                if (n + 1 < BlockN / 16)
                    Fp8Ops::ldm<false, QSpan / 2, 1, 1>(rk[(n + 1) & 1],
                        kt + (n + 1) * QSpan + (di / (QSpan / 32)) * BlockN * (QSpan / 16) + (di % (QSpan / 32)) * 2);
                Fp8Ops::mma(score[n], qa, rk[n & 1]);
            }
        }
        // Release K after QK, allowing the next group to overlap QK with this
        // group's softmax/PV. KV storage remains live until the PV release.
        if (group < NT / WG - 1) {
            __syncwarp();
            __ppu_sched_bound();
            __ppu_barrier_arrive(ReadyBarrier + group * Stages + slot, 2 * WG, 15u);
        }
        if (scale_mode) {
            #pragma unroll
            for (int n = 0; n < BlockN / 16; ++n) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) score[n][e] = scale_mode == 2 ? 0.f : -score[n][e];
            }
        }
        if (uniform(int((key_tile + 1) * BlockN > klen ||
                        (key_tile + 1) * BlockN > qstart + group * 64 + klen - qlen + 1))) {
            #pragma unroll
            for (int n = 0; n < BlockN / 16; ++n) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    int row = warp * 16 + lane / 4 + 8 * ((e >> 1) & 1);
                    int col = key_tile * BlockN + n * 16 + 2 * (lane & 3) + (e & 1) + 8 * (e >> 2);
                    if (col >= klen || col > qstart + row + klen - qlen) score[n][e] = -INFINITY;
                }
            }
        }
        using ScoreLayout = cute::Layout<
            cute::Shape<cute::Shape<cute::_2, cute::_2, cute::_2>, cute::_1, cute::Int<BlockN / 16>>,
            cute::Stride<cute::Stride<cute::_1, cute::_2, cute::_4>, cute::_0, cute::_8>>;
        auto scores = cute::make_tensor(cute::make_rmem_ptr(&score[0][0]), ScoreLayout{});
        auto update = [&](auto first) {
            constexpr bool First = decltype(first)::value;
            auto scales = softmax.template max_get_scale<First, true>(scores);
            softmax.template online_softmax<First, true>(scores);
            if constexpr (!First) softmax.rescale_o(output, scales);
        };
        if (step == 0) update(cute::true_type{});
        else update(cute::false_type{});
        int vt = int(reinterpret_cast<uintptr_t>(s.value[slot]) / 16);
        #pragma unroll
        for (int mk = 0; mk < BlockN / 32; ++mk) {
            int rp[4] = {
                Fp8Ops::pack4(score[2*mk][0], score[2*mk][1], score[2*mk][4], score[2*mk][5]),
                Fp8Ops::pack4(score[2*mk][2], score[2*mk][3], score[2*mk][6], score[2*mk][7]),
                Fp8Ops::pack4(score[2*mk+1][0], score[2*mk+1][1], score[2*mk+1][4], score[2*mk+1][5]),
                Fp8Ops::pack4(score[2*mk+1][2], score[2*mk+1][3], score[2*mk+1][6], score[2*mk+1][7])};
            int rv[2][4];
            Fp8Ops::ldm<false, 32, 1, 1>(rv[0], vt + (mk / 2) * ValueDim * 4 + (mk % 2) * 2);
            #pragma unroll
            for (int n = 0; n < ValueDim / 16; ++n) {
                if (n + 1 < ValueDim / 16)
                    Fp8Ops::ldm<false, 32, 1, 1>(
                        rv[(n + 1) & 1], vt + (n + 1) * 64 + (mk / 2) * ValueDim * 4 + (mk % 2) * 2);
                Fp8Ops::mma(acc[n], rp, rv[n & 1]);
            }
        }
        if (group > 0) {
            __syncwarp();
            __ppu_sched_bound();
            __ppu_barrier_arrive(ReclaimBarrier + slot, NT, 15u);
        }
        if (group == 0 && step + 1 < tiles) {
            cute::cp_async_wait<0>();
            __ppu_barrier_sync(CopyBarrier, WG, 15u);
        }
    }
    if (group == 0) {
        if (tiles >= 2) __ppu_barrier_sync(ReclaimBarrier + (tiles - 2) % Stages, NT, 15u);
        __ppu_barrier_sync(ReclaimBarrier + (tiles - 1) % Stages, NT, 15u);
    }
    auto final_scales = softmax.finalize(vd);
    softmax.rescale_o(output, final_scales);
    #pragma unroll
    for (int n = 0; n < ValueDim / 16; ++n) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            int row = warp * 16 + lane / 4 + 8 * ((e >> 1) & 1);
            int col = n * 16 + 2 * (lane & 3) + (e & 1) + 8 * (e >> 2);
            if (qstart + row < qlen) {
                o[int64_t(qstart + row) * p.o_row_stride + col] = B16(acc[n][e]);
                if (n == 0 && (lane & 3) == 0 && e < 4 && (e & 1) == 0)
                    lse[qstart + row] = softmax.row_sum((e >> 1) & 1);
            }
        }
    }
#endif
}

} // namespace flash::wi
#endif

// Warp specialization (WS): separate producer and consumer groups.
namespace flash::ws {
using mha::copy_kv_tile;
using mha::uniform;
using Ops = mha::MhaOps<cutlass::bfloat16_t>;
using B16 = cutlass::bfloat16_t;
constexpr int M = detail::MhaWsQueryRows, N = detail::MhaWsKeyColumns, D = 192, V = 128, G = 256, Threads = 2 * G;
using Atom = decltype(cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_8, cute::_64>, cute::Stride<cute::_64, cute::_1>>{}));
template <int Rows, int Cols>
using Layout = decltype(cute::tile_to_shape(Atom{}, cute::Shape<cute::Int<Rows>, cute::Int<Cols>>{}));

struct alignas(128) Shared {
    B16 q[M * D];
    B16 k[2][N * D];
    B16 v[3][N * V];
    B16 p[2][M * N];
    float alpha[2][M], max[M], sum[M], partial[4][M];
};

__device__ __forceinline__ void group_sync() {
    __ppu_barrier_sync(12, G, 15u);
}

template <bool Paged>
__device__ __forceinline__ void produce_qk(Shared &s, const Flash_fwd_params &p, const B16 *k, const B16 *v,
                                        int batch, int qstart, int qlen, int klen, int first, int tiles, int tid) {
    int lane = tid & 31, warp = uniform(tid / 32), wm = warp % 2, wn = warp / 2;
    int r0 = wm * 16 + lane / 4, r1 = r0 + 8;
    int qt = int(reinterpret_cast<uintptr_t>(s.q) / 16) + wm * 128;
    int rq[12][4];
#pragma unroll
    for (int d = 0; d < 12; d++)
        Ops::ldm<false, 1, 64>(rq[d], qt + (d / 4) * M * 8 + (d % 4) * 2);
    auto gather = [&](int t) {
        copy_kv_tile<Paged, N, D>(s.k[t & 1], k, p.k_row_stride, p.k_batch_stride,
                                  p, batch, t * N, klen, tid, G);
        copy_kv_tile<Paged, N, V>(s.v[t % 3], v, p.v_row_stride, p.v_batch_stride,
                                  p, batch, t * N, klen, tid, G);
        cute::cp_async_fence();
    };
    gather(first);
    cute::cp_async_wait<0>();
    group_sync();
    float mx0 = -INFINITY, mx1 = -INFINITY, sum0 = 0, sum1 = 0;
    float scale = p.scale_softmax * 1.4426950408889634f;
    for (int t = first; t < first + tiles; t++) {
        int slot = t & 1;
        if (t >= first + 2)
            __ppu_barrier_sync(8 + ((t - 2) & 1), Threads, 15u);
        if (t + 1 < first + tiles)
            gather(t + 1);
        float chains[3][8] = {};
        float score[8];
        int kt = int(reinterpret_cast<uintptr_t>(s.k[slot]) / 16) + wn * 128;
#pragma unroll
        for (int d = 0; d < 12; d++) {
            int rk[4];
            Ops::ldm<false, 64, 1>(rk, kt + (d / 4) * N * 8 + (d % 4) * 2);
            Ops::mma(chains[d % 3], rq[d], rk);
        }
#pragma unroll
        for (int e = 0; e < 8; e++)
            score[e] = (chains[0][e] + chains[1][e]) + chains[2][e];
        float a = -INFINITY, b = -INFINITY;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int r = ((e >> 1) & 1) ? r1 : r0;
            int c = wn * 16 + (lane & 3) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
            bool valid = qstart + r < qlen && t * N + c < klen;
            valid = valid && t * N + c <= qstart + r + klen - qlen;
            score[e] = valid ? score[e] * scale : -INFINITY;
            if ((e >> 1) & 1)
                b = fmaxf(b, score[e]);
            else
                a = fmaxf(a, score[e]);
        }
#pragma unroll
        for (int delta = 1; delta <= 2; delta *= 2) {
            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, delta));
            b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, delta));
        }
        if ((lane & 3) == 0) {
            s.partial[wn][r0] = a;
            s.partial[wn][r1] = b;
        }
        group_sync();
        a = fmaxf(fmaxf(s.partial[0][r0], s.partial[1][r0]), fmaxf(s.partial[2][r0], s.partial[3][r0]));
        b = fmaxf(fmaxf(s.partial[0][r1], s.partial[1][r1]), fmaxf(s.partial[2][r1], s.partial[3][r1]));
        a = fmaxf(mx0, a);
        b = fmaxf(mx1, b);
        float alpha0 = mx0 == -INFINITY ? 0.f : exp2f(mx0 - a);
        float alpha1 = mx1 == -INFINITY ? 0.f : exp2f(mx1 - b);
        mx0 = a;
        mx1 = b;
        if (wn == 0 && (lane & 3) == 0) {
            s.alpha[slot][r0] = alpha0;
            s.alpha[slot][r1] = alpha1;
        }
        float z0 = 0, z1 = 0;
        auto sp = cute::make_tensor(cute::make_smem_ptr(s.p[slot]), Layout<M, N>{});
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int r = ((e >> 1) & 1) ? r1 : r0;
            int c = wn * 16 + (lane & 3) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
            float z = score[e] == -INFINITY ? 0.f : exp2f(score[e] - (((e >> 1) & 1) ? mx1 : mx0));
            sp(r, c) = B16(z);
            if ((e >> 1) & 1)
                z1 += z;
            else
                z0 += z;
        }
#pragma unroll
        for (int delta = 1; delta <= 2; delta *= 2) {
            z0 += __shfl_xor_sync(0xffffffff, z0, delta);
            z1 += __shfl_xor_sync(0xffffffff, z1, delta);
        }
        sum0 = sum0 * alpha0 + z0;
        sum1 = sum1 * alpha1 + z1;
        __syncwarp();
        __ppu_barrier_arrive(4 + slot, Threads, 15u);
        if (t + 1 < first + tiles) {
            cute::cp_async_wait<0>();
            group_sync();
        }
    }
    if (tiles >= 2)
        __ppu_barrier_sync(8 + ((first + tiles - 2) & 1), Threads, 15u);
    __ppu_barrier_sync(8 + ((first + tiles - 1) & 1), Threads, 15u);
    if ((lane & 3) == 0) {
        s.partial[wn][r0] = sum0;
        s.partial[wn][r1] = sum1;
        if (wn == 0) {
            s.max[r0] = mx0;
            s.max[r1] = mx1;
        }
    }
    group_sync();
    if (tid < M)
        s.sum[tid] = s.partial[0][tid] + s.partial[1][tid] + s.partial[2][tid] + s.partial[3][tid];
    __ppu_barrier_sync(0, Threads, 15u);
}

template <int Splits>
__device__ __forceinline__ void consume_pv(Shared &s, const Flash_fwd_params &p, B16 *out, float *lse,
                                        float *partial, float *partial_lse, int qstart, int qlen, int totalq,
                                        int first, int tiles, int tid) {
    int lane = tid & 31, warp = uniform(tid / 32);
    float acc[2][8] = {};
    for (int t = first; t < first + tiles; t++) {
        int slot = t & 1;
        __ppu_barrier_sync(4 + slot, Threads, 15u);
#pragma unroll
        for (int m = 0; m < 2; m++) {
#pragma unroll
            for (int e = 0; e < 8; e++)
                acc[m][e] *= s.alpha[slot][m * 16 + lane / 4 + ((e >> 1) & 1) * 8];
        }
        int rp[2][4][4], rv[4][4];
        int pt = int(reinterpret_cast<uintptr_t>(s.p[slot]) / 16);
        int vt = int(reinterpret_cast<uintptr_t>(s.v[t % 3]) / 16) + (warp / 4) * N * 8 + (warp % 4) * 2;
#pragma unroll
        for (int d = 0; d < 4; d++) {
#pragma unroll
            for (int m = 0; m < 2; m++)
                Ops::ldm<false, 1, 64>(rp[m][d], pt + m * 128 + d * 2);
            Ops::ldm<true, 1, 64>(rv[d], vt + d * 128);
        }
        __syncwarp();
        __ppu_sched_bound();
        __ppu_barrier_arrive(8 + slot, Threads, 15u);
#pragma unroll
        for (int d = 0; d < 4; d++) {
#pragma unroll
            for (int m = 0; m < 2; m++)
                Ops::mma(acc[m], rp[m][d], rv[d]);
        }
    }
    __ppu_barrier_sync(0, Threads, 15u);
#pragma unroll
    for (int m = 0; m < 2; m++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            int r = m * 16 + lane / 4 + ((e >> 1) & 1) * 8;
            int c = warp * 16 + (lane & 3) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
            if (qstart + r < qlen) {
                float inv = s.sum[r] > 0 ? __ppu_rcpf(s.sum[r]) : 0.f;
                if constexpr (Splits > 1) {
                    partial[int64_t(qstart + r) * V + c] = acc[m][e] * inv;
                    if (warp == 0 && (lane & 3) == 0 && e < 4 && (e & 1) == 0)
                        partial_lse[qstart + r] =
                            s.sum[r] > 0 ? (s.max[r] + log2f(s.sum[r])) * 0.6931471805599453f : -INFINITY;
                } else {
                    out[int64_t(qstart + r) * p.o_row_stride + c] = B16(acc[m][e] * inv);
                    if (warp == 0 && (lane & 3) == 0 && e < 4 && (e & 1) == 0)
                        lse[qstart + r] = s.sum[r] > 0 ? (s.max[r] + log2f(s.sum[r])) * 0.6931471805599453f
                                                       : (p.seqlen_k == 0 ? INFINITY : -INFINITY);
                }
            }
        }
    }
}

template <int Splits = 1, bool Paged = false, bool Varlen = true, bool Retry = false>
__global__ __launch_bounds__(Threads) void mha_ws_bf16(Flash_fwd_params p, int const* retry_flags) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    extern __shared__ __align__(128) unsigned char bytes[];
    Shared &s = *reinterpret_cast<Shared *>(bytes);
    int batch, split, head, qstart, partitions = Splits, retry_slot = 0;
    if constexpr (Retry) {
        partitions = mha::mha_ws_retry_splits(p, retry_flags);
        retry_slot = blockIdx.y;
        split = blockIdx.z;
        if (partitions == 0 || split >= partitions || retry_slot >= uniform(retry_flags[0])) return;
        auto const tile = mha::mha_retry_queue(p, retry_flags)[retry_slot];
        int const query_head = uniform(tile.query_head);
        int const blocks = (p.seqlen_q + detail::MhaWiQueryRows - 1) / detail::MhaWiQueryRows;
        head = query_head % p.h;
        batch = uniform(tile.batch);
        qstart = (blocks - 1 - query_head / p.h) * detail::MhaWiQueryRows + blockIdx.x * M;
    } else {
        batch = blockIdx.z / Splits;
        split = blockIdx.z % Splits;
        head = blockIdx.y;
        qstart = blockIdx.x * M;
    }
    SeqlenInfoQK<Varlen, M> const info(batch, p.seqlen_q, p.seqlen_k,
        p.cu_seqlens_q, Paged ? nullptr : p.cu_seqlens_k,
        nullptr, Paged ? p.seqused_k : nullptr);
    int qo = info.offset_q, ko = info.offset_k;
    int qlen = uniform(info.seqlen_q), klen = uniform(info.seqlen_k);
    if (qstart >= qlen)
        return;
    const B16 *q = static_cast<const B16 *>(p.q_ptr) + int64_t(qo) * p.q_row_stride + head * p.q_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.q_batch_stride);
    const B16 *k = static_cast<const B16 *>(p.k_ptr) + int64_t(ko) * p.k_row_stride + head * p.k_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.k_batch_stride);
    const B16 *v = static_cast<const B16 *>(p.v_ptr) + int64_t(ko) * p.v_row_stride + head * p.v_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.v_batch_stride);
    B16 *o = static_cast<B16 *>(p.o_ptr) + int64_t(qo) * p.o_row_stride + head * p.o_head_stride
        + (Varlen ? 0 : int64_t(batch) * p.o_batch_stride);
    float *lse = static_cast<float *>(p.softmax_lse_ptr)
        + (Varlen ? int64_t(head) * p.total_q + qo
                  : (int64_t(batch) * p.h + head) * p.seqlen_q);
    Ops::copy_tile<M, D>(s.q, q, p.q_row_stride, qstart, qlen, threadIdx.x, Threads);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
    int last = klen;
    last = min(klen, max(0, qstart + M + klen - qlen));
    int tiles = max(1, (last + N - 1) / N);
    int first = 0;
    float *partial = nullptr;
    float *partial_lse = nullptr;
    if constexpr (Splits > 1) {
        int chunk = (tiles + partitions - 1) / partitions;
        first = min(split * chunk, tiles);
        tiles = max(1, min(chunk, tiles - first));
        int64_t offset;
        if constexpr (Retry) {
            offset = (int64_t(split) * mha::mha_ws_retry_capacity(p) + retry_slot) * detail::MhaWiQueryRows;
        } else {
            // Both layouts use packed logical rows in the partial buffers.
            offset = (int64_t(split) * p.h + head) * p.total_q
                + (Varlen ? qo : int64_t(batch) * p.seqlen_q);
        }
        partial = static_cast<float *>(p.oaccum_ptr) + offset * V;
        partial_lse = static_cast<float *>(p.softmax_lseaccum_ptr) + offset;
    }
    if (uniform(threadIdx.x / G) == 0) {
        produce_qk<Paged>(s, p, k, v, batch, qstart, qlen, klen, first, tiles, threadIdx.x);
    } else {
        // Compact partials fold the original batch/head into retry_slot.
        int const query_base = Retry ? (qstart / detail::MhaWiQueryRows) * detail::MhaWiQueryRows : 0;
        consume_pv<Splits>(s, p, o, lse, partial, partial_lse, qstart - query_base, qlen - query_base,
                          p.total_q, first, tiles, threadIdx.x - G);
    }
#endif
}

}  // namespace flash::ws

namespace flash::mha {
// Normalized float partials and LSE use the same layout for WI and WS.
template <int Splits, int ValueDim, bool Varlen, int RowsPerBlock = 8, bool RetryGuard = false, bool CompactRetry = false>
__global__ void merge_mha_bf16_splits(Flash_fwd_params p, int const* retry_flags) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    using B16 = cutlass::bfloat16_t;
    static_assert(RowsPerBlock % 8 == 0);
    int head = RetryGuard ? int(blockIdx.x) % p.h : int(blockIdx.y);
    int const lane = threadIdx.x & 31;
    int first_row = blockIdx.x * RowsPerBlock, end_row = p.total_q;
    int active_splits = Splits, retry_slot = 0;
    int64_t partial_stride = int64_t(p.h) * p.total_q;
    static_assert(!CompactRetry || RetryGuard);
    if constexpr (RetryGuard) {
        int const sparse_splits = mha_ws_retry_splits(p, retry_flags);
        if constexpr (CompactRetry) {
            if (sparse_splits == 0) return;
            active_splits = sparse_splits;
            partial_stride = int64_t(mha_ws_retry_capacity(p)) * detail::MhaWiQueryRows;
        } else {
            if (sparse_splits) return;
        }
        static_assert(detail::MhaWiQueryRows % RowsPerBlock == 0);
        int64_t index;
        int batch, query_block;
        int const blocks = (p.seqlen_q + detail::MhaWiQueryRows - 1) / detail::MhaWiQueryRows;
        if constexpr (CompactRetry) {
            retry_slot = blockIdx.x;
            if (retry_slot >= uniform(retry_flags[0])) return;
            auto const tile = mha_retry_queue(p, retry_flags)[retry_slot];
            int const query_head = uniform(tile.query_head);
            head = query_head % p.h;
            batch = uniform(tile.batch);
            query_block = blocks - 1 - query_head / p.h;
        } else {
            // Keep unmodified normalized KV partials and merge only the rows
            // belonging to a repaired query/head/batch tile.
            index = int64_t(blockIdx.y) * gridDim.x + blockIdx.x;
            if (!uniform(retry_flags[1 + index])) return;
            batch = blockIdx.y;
            query_block = blocks - 1 - blockIdx.x / p.h;
        }
        int const offset = Varlen ? p.cu_seqlens_q[batch] : batch * p.seqlen_q;
        int const length = Varlen ? p.cu_seqlens_q[batch + 1] - offset : p.seqlen_q;
        first_row = offset + query_block * detail::MhaWiQueryRows + blockIdx.z * RowsPerBlock;
        end_row = offset + length;
    }
#pragma unroll
    for (int i = 0; i < RowsPerBlock / 8; ++i) {
        int const row = first_row + threadIdx.x / 32 + i * 8;
        if (row >= end_row) continue;
        int64_t const partial_row = CompactRetry
            ? int64_t(retry_slot) * detail::MhaWiQueryRows + blockIdx.z * RowsPerBlock + threadIdx.x / 32 + i * 8
            : int64_t(head) * p.total_q + row;
        const float *pl = static_cast<const float *>(p.softmax_lseaccum_ptr);
        const float *po = static_cast<const float *>(p.oaccum_ptr);
        float weights[Splits], mx = -INFINITY, denom = 0;
#pragma unroll
        for (int s = 0; s < Splits; s++) {
            weights[s] = s < active_splits ? pl[int64_t(s) * partial_stride + partial_row] : -INFINITY;
            mx = fmaxf(mx, weights[s]);
        }
#pragma unroll
        for (int s = 0; s < Splits; s++) {
            weights[s] = mx == -INFINITY ? 0.f : expf(weights[s] - mx);
            denom += weights[s];
        }
        float inv = denom > 0 ? __ppu_rcpf(denom) : 0.f;
        // Form a convex combination before accumulating. Summing unnormalized
        // partials can overflow even when every partial and the result are finite.
#pragma unroll
        for (int s = 0; s < Splits; ++s) weights[s] *= inv;
        int const batch = Varlen ? 0 : row / p.seqlen_q;
        int const qrow = Varlen ? row : row % p.seqlen_q;
        B16 *out = static_cast<B16 *>(p.o_ptr) + int64_t(batch) * p.o_batch_stride
            + int64_t(qrow) * p.o_row_stride + head * p.o_head_stride;
        int64_t const lse_offset = Varlen ? int64_t(head) * p.total_q + row
            : (int64_t(batch) * p.h + head) * p.seqlen_q + qrow;
        if constexpr (RowsPerBlock > 8 && ValueDim % 128 == 0) {
            // Partials are contiguous/aligned; the caller's output may have
            // arbitrary positive strides, so vector stores retain an alignment fallback.
#pragma unroll
            for (int col = lane * 4; col < ValueDim; col += 128) {
                float4 value = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int split = 0; split < Splits; ++split) {
                    if (split >= active_splits) continue;
                    auto index = (int64_t(split) * partial_stride + partial_row) * ValueDim + col;
                    float4 v = *reinterpret_cast<float4 const *>(po + index);
                    value.x = fmaf(weights[split], v.x, value.x);
                    value.y = fmaf(weights[split], v.y, value.y);
                    value.z = fmaf(weights[split], v.z, value.z);
                    value.w = fmaf(weights[split], v.w, value.w);
                }
                if ((reinterpret_cast<uintptr_t>(out) & 7) == 0) {
                    uint2 packed = {unsigned(MhaOps<B16>::pack(value.x, value.y)),
                                    unsigned(MhaOps<B16>::pack(value.z, value.w))};
                    *reinterpret_cast<uint2 *>(out + col) = packed;
                } else {
                    out[col] = B16(value.x);
                    out[col + 1] = B16(value.y);
                    out[col + 2] = B16(value.z);
                    out[col + 3] = B16(value.w);
                }
            }
        } else {
#pragma unroll
            for (int col = lane; col < ValueDim; col += 32) {
                float value = 0;
#pragma unroll
                for (int s = 0; s < Splits; s++) {
                    if (s < active_splits)
                        value = fmaf(weights[s], po[(int64_t(s) * partial_stride + partial_row) * ValueDim + col], value);
                }
                out[col] = B16(value);
            }
        }
        if (lane == 0)
            static_cast<float *>(p.softmax_lse_ptr)[lse_offset] =
                denom > 0 ? mx + logf(denom) : (p.seqlen_k == 0 ? INFINITY : -INFINITY);
    }
#endif
}

}  // namespace flash::mha

namespace flash::ws {
template <bool Paged, bool Varlen, int Splits>
inline hggcError_t launch_mha_ws_impl(Flash_fwd_params &p, hggcStream_t stream) {
    auto k = mha_ws_bf16<Splits, Paged, Varlen>;
    auto status = hggcFuncSetAttribute(k, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared));
    if (status != hggcSuccess)
        return status;
    k<<<dim3((p.seqlen_q + M - 1) / M, p.h, p.b * Splits), Threads, sizeof(Shared), stream>>>(p, nullptr);
    if constexpr (Splits > 1)
        mha::merge_mha_bf16_splits<Splits, V, Varlen><<<dim3((p.total_q + 7) / 8, p.h), 256, 0, stream>>>(p, nullptr);
    return hggcGetLastError();
}
template <bool Paged, bool Varlen>
inline hggcError_t launch_mha_ws(Flash_fwd_params &p, int splits, hggcStream_t stream) {
    if (p.total_q == 0)
        return hggcSuccess;
    if (p.oaccum_ptr && p.softmax_lseaccum_ptr) {
        if (splits == 8) return launch_mha_ws_impl<Paged, Varlen, 8>(p, stream);
        if (splits == 4) return launch_mha_ws_impl<Paged, Varlen, 4>(p, stream);
        if (splits == 2) return launch_mha_ws_impl<Paged, Varlen, 2>(p, stream);
    }
    return launch_mha_ws_impl<Paged, Varlen, 1>(p, stream);
}
} // namespace flash::ws

#endif  // FLASHATTENTION_MHA_WI_WS

// FP8 WI packs V for both dense and paged KV.
#ifdef FLASHATTENTION_MHA_FP8_FAST_PATH
namespace flash::mha {
// V packing is optional scratch: do not let rectangular batch padding or a
// page-table capacity dominate the caller's actual KV storage. Also cap its
// device-wide footprint at 1/64 of memory. Query driver-free space without
// reading device lengths or explicitly synchronizing the caller stream.
inline bool fp8_v_workspace_bytes(Flash_fwd_params const& p, int64_t& bytes) {
    constexpr int64_t PageRows = 64, MaxStorageExpansion = 2, MemoryBudgetDivisor = 64;
    constexpr size_t FreeMemoryDivisor = 8;
    if (p.b <= 0 || p.h_k <= 0 || p.dv <= 0 || p.seqlen_k <= 0) return false;
    int64_t const stored_pages = p.page_table ? p.num_pages
        : (int64_t(p.total_k) + (PageRows - 1) * p.b) / PageRows;
    if (stored_pages <= 0) return false;
    int64_t const page_bytes = PageRows * p.h_k * p.dv;
    int64_t const budget = at::cuda::getCurrentDeviceProperties()->totalGlobalMem / MemoryBudgetDivisor;
    int64_t const max_pages = std::min(budget / page_bytes, MaxStorageExpansion * stored_pages);
    int64_t const pages = (int64_t(p.seqlen_k) + PageRows - 1) / PageRows;
    // Divide before multiplying so rejected dimensions cannot overflow sizing.
    if (pages > max_pages / p.b) return false;
    bytes = pages * p.b * page_bytes;
    // Only spend a small fraction of current driver-free memory. Do not count
    // allocator caches or graph-private pools as freely reusable KV capacity.
    size_t free_bytes, total_bytes;
    CHECK_CUDA(hggcMemGetInfo(&free_bytes, &total_bytes));
    return uint64_t(bytes) <= free_bytes / FreeMemoryDivisor;
}

// Arrange each 64-token V page for the existing FP8 MMA A-fragment permutation.
// Packing is part of the public API invocation, including graph replay.
template <int ValueDim, bool Paged, bool Varlen>
__global__ void prepack_fp8_v(Flash_fwd_params p, cutlass::float_e4m3_t *packed) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    __shared__ unsigned char tile[64][ValueDim + 4];
    int head = blockIdx.y, batch = blockIdx.z, tid = threadIdx.x;
    int const pages = (p.seqlen_k + 63) / 64;
    int used_pages = pages;
    if constexpr (Paged) used_pages = min(pages, max(0, (p.seqused_k[batch] + 63) / 64));
    // Bound the launch by the host work estimate while retaining capacity
    // strides. Device lengths may grow on graph replay: grid-stride pages
    // cover the tail without launching one CTA per reserved empty page.
    for (int page = blockIdx.x; page < used_pages; page += gridDim.x) {
        int length, offset = 0;
        if constexpr (Paged) {
            length = max(0, min(64, p.seqused_k[batch] - 64 * page));
        } else {
            SeqlenInfo<Varlen, 64> const info(batch, p.seqlen_k, p.cu_seqlens_k, nullptr);
            offset = info.offset;
            length = max(0, min(64, info.seqlen - 64 * page));
        }
        // Consumers never read pages beyond the sequence's used KV length.
        // The check is block-uniform and precedes page-table reads and the barrier.
        if (length == 0) return;
        auto src = static_cast<const cutlass::float_e4m3_t *>(p.v_ptr) + head * p.v_head_stride;
        if constexpr (Paged) {
            int physical = p.page_table[int64_t(batch) * p.page_table_batch_stride + page];
            src += int64_t(physical) * p.v_batch_stride;
        } else {
            src += int64_t(offset + 64 * page) * p.v_row_stride
                + (Varlen ? 0 : int64_t(batch) * p.v_batch_stride);
        }
        for (int i = tid; i < 64 * ValueDim; i += 256) {
            int token = i / ValueDim, column = i % ValueDim;
            tile[token][column] = token < length ? src[int64_t(token) * p.v_row_stride + column].storage : 0;
        }
        __syncthreads();
        auto dst = packed + ((int64_t(batch) * pages + page) * p.h_k + head) * ValueDim * 64;
        for (int i = tid; i < 64 * ValueDim; i += 256) {
            int column = i / 64, j = i % 64, low = j & 15;
            int token = (j & ~15) + 2 * (low / 4) + (low & 1) + 8 * ((low % 4) / 2);
            dst[i].storage = tile[token][column];
        }
        if constexpr (!Paged) break;
        // All threads must finish reading the shared tile before its reuse.
        if (page + gridDim.x < used_pages) __syncthreads();
    }
#endif
}
}  // namespace flash::mha
#endif

// Execution entries; public eligibility contracts live in flash_fwd_fastpaths.h.
#ifdef FLASHATTENTION_MHA_WI_WS
namespace flash {
namespace {

// Both BF16 schedules keep normalized partials in packed logical-row order.
// Owning these tensors through launch and merge preserves stream/graph lifetime.
struct MhaSplitWorkspace {
    Flash_fwd_params params;
    at::Tensor output, lse;
    // compact_rows folds the affected batch/head tiles into bounded slots;
    // params retains the original Q layout for addressing the caller's tensors.
    MhaSplitWorkspace(Flash_fwd_params const& p, int splits, int compact_rows = 0) : params(p) {
        if (splits > 1) {
            auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
            int const heads = compact_rows ? 1 : p.h, rows = compact_rows ? compact_rows : p.total_q;
            output = at::empty({splits, heads, rows, p.dv}, opts);
            lse = at::empty({splits, heads, rows}, opts);
            params.oaccum_ptr = output.data_ptr();
            params.softmax_lseaccum_ptr = lse.data_ptr();
        }
    }
};

template <bool Paged, bool Varlen>
void launch_mha_bf16_ws_retry(Flash_fwd_params& p, hggcStream_t stream, int* retry_flags) {
    int const capacity = mha::mha_ws_retry_capacity(p);
    if (capacity == 0) return;
    int64_t const tiles = int64_t(p.b) * p.h * ((int64_t(p.seqlen_q) + wi::BM - 1) / wi::BM);
    int const blocks = int(std::min((tiles + 127) / 128, int64_t(p.num_sm)));
    mha::prepare_mha_wi_retry<<<blocks, 128, 0, stream>>>(retry_flags, tiles, int(tiles / p.b), mha::mha_ws_retry_capacity(p));
    CHECK_CUDA(hggcGetLastError());
    constexpr int MaxSplits = mha::MhaWsRetryMaxSplits, MergeRows = 32;
    MhaSplitWorkspace workspace(p, MaxSplits, capacity * wi::BM);
    auto kernel = ws::mha_ws_bf16<MaxSplits, Paged, Varlen, true>;
    CHECK_CUDA(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(ws::Shared)));
    kernel<<<dim3(wi::BM / ws::M, capacity, MaxSplits), ws::Threads, sizeof(ws::Shared), stream>>>(workspace.params, retry_flags);
    CHECK_CUDA(hggcGetLastError());
    mha::merge_mha_bf16_splits<MaxSplits, wi::V, Varlen, MergeRows, true, true>
        <<<dim3(capacity, 1, wi::BM / MergeRows), 256, 0, stream>>>(workspace.params, retry_flags);
    CHECK_CUDA(hggcGetLastError());
}

template <bool Paged, bool Varlen, int Splits, bool Retry>
void launch_mha_bf16_wi(Flash_fwd_params& p, hggcStream_t stream, int* retry_flags) {
    if constexpr (Retry) launch_mha_bf16_ws_retry<Paged, Varlen>(p, stream, retry_flags);
    CHECK_CUDA((wi::launch_mha_wi<Paged, Varlen, Splits, Retry>(p, stream, retry_flags)));
    if constexpr (Splits > 1) {
        constexpr int MergeRows = 32;
        dim3 const grid = Retry
            ? dim3(((p.seqlen_q + wi::BM - 1) / wi::BM) * p.h, p.b, wi::BM / MergeRows)
            : dim3((p.total_q + MergeRows - 1) / MergeRows, p.h);
        mha::merge_mha_bf16_splits<Splits, wi::V, Varlen, MergeRows, Retry>
            <<<grid, 256, 0, stream>>>(p, retry_flags);
        CHECK_CUDA(hggcGetLastError());
    }
}

template <bool Paged, bool Varlen, bool Retry>
void launch_mha_bf16_wi(Flash_fwd_params& p, int splits, hggcStream_t stream, int* retry_flags) {
    if (splits == 2) launch_mha_bf16_wi<Paged, Varlen, 2, Retry>(p, stream, retry_flags);
    else launch_mha_bf16_wi<Paged, Varlen, 1, Retry>(p, stream, retry_flags);
}

template <bool Retry>
void launch_mha_bf16_wi_by_layout(Flash_fwd_params& p, int splits, hggcStream_t stream, int* retry_flags) {
#ifndef FLASHATTENTION_DISABLE_PAGEDKV
    if (p.page_table) launch_mha_bf16_wi<true, true, Retry>(p, splits, stream, retry_flags);
    else
#endif
    if (p.cu_seqlens_q) launch_mha_bf16_wi<false, true, Retry>(p, splits, stream, retry_flags);
    else launch_mha_bf16_wi<false, false, Retry>(p, splits, stream, retry_flags);
}

#if !defined(FLASHATTENTION_DISABLE_FP8)
template <bool Paged, bool Varlen>
bool launch_mha_fp8_wi(Flash_fwd_params &p, hggcStream_t stream, int max_seqlen_k) {
    int64_t bytes;
    if (!mha::fp8_v_workspace_bytes(p, bytes)) return false;
    using namespace flash::wi;
    using Shared = Fp8MhaStorage<192, 128, 128>;
    int pages = (p.seqlen_k + 63) / 64;
    if constexpr (Paged) pages = (detail::mha_kv_workload_bound(p, max_seqlen_k) + 63) / 64;
    at::Tensor packed;
    try {
        packed = at::empty({bytes},
            at::TensorOptions().device(at::kCUDA).dtype(at::kByte));
    } catch (c10::OutOfMemoryError const&) {
        // A competing allocation can consume memory after the free-space check.
        // Eager calls can fall back; do not hide an invalidated graph capture.
        hggcStreamCaptureStatus capture_status;
        CHECK_CUDA(hggcStreamIsCapturing(stream, &capture_status));
        if (capture_status != hggcStreamCaptureStatusNone) throw;
        return false;
    }
    auto data = reinterpret_cast<Fp8 *>(packed.data_ptr());
    mha::prepack_fp8_v<128, Paged, Varlen><<<dim3(pages, p.h_k, p.b), 256, 0, stream>>>(p, data);
    CHECK_CUDA(hggcGetLastError());
    auto kernel = mha_wi_fp8<192, 128, 128, Paged, Varlen>;
    CHECK_CUDA(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared)));
    kernel<<<dim3(((p.seqlen_q + Shared::BlockM - 1) / Shared::BlockM) * p.h, p.b),
             Shared::Threads, sizeof(Shared), stream>>>(p, data, float(p.scale_softmax * M_LOG2E));
    CHECK_CUDA(hggcGetLastError());
    return true;
}
#endif
}  // namespace

static bool try_run_mha_bf16_wi(Flash_fwd_params &p, hggcStream_t stream, int max_seqlen_k) {
    if (!supports_mha_bf16_wi(p, max_seqlen_k)) return false;
    int const splits = detail::mha_wi_num_splits(p, max_seqlen_k);
    MhaSplitWorkspace workspace(p, splits);
    // Actual paged lengths still come from seqused_k, including graph replay.
    // This private bound selects the loop/retry schedule without changing
    // native dispatch, page addressing or the caller's capacity metadata.
    if (p.page_table) {
        int64_t const work_k = detail::mha_kv_workload_bound(p, max_seqlen_k);
        workspace.params.seqlen_k = int(std::min(int64_t(p.seqlen_k),
            ((work_k + wi::N - 1) / wi::N) * wi::N));
    }
    workspace.params.num_splits = splits;
    launch_mha_bf16_wi_by_layout<false>(workspace.params, splits, stream, nullptr);
    return true;
}

#if !defined(FLASHATTENTION_DISABLE_FP8)
static bool try_run_mha_fp8_wi(Flash_fwd_params &p, hggcStream_t stream, int max_seqlen_k) {
    if (!supports_mha_fp8_wi(p, max_seqlen_k)) return false;
#ifndef FLASHATTENTION_DISABLE_PAGEDKV
    if (p.page_table) return launch_mha_fp8_wi<true, true>(p, stream, max_seqlen_k);
    else
#endif
    if (p.cu_seqlens_q) return launch_mha_fp8_wi<false, true>(p, stream, max_seqlen_k);
    return launch_mha_fp8_wi<false, false>(p, stream, max_seqlen_k);
}
#endif
}  // namespace flash

namespace flash {

static bool try_run_mha_bf16_ws(Flash_fwd_params &p, hggcStream_t stream, int max_seqlen_k) {
    if (!supports_mha_bf16_ws(p, max_seqlen_k))
        return false;
    // Allocate on the API's current CUDA stream; the caching allocator also
    // preserves these buffers across CUDA graph capture and replay.
    int const splits = detail::mha_ws_num_splits(p, max_seqlen_k);
    MhaSplitWorkspace workspace(p, splits);
    if (p.page_table) workspace.params.seqlen_k = detail::mha_kv_workload_bound(p, max_seqlen_k);
    Flash_fwd_params& params = workspace.params;
#ifndef FLASHATTENTION_DISABLE_PAGEDKV
    if (p.page_table) CHECK_CUDA((ws::launch_mha_ws<true, true>(params, splits, stream)));
    else
#endif
    if (p.cu_seqlens_q) CHECK_CUDA((ws::launch_mha_ws<false, true>(params, splits, stream)));
    else CHECK_CUDA((ws::launch_mha_ws<false, false>(params, splits, stream)));
    return true;
}

}  // namespace flash

#endif  // FLASHATTENTION_MHA_WI_WS

namespace flash {
#ifdef FLASHATTENTION_MHA_WI_WS
static bool try_run_mha_bf16_fast_path(Flash_fwd_params &p, hggcStream_t stream, int max_seqlen_k) {
    return try_run_mha_bf16_wi(p, stream, max_seqlen_k) || try_run_mha_bf16_ws(p, stream, max_seqlen_k);
}
#endif

#ifdef FLASHATTENTION_MHA_FP8_FAST_PATH
static bool try_run_mha_fp8_fast_path(Flash_fwd_params &p, hggcStream_t stream, int max_seqlen_k) {
    return try_run_mha_fp8_wi(p, stream, max_seqlen_k);
}
#endif
}  // namespace flash

#ifdef FLASHATTENTION_FWD_FASTPATHS
namespace flash {
bool try_run_mha_fast_path(Flash_fwd_params &p, hggcStream_t stream, int max_seqlen_k) {
#ifdef FLASHATTENTION_MHA_WI_WS
    if (p.is_bf16) return try_run_mha_bf16_fast_path(p, stream, max_seqlen_k);
#endif
#ifdef FLASHATTENTION_MHA_FP8_FAST_PATH
    if (p.is_e4m3) return try_run_mha_fp8_fast_path(p, stream, max_seqlen_k);
#endif
    return false;
}
}  // namespace flash
#endif
