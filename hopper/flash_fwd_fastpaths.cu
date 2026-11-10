// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// Warp-interleaved MHA forward: BF16 and FP8 with optional V prepacking.

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

#ifdef FLASHATTENTION_MHA_WI
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
    union {
        B16 k[Stages][N * D];
        int query_tile[2];  // Scratch before Q preload completes.
    };
};
static_assert(sizeof(Storage) == 245760);

// Update the online maximum before BF16 probability packing. Keeping each
// tile normalized avoids the additional rounding and overflow of deferred maxima.
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
    // A warp vote is sufficient when every lane's local scores remain
    // below the running maximum. Reduce across lanes only for an update.
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

// Locate a compact query tile without a host readback or a metadata kernel.
// Contiguous KV shares the prefix scan through K scratch. Paged KV uses
// warp-local indices to avoid an extra CTA rendezvous in its entry path.
template <bool SharedBroadcast>
__device__ __forceinline__ bool compact_query_tile(
    Flash_fwd_params const& p, int tile, int& batch, int& qstart, int* metadata) {
    constexpr int WarpSize = cutlass::NumThreadsPerWarp;
    if (!SharedBroadcast || threadIdx.x < WarpSize) {
        int const lane = threadIdx.x % WarpSize;
        int selected_batch = -1, selected_start = 0;
        for (int first = 0; first < p.b; first += WarpSize) {
            int const sequence = first + lane;
            int const length = sequence < p.b
                ? p.cu_seqlens_q[sequence + 1] - p.cu_seqlens_q[sequence] : 0;
            int const tiles = (length + BM - 1) / BM;
            int const prefix = warp_prefix_sum(tiles);
            int const group_tiles = uniform(__shfl_sync(0xffffffff, prefix, WarpSize - 1));
            if (tile < group_tiles) {
                int const owner = uniform(__ffs(__ballot_sync(0xffffffff, tile < prefix)) - 1);
                if constexpr (SharedBroadcast) {
                    selected_batch = first + owner;
                    selected_start = uniform((__shfl_sync(0xffffffff, prefix, owner) - 1 - tile) * BM);
                    break;
                } else {
                    batch = first + owner;
                    qstart = uniform((__shfl_sync(0xffffffff, prefix, owner) - 1 - tile) * BM);
                    return true;
                }
            }
            tile -= group_tiles;
        }
        if constexpr (SharedBroadcast) {
            if (lane == 0) {
                metadata[0] = selected_batch;
                metadata[1] = selected_start;
            }
        }
    }
    if constexpr (SharedBroadcast) {
        __syncthreads();
        batch = uniform(metadata[0]);
        qstart = uniform(metadata[1]);
        return batch >= 0;
    }
    return false;  // The host bound can include unused final tiles.
}

template <bool Paged = false, bool Varlen = true, bool Compact = false>
__global__ __launch_bounds__(NT) void mha_wi_bf16_guarded(Flash_fwd_params p) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150

    extern __shared__ __align__(128) unsigned char bytes[];
    auto &s = *reinterpret_cast<Storage *>(bytes);
    int tid = threadIdx.x, lane = tid & 31, w = uniform(tid / 32), group = uniform(tid / WG), gt = tid % WG;
    int blocks = (p.seqlen_q + BM - 1) / BM;
    int head = blockIdx.x % p.h, batch = blockIdx.y;
    int qstart = Compact ? 0 : (blocks - 1 - blockIdx.x / p.h) * BM;
    if constexpr (Compact) {
        // K staging starts only after the existing Q copy/preload barriers,
        // so every warp reads this scratch before gather overwrites it.
        if (!compact_query_tile<!Paged>(p, blockIdx.x / p.h, batch, qstart, s.query_tile)) return;
    }
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

template <bool Paged = false, bool Varlen = true, int Splits = 1, bool Compact = false>
__global__ __launch_bounds__(NT) void mha_wi_bf16_peeled(Flash_fwd_params p) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150

    extern __shared__ __align__(128) unsigned char bytes[];
    auto &s = *reinterpret_cast<Storage *>(bytes);
    int tid = threadIdx.x, lane = tid & 31, w = uniform(tid / 32), group = uniform(tid / WG), gt = tid % WG;
    int blocks = (p.seqlen_q + BM - 1) / BM;
    int head = blockIdx.x % p.h, batch = blockIdx.y;
    int qstart = Compact ? 0 : (blocks - 1 - blockIdx.x / p.h) * BM;
    if constexpr (Compact) {
        // K staging starts only after the existing Q copy/preload barriers,
        // so every warp reads this scratch before gather overwrites it.
        if (!compact_query_tile<!Paged>(p, blockIdx.x / p.h, batch, qstart, s.query_tile)) return;
    }
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

template <bool Paged = false, bool Varlen = true, int Splits = 1>
inline hggcError_t launch_mha_wi(Flash_fwd_params &p, hggcStream_t stream) {
    auto launch = [&](auto compact) {
        constexpr bool Compact = decltype(compact)::value;
        auto k = mha_wi_bf16_peeled<Paged, Varlen, Splits, Compact>;
        if constexpr (Splits == 1) {
            if (p.seqlen_k < 32768) k = mha_wi_bf16_guarded<Paged, Varlen, Compact>;
        }
        auto status = hggcFuncSetAttribute(k, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage));
        if (status != hggcSuccess) return status;
        int64_t const tiles = Compact ? detail::mha_query_tiles_upper_bound(p, BM)
                                      : (int64_t(p.seqlen_q) + BM - 1) / BM;
        k<<<dim3(tiles * p.h, Compact ? 1 : p.b, Splits), NT, sizeof(Storage), stream>>>(p);
        return hggcGetLastError();
    };
    if constexpr (Varlen) {
        if (detail::mha_wi_compact_grid(p)) return launch(std::true_type{});
    }
    return launch(std::false_type{});
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

#endif  // FLASHATTENTION_MHA_WI

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
#ifdef FLASHATTENTION_MHA_WI
namespace flash {
namespace {

// WI partials use packed logical rows, including fixed and ragged batches.
// Reuse the existing FP32 split-combine implementation and its stride contract.
struct MhaSplitWorkspace {
    Flash_fwd_params params;
    at::Tensor output, lse;
    MhaSplitWorkspace(Flash_fwd_params const& p, int splits) : params(p) {
        params.num_splits = splits;
        params.num_splits_dynamic_ptr = nullptr;
        params.tile_count_semaphore = nullptr;
        if (splits > 1) {
            auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
            output = at::empty({splits, p.h, p.total_q, p.dv}, opts);
            lse = at::empty({splits, p.h, p.total_q}, opts);
            params.oaccum_ptr = output.data_ptr();
            params.softmax_lseaccum_ptr = lse.data_ptr();
            params.oaccum_row_stride = p.dv;
            params.oaccum_head_stride = int64_t(p.total_q) * p.dv;
            params.oaccum_split_stride = int64_t(p.h) * p.total_q * p.dv;
            params.oaccum_batch_stride = int64_t(p.seqlen_q) * p.dv;
            params.lseaccum_head_stride = p.total_q;
            params.lseaccum_split_stride = int64_t(p.h) * p.total_q;
            params.lseaccum_batch_stride = p.seqlen_q;
        }
    }
};

template <bool Paged, bool Varlen, int Splits>
void launch_mha_bf16_wi(Flash_fwd_params& p, hggcStream_t stream) {
    CHECK_CUDA((wi::launch_mha_wi<Paged, Varlen, Splits>(p, stream)));
#ifndef FLASHATTENTION_DISABLE_SPLIT
    if constexpr (Splits > 1)
        run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(p, stream, false);
#endif
}

template <bool Paged, bool Varlen>
void launch_mha_bf16_wi(Flash_fwd_params& p, int splits, hggcStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_SPLIT
    if (splits == 2) return launch_mha_bf16_wi<Paged, Varlen, 2>(p, stream);
#endif
    launch_mha_bf16_wi<Paged, Varlen, 1>(p, stream);
}

void launch_mha_bf16_wi_by_layout(Flash_fwd_params& p, int splits, hggcStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_PAGEDKV
    if (p.page_table) return launch_mha_bf16_wi<true, true>(p, splits, stream);
#endif
    if (p.cu_seqlens_q) return launch_mha_bf16_wi<false, true>(p, splits, stream);
    launch_mha_bf16_wi<false, false>(p, splits, stream);
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
    // Address pages with their capacity; use the separate work bound only
    // for this private copy's loop schedule. Device lengths remain authoritative.
    if (p.page_table) {
        int64_t const work_k = detail::mha_kv_workload_bound(p, max_seqlen_k);
        workspace.params.seqlen_k = int(std::min(int64_t(p.seqlen_k),
            ((work_k + wi::N - 1) / wi::N) * wi::N));
    }
    launch_mha_bf16_wi_by_layout(workspace.params, splits, stream);
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
#endif  // FLASHATTENTION_MHA_WI

#ifdef FLASHATTENTION_FWD_FASTPATHS
namespace flash {
bool try_run_mha_fast_path(Flash_fwd_params &p, hggcStream_t stream, int max_seqlen_k) {
#ifdef FLASHATTENTION_MHA_WI
    if (p.is_bf16) return try_run_mha_bf16_wi(p, stream, max_seqlen_k);
#ifdef FLASHATTENTION_MHA_FP8_FAST_PATH
    if (p.is_e4m3) return try_run_mha_fp8_wi(p, stream, max_seqlen_k);
#endif
#endif
    return false;
}
}  // namespace flash
#endif
