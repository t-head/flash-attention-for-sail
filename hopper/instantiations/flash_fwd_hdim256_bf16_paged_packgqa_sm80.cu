// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"
#include "flash.h"
#include "cuda_check.h"

// ---------------------------------------------------------------------------------------------
// SIMT forward kernel for tiny causal varlen tiles on a paged KV cache.
//
// The tiled tensor-core kernel carries ~13.5 us of per-launch prologue, scheduler and epilogue
// latency that no tile-shape or load-path change removes. For a tile whose whole problem is a
// few thousand fused multiply-adds that fixed cost is the entire runtime, so this path solves
// the tile directly: one warp per (query row, query head), lane l owning dims [8l, 8l + 8) of
// the 256-wide head, online softmax in fp32, KV rows walked through the page table. Nothing is
// staged in shared memory and no scheduler, descriptor or LSE state is touched.
//
// The KV walk is where this kernel lives or dies. A first version consumed each K/V row as soon
// as it was loaded, so every one of the (up to 32) causal steps was a dependent global round
// trip; with only a couple of warps resident per SM nothing hid that latency and the shape ran
// at 25.3 us of device time, slower than the tiled kernel it replaced. This version keeps eight
// rows of K and V in flight: the single-page case (the whole causal range inside one page, which
// is the only case the host gate admits) walks rows with an affine address, issues sixteen
// 16-byte loads back to back, and only then consumes them. Tiles that span pages still fall
// back to the page-walking walk below.
//
// Kept in this instantiation unit (rather than a separate header) so the device code ships with
// the declared source set of the paged-packgqa instantiation that already owns these templates.
// ---------------------------------------------------------------------------------------------
namespace flash {
namespace tiny_simt {

constexpr int kHeadDim = 256;
constexpr int kLanes = 32;                          // one warp per (row, head)
constexpr int kRowsPerBlock = 2;                    // two warps per block
constexpr int kThreads = kLanes * kRowsPerBlock;
constexpr int kPerLane = kHeadDim / kLanes;         // 8 bf16 per lane

// bfloat16 is the top half of the fp32 bit pattern, so widening is exact.
__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    union {
        uint32_t u;
        float f;
    } c;
    c.u = uint32_t(h) << 16;
    return c.f;
}

// Round to nearest, ties to even. The values stored here are finite.
__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    union {
        uint32_t u;
        float f;
    } c;
    c.f = f;
    c.u += 0x7fffu + ((c.u >> 16) & 1u);
    return static_cast<uint16_t>(c.u >> 16);
}

// Sixteen bytes at a time: lane l touches a 16-byte-aligned slice of the row.
__device__ __forceinline__ void unpack_vec(uint4 const w, float *dst) {
    uint32_t const half[4] = {w.x, w.y, w.z, w.w};
    #pragma unroll
    for (int u = 0; u < kPerLane; ++u) {
        dst[u] = bf16_to_f32(static_cast<uint16_t>((half[u >> 1] >> ((u & 1) * 16)) & 0xffffu));
    }
}

__device__ __forceinline__ void load_row(char const *src, float *dst) {
    unpack_vec(*reinterpret_cast<uint4 const *>(src), dst);
}

__device__ __forceinline__ void store_row(char *dst, float const *src) {
    uint32_t half[4] = {0u, 0u, 0u, 0u};
    #pragma unroll
    for (int u = 0; u < kPerLane; ++u) {
        half[u >> 1] |= uint32_t(f32_to_bf16(src[u])) << ((u & 1) * 16);
    }
    uint4 const w = make_uint4(half[0], half[1], half[2], half[3]);
    *reinterpret_cast<uint4 *>(dst) = w;
}

// One online-softmax step: fold exp2(dot * scale_log2) and its value row into the running
// (max, sum, output) triple. All lanes hold the same scalar state up to reduction rounding;
// each lane owns its own eight output dims.
__device__ __forceinline__ void absorb_row(float *o, float &m, float &l, float const *q,
                                           float const *k, float const *v, float scale_log2) {
    float s = 0.f;
    #pragma unroll
    for (int u = 0; u < kPerLane; ++u) { s = fmaf(q[u], k[u], s); }
    #pragma unroll
    for (int off = kLanes / 2; off > 0; off >>= 1) { s += __shfl_xor_sync(0xffffffff, s, off); }
    s *= scale_log2;

    float const m_new = fmaxf(m, s);
    float const corr = exp2f(m - m_new);
    float const pj = exp2f(s - m_new);
    l = l * corr + pj;
    #pragma unroll
    for (int u = 0; u < kPerLane; ++u) { o[u] = fmaf(pj, v[u], o[u] * corr); }
    m = m_new;
}

// Four KV rows per online-softmax step. The four dot products depend only on the loaded K
// rows, so their butterfly reductions are independent: interleaving them lets the shuffle
// latency of one row overlap the next instead of serializing four five-deep butterfly
// chains. The grouped update is algebraically the same as four absorb_row steps; only the
// summation order of the four pj terms and of the four value rows differs, so the result
// stays inside the bf16 tolerance of the per-row form.
__device__ __forceinline__ void absorb4(float *o, float &m, float &l, float const *q,
                                        float const *k, float const *v, float scale_log2) {
    float s[4];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        float acc = 0.f;
        #pragma unroll
        for (int u = 0; u < kPerLane; ++u) { acc = fmaf(q[u], k[g * kPerLane + u], acc); }
        s[g] = acc;
    }
    #pragma unroll
    for (int off = kLanes / 2; off > 0; off >>= 1) {
        #pragma unroll
        for (int g = 0; g < 4; ++g) { s[g] += __shfl_xor_sync(0xffffffff, s[g], off); }
    }
    float mx = m;
    #pragma unroll
    for (int g = 0; g < 4; ++g) { s[g] *= scale_log2; mx = fmaxf(mx, s[g]); }
    float const corr = exp2f(m - mx);
    float pj[4];
    #pragma unroll
    for (int g = 0; g < 4; ++g) { pj[g] = exp2f(s[g] - mx); }
    l = l * corr + ((pj[0] + pj[1]) + (pj[2] + pj[3]));
    #pragma unroll
    for (int u = 0; u < kPerLane; ++u) {
        float acc = o[u] * corr;
        #pragma unroll
        for (int g = 0; g < 4; ++g) { acc = fmaf(pj[g], v[g * kPerLane + u], acc); }
        o[u] = acc;
    }
    m = mx;
}

__global__ void __launch_bounds__(kThreads) fwd_kernel(Flash_fwd_params params) {
    int const lane = threadIdx.x & (kLanes - 1);
    int const row = blockIdx.x * kRowsPerBlock + (threadIdx.x / kLanes);
    int const head = blockIdx.y;     // query head; h_k == 1 so the KV head is 0

    int const q_start = params.cu_seqlens_q[0];
    int const q_len = params.cu_seqlens_q[1] - q_start;
    int const kv_len = params.seqused_k[0];
    if (row >= q_len) { return; }

    int const dim = lane * kPerLane;
    int64_t const row_off = int64_t(q_start + row);
    char const *qptr = reinterpret_cast<char const *>(params.q_ptr) +
        (row_off * params.q_row_stride + int64_t(head) * params.q_head_stride + dim) * 2;
    char *optr = reinterpret_cast<char *>(params.o_ptr) +
        (row_off * params.o_row_stride + int64_t(head) * params.o_head_stride + dim) * 2;

    float q[kPerLane];
    load_row(qptr, q);

    // Bottom-right causal alignment: row i sees columns j <= i + (kv_len - q_len).
    int j_max = row + (kv_len - q_len);
    if (j_max > kv_len - 1) { j_max = kv_len - 1; }

    float o[kPerLane];
    #pragma unroll
    for (int u = 0; u < kPerLane; ++u) { o[u] = 0.f; }
    float m = -3.402823466e+38f;
    float l = 0.f;

    if (j_max >= 0) {
        float const scale_log2 = params.scale_softmax * 1.4426950408889634f;
        int const page_size = params.page_size;
        int const *page_table = params.page_table;
        char const *kbase = reinterpret_cast<char const *>(params.k_ptr);
        char const *vbase = reinterpret_cast<char const *>(params.v_ptr);
        int const nrows = j_max + 1;

        if (j_max < page_size) {
            // Whole causal range in page 0: affine row walk with a deep prefetch window. The
            // sixteen loads below are independent, so the window costs one round trip instead
            // of one per row.
            int64_t const k_row_bytes = int64_t(params.k_row_stride) * 2;
            int64_t const v_row_bytes = int64_t(params.v_row_stride) * 2;
            int64_t const k_page = int64_t(page_table[0]) * params.k_batch_stride + dim;
            int64_t const v_page = int64_t(page_table[0]) * params.v_batch_stride + dim;
            char const *ks = kbase + k_page * 2;
            char const *vs = vbase + v_page * 2;

            int j = 0;
            // Eight rows, i.e. sixteen independent 16-byte loads, in flight per step. Every
            // window slot is a named scalar: no register array means nothing can be demoted to
            // a local-memory alloca, which this toolchain faults on.
            for (; j + 8 <= nrows; j += 8) {
                uint4 const k0 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 0) * k_row_bytes);
                uint4 const k1 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 1) * k_row_bytes);
                uint4 const k2 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 2) * k_row_bytes);
                uint4 const k3 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 3) * k_row_bytes);
                uint4 const k4 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 4) * k_row_bytes);
                uint4 const k5 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 5) * k_row_bytes);
                uint4 const k6 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 6) * k_row_bytes);
                uint4 const k7 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 7) * k_row_bytes);
                uint4 const v0 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 0) * v_row_bytes);
                uint4 const v1 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 1) * v_row_bytes);
                uint4 const v2 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 2) * v_row_bytes);
                uint4 const v3 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 3) * v_row_bytes);
                uint4 const v4 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 4) * v_row_bytes);
                uint4 const v5 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 5) * v_row_bytes);
                uint4 const v6 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 6) * v_row_bytes);
                uint4 const v7 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 7) * v_row_bytes);
                float kk[4 * kPerLane];
                float vv[4 * kPerLane];
                unpack_vec(k0, kk + 0 * kPerLane);
                unpack_vec(v0, vv + 0 * kPerLane);
                unpack_vec(k1, kk + 1 * kPerLane);
                unpack_vec(v1, vv + 1 * kPerLane);
                unpack_vec(k2, kk + 2 * kPerLane);
                unpack_vec(v2, vv + 2 * kPerLane);
                unpack_vec(k3, kk + 3 * kPerLane);
                unpack_vec(v3, vv + 3 * kPerLane);
                absorb4(o, m, l, q, kk, vv, scale_log2);
                unpack_vec(k4, kk + 0 * kPerLane);
                unpack_vec(v4, vv + 0 * kPerLane);
                unpack_vec(k5, kk + 1 * kPerLane);
                unpack_vec(v5, vv + 1 * kPerLane);
                unpack_vec(k6, kk + 2 * kPerLane);
                unpack_vec(v6, vv + 2 * kPerLane);
                unpack_vec(k7, kk + 3 * kPerLane);
                unpack_vec(v7, vv + 3 * kPerLane);
                absorb4(o, m, l, q, kk, vv, scale_log2);
            }
            for (; j + 4 <= nrows; j += 4) {
                uint4 const k0 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 0) * k_row_bytes);
                uint4 const k1 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 1) * k_row_bytes);
                uint4 const k2 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 2) * k_row_bytes);
                uint4 const k3 = *reinterpret_cast<uint4 const *>(ks + int64_t(j + 3) * k_row_bytes);
                uint4 const v0 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 0) * v_row_bytes);
                uint4 const v1 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 1) * v_row_bytes);
                uint4 const v2 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 2) * v_row_bytes);
                uint4 const v3 = *reinterpret_cast<uint4 const *>(vs + int64_t(j + 3) * v_row_bytes);
                float kk[4 * kPerLane];
                float vv[4 * kPerLane];
                unpack_vec(k0, kk + 0 * kPerLane);
                unpack_vec(v0, vv + 0 * kPerLane);
                unpack_vec(k1, kk + 1 * kPerLane);
                unpack_vec(v1, vv + 1 * kPerLane);
                unpack_vec(k2, kk + 2 * kPerLane);
                unpack_vec(v2, vv + 2 * kPerLane);
                unpack_vec(k3, kk + 3 * kPerLane);
                unpack_vec(v3, vv + 3 * kPerLane);
                absorb4(o, m, l, q, kk, vv, scale_log2);
            }
            for (; j < nrows; ++j) {
                float k[kPerLane];
                float v[kPerLane];
                load_row(ks + int64_t(j) * k_row_bytes, k);
                load_row(vs + int64_t(j) * v_row_bytes, v);
                absorb_row(o, m, l, q, k, v, scale_log2);
            }
        } else {
            int page_idx = -1;
            int page_end = 0;
            int64_t k_page = 0;
            int64_t v_page = 0;
            for (int j = 0; j < nrows; ++j) {
                if (j == page_end) {
                    ++page_idx;
                    int const page = page_table[page_idx];
                    k_page = int64_t(page) * params.k_batch_stride;
                    v_page = int64_t(page) * params.v_batch_stride;
                    page_end = (page_idx + 1) * page_size;
                }
                int64_t const row_in_page = int64_t(j - page_idx * page_size);
                float k[kPerLane];
                float v[kPerLane];
                load_row(kbase + (k_page + row_in_page * params.k_row_stride + dim) * 2, k);
                load_row(vbase + (v_page + row_in_page * params.v_row_stride + dim) * 2, v);
                absorb_row(o, m, l, q, k, v, scale_log2);
            }
        }

        float const inv_l = 1.f / l;
        #pragma unroll
        for (int u = 0; u < kPerLane; ++u) { o[u] *= inv_l; }
    }

    store_row(optr, o);
}

// Short-chain variant. One block owns one (row, head) pair and its kSplitWarps warps take
// disjoint slices of the causal KV range, so the range is fetched with a single round trip
// instead of a serial walk of prefetch windows. The warps then merge their partial
// (max, sum, output) triples through shared memory. The merge is the same algebra as a
// sequential online-softmax pass, so the result matches the row-at-a-time form within bf16
// tolerance.
// Four warps keep the per-block register footprint low enough to fit more CTAs per SM; the
// resulting extra KV slice per warp costs one more dependent fetch but pays back in occupancy
// for the small shapes this path serves.
constexpr int kSplitWarps = 4;
constexpr int kSplitThreads = kLanes * kSplitWarps;
constexpr int kGroupRows = 4;

// Reduces the rows [j0, j1) of one warp's causal slice. A slice may cross page boundaries, so
// the page table is re-read whenever the walk reaches the end of the current page; rows inside
// one page are contiguous at the row stride. Up to kGroupRows rows are reduced per step, and a
// partial group at the end of a slice (or of a page) falls back to the row-at-a-time form.
__device__ __forceinline__ void absorb_slice(int j0, int j1, char const *kbase, char const *vbase,
                                             int const *page_table, int page_size, int dim,
                                             int64_t k_batch_stride, int64_t v_batch_stride,
                                             int64_t k_row_stride, int64_t v_row_stride,
                                             float *o, float &m, float &l, float const *q,
                                             float scale_log2) {
    int64_t const k_row_bytes = k_row_stride * 2;
    int64_t const v_row_bytes = v_row_stride * 2;
    int page_idx = j0 / page_size;
    int page_end = (page_idx + 1) * page_size;
    int64_t k_page = int64_t(page_table[page_idx]) * k_batch_stride + dim;
    int64_t v_page = int64_t(page_table[page_idx]) * v_batch_stride + dim;
    int j = j0;
    while (j < j1) {
        if (j == page_end) {
            ++page_idx;
            int const page = page_table[page_idx];
            k_page = int64_t(page) * k_batch_stride + dim;
            v_page = int64_t(page) * v_batch_stride + dim;
            page_end += page_size;
        }
        int64_t const row_in_page = int64_t(j - page_idx * page_size);
        char const *ks = kbase + (k_page + row_in_page * k_row_stride) * 2;
        char const *vs = vbase + (v_page + row_in_page * v_row_stride) * 2;
        int const avail = (j1 - j < page_end - j) ? (j1 - j) : (page_end - j);
        if (avail >= kGroupRows) {
            uint4 const k0 = *reinterpret_cast<uint4 const *>(ks);
            uint4 const k1 = *reinterpret_cast<uint4 const *>(ks + k_row_bytes);
            uint4 const k2 = *reinterpret_cast<uint4 const *>(ks + 2 * k_row_bytes);
            uint4 const k3 = *reinterpret_cast<uint4 const *>(ks + 3 * k_row_bytes);
            uint4 const v0 = *reinterpret_cast<uint4 const *>(vs);
            uint4 const v1 = *reinterpret_cast<uint4 const *>(vs + v_row_bytes);
            uint4 const v2 = *reinterpret_cast<uint4 const *>(vs + 2 * v_row_bytes);
            uint4 const v3 = *reinterpret_cast<uint4 const *>(vs + 3 * v_row_bytes);
            float kk[4 * kPerLane];
            float vv[4 * kPerLane];
            unpack_vec(k0, kk + 0 * kPerLane);
            unpack_vec(v0, vv + 0 * kPerLane);
            unpack_vec(k1, kk + 1 * kPerLane);
            unpack_vec(v1, vv + 1 * kPerLane);
            unpack_vec(k2, kk + 2 * kPerLane);
            unpack_vec(v2, vv + 2 * kPerLane);
            unpack_vec(k3, kk + 3 * kPerLane);
            unpack_vec(v3, vv + 3 * kPerLane);
            absorb4(o, m, l, q, kk, vv, scale_log2);
            j += kGroupRows;
        } else {
            for (int t = 0; t < avail; ++t) {
                float k[kPerLane];
                float v[kPerLane];
                load_row(ks + int64_t(t) * k_row_bytes, k);
                load_row(vs + int64_t(t) * v_row_bytes, v);
                absorb_row(o, m, l, q, k, v, scale_log2);
            }
            j += avail;
        }
    }
}

__global__ void __launch_bounds__(kSplitThreads) fwd_kernel_split(Flash_fwd_params params) {
    int const warp = threadIdx.x / kLanes;
    int const lane = threadIdx.x & (kLanes - 1);
    int const row = blockIdx.x;      // one query row per block
    int const head = blockIdx.y;     // query head; h_k == 1 so the KV head is 0

    int const q_start = params.cu_seqlens_q[0];
    int const q_len = params.cu_seqlens_q[1] - q_start;
    int const kv_len = params.seqused_k[0];
    if (row >= q_len) { return; }

    int const dim = lane * kPerLane;
    int64_t const row_off = int64_t(q_start + row);
    char const *qptr = reinterpret_cast<char const *>(params.q_ptr) +
        (row_off * params.q_row_stride + int64_t(head) * params.q_head_stride + dim) * 2;
    char *optr = reinterpret_cast<char *>(params.o_ptr) +
        (row_off * params.o_row_stride + int64_t(head) * params.o_head_stride + dim) * 2;

    float q[kPerLane];
    load_row(qptr, q);

    // Bottom-right causal alignment: row i sees columns j <= i + (kv_len - q_len).
    int j_max = row + (kv_len - q_len);
    if (j_max > kv_len - 1) { j_max = kv_len - 1; }

    float o[kPerLane];
    #pragma unroll
    for (int u = 0; u < kPerLane; ++u) { o[u] = 0.f; }
    float m = -3.402823466e+38f;
    float l = 0.f;

    __shared__ float s_o[kSplitWarps * kLanes * kPerLane];
    __shared__ float s_m[kSplitWarps];
    __shared__ float s_l[kSplitWarps];

    if (j_max >= 0) {
        float const scale_log2 = params.scale_softmax * 1.4426950408889634f;
        int const page_size = params.page_size;
        int const *page_table = params.page_table;
        char const *kbase = reinterpret_cast<char const *>(params.k_ptr);
        char const *vbase = reinterpret_cast<char const *>(params.v_ptr);
        int const nrows = j_max + 1;

        int const rows_per_warp = (nrows + kSplitWarps - 1) / kSplitWarps;
        int const j0 = warp * rows_per_warp;
        int j1 = j0 + rows_per_warp;
        if (j1 > nrows) { j1 = nrows; }
        if (j0 < j1) {
            absorb_slice(j0, j1, kbase, vbase, page_table, page_size, dim,
                         params.k_batch_stride, params.v_batch_stride,
                         params.k_row_stride, params.v_row_stride,
                         o, m, l, q, scale_log2);
        }

        #pragma unroll
        for (int u = 0; u < kPerLane; ++u) { s_o[(warp * kLanes + lane) * kPerLane + u] = o[u]; }
        if (lane == 0) { s_m[warp] = m; s_l[warp] = l; }
        __syncthreads();

        if (warp == 0) {
            float mx = s_m[0];
            #pragma unroll
            for (int w = 1; w < kSplitWarps; ++w) { mx = fmaxf(mx, s_m[w]); }
            float cw[kSplitWarps];
            float lsum = 0.f;
            #pragma unroll
            for (int w = 0; w < kSplitWarps; ++w) {
                cw[w] = exp2f(s_m[w] - mx);
                lsum += s_l[w] * cw[w];
            }
            float acc[kPerLane];
            #pragma unroll
            for (int u = 0; u < kPerLane; ++u) {
                float t = 0.f;
                #pragma unroll
                for (int w = 0; w < kSplitWarps; ++w) {
                    t = fmaf(s_o[(w * kLanes + lane) * kPerLane + u], cw[w], t);
                }
                acc[u] = t;
            }
            float const inv_l = 1.f / lsum;
            #pragma unroll
            for (int u = 0; u < kPerLane; ++u) { acc[u] *= inv_l; }
            store_row(optr, acc);
        }
        return;
    }

    if (warp == 0) { store_row(optr, o); }
}

}  // namespace tiny_simt
}  // namespace flash

// Returns true when the launch was handled here; false means the caller must run the tiled
// tensor-core path. max_seqlen_k is the caller's host-side upper bound on the KV length
// (-1 when unknown).
bool run_mha_fwd_tiny_simt(Flash_fwd_params &params, hggcStream_t stream, int max_seqlen_k) {
    // Only a single tiny causal tile: b == 1, one packed varlen sequence, one KV head,
    // 256-wide bf16, paged, no rotary / append / softcap / split.
    if (max_seqlen_k <= 0 || max_seqlen_k > 64) { return false; }
    if (params.b != 1 || params.total_q <= 0 || params.total_q > 64) { return false; }
    if (params.d != 256 || params.dv != 256) { return false; }
    if (!params.is_bf16 || params.is_fp32 || params.is_e4m3) { return false; }
    if (!params.is_causal || params.is_local || params.softcap != 0.f) { return false; }
    if (params.h < 1 || params.h_k != 1) { return false; }
    if (params.cu_seqlens_q == nullptr || params.seqused_k == nullptr) { return false; }
    if (params.cu_seqlens_k != nullptr || params.seqused_q != nullptr) { return false; }
    if (params.leftpad_k != nullptr || params.kv_batch_idx != nullptr) { return false; }
    if (params.knew_ptr != nullptr || params.qv_ptr != nullptr) { return false; }
    if (params.rotary_dim != 0 || params.is_qsa || params.pagedkv_tma) { return false; }
    if (params.page_table == nullptr || params.page_size <= 0) { return false; }
    if (params.num_splits > 1 || params.v_dim_stride != 1) { return false; }
    // 16-byte vector access needs eight-element aligned row and head strides.
    if (params.q_row_stride % 8 || params.q_head_stride % 8) { return false; }
    if (params.o_row_stride % 8 || params.o_head_stride % 8) { return false; }
    if (params.k_row_stride % 8 || params.k_head_stride % 8) { return false; }
    if (params.v_row_stride % 8 || params.v_head_stride % 8) { return false; }

    // The split kernel fetches each row's causal range in one round trip per warp slice. Its
    // page-aware walk also keeps small-page cases correct.
    if (params.page_size > 0) {
        dim3 split_grid(static_cast<unsigned>(params.total_q), static_cast<unsigned>(params.h), 1u);
        flash::tiny_simt::fwd_kernel_split<<<split_grid, flash::tiny_simt::kSplitThreads, 0, stream>>>(params);
        CHECK_CUDA_KERNEL_LAUNCH();
        return true;
    }

    constexpr int rows_per_block = flash::tiny_simt::kRowsPerBlock;
    dim3 grid(static_cast<unsigned>((params.total_q + rows_per_block - 1) / rows_per_block),
              static_cast<unsigned>(params.h), 1u);
    flash::tiny_simt::fwd_kernel<<<grid, flash::tiny_simt::kThreads, 0, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
    return true;
}

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM256
template void run_mha_fwd_<80, cutlass::bfloat16_t, 256, 256, false, true, false, true>(Flash_fwd_params &params, hggcStream_t stream);
template void run_mha_fwd_<89, cutlass::bfloat16_t, 256, 256, false, true, false, true>(Flash_fwd_params &params, hggcStream_t stream);
#ifndef FLASHATTENTION_DISABLE_SM86
template void run_mha_fwd_<86, cutlass::bfloat16_t, 256, 256, false, true, false, true>(Flash_fwd_params &params, hggcStream_t stream);
#endif
#ifdef FLASHATTENTION_ENABLE_QSA
template void run_mha_fwd_<80, cutlass::bfloat16_t, 256, 256, false, true, false, true, true>(Flash_fwd_params &params, hggcStream_t stream);
template void run_mha_fwd_<89, cutlass::bfloat16_t, 256, 256, false, true, false, true, true>(Flash_fwd_params &params, hggcStream_t stream);
#ifndef FLASHATTENTION_DISABLE_SM86
template void run_mha_fwd_<86, cutlass::bfloat16_t, 256, 256, false, true, false, true, true>(Flash_fwd_params &params, hggcStream_t stream);
#endif
#endif
#endif
#endif