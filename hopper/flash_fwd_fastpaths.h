// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

// Eligibility and entry points for selected forward fast paths:
// BF16 and FP8 WI, including fixed, packed and page64 KV layouts.

#pragma once

#include <hggc_runtime.h>
#include "flash.h"

#if defined(FLASHATTENTION_FWD_FASTPATHS) && !defined(FLASHATTENTION_DISABLE_HDIM192)
#define FLASHATTENTION_MHA_WI
#endif
#if defined(FLASHATTENTION_MHA_WI) && !defined(FLASHATTENTION_DISABLE_FP8)
#define FLASHATTENTION_MHA_FP8_FAST_PATH
#endif

#ifdef FLASHATTENTION_FWD_FASTPATHS
#include <climits>
#include <cmath>
#include <algorithm>
#endif

namespace flash {

#ifdef FLASHATTENTION_FWD_FASTPATHS
// Pure host policies for the individual strategies used by the runtime dtype dispatch
// below. Each supports_* function describes its strategy's complete contract;
// none allocates memory or launches work.
namespace detail {
constexpr int MhaWiQueryRows = 256;
constexpr int MhaWiKeyColumns = 128;
// Uniform WI batches use grid.y; V packing uses grid.z.
constexpr int MhaMaxGridYZ = 65535;

// The paged API keeps seqlen_k as storage capacity for native addressing.
// A caller-provided maximum is a work bound, not a measured per-sequence length.
// Without it, capacity alone cannot establish that a fast path is profitable.
inline int mha_kv_workload_bound(Flash_fwd_params const& p, int max_seqlen_k) {
    return p.page_table ? (max_seqlen_k > 0 ? std::min(max_seqlen_k, p.seqlen_k) : 0)
                        : p.seqlen_k;
}

// Bound the sum of rounded tiles without reading device sequence lengths.
// For nonempty sequences, ceil(q / rows) <= (q + rows - 1) / rows.
// Ragged packed launches use this bound and map tiles from device sequence
// lengths, so empty batch slots do not expand the grid to batch * maxQ.
inline int64_t mha_query_tiles_upper_bound(Flash_fwd_params const& p, int rows) {
    int64_t const padded = ((int64_t(p.seqlen_q) + rows - 1) / rows) * p.b;
    int64_t const total = p.total_q;
    int64_t const nonempty = std::min(int64_t(p.b), total);
    return std::min({padded, total, (total + nonempty * (rows - 1)) / rows});
}

// Uniform batches retain the original two-dimensional launch. Packed ragged
// queries use a compact grid shared by contiguous and page64 KV layouts.
inline bool mha_wi_compact_grid(Flash_fwd_params const& p) {
    return p.cu_seqlens_q && int64_t(p.total_q) < int64_t(p.b) * p.seqlen_q;
}

// WI eligibility validates grid/index bounds before estimating partitions.
inline int mha_wi_num_splits(Flash_fwd_params const& p, int max_seqlen_k = -1) {
#ifdef FLASHATTENTION_DISABLE_SPLIT
    return 1;
#else
    int64_t const blocks = mha_query_tiles_upper_bound(p, MhaWiQueryRows) * p.h;
    int64_t const waves = (blocks + p.num_sm - 1) / p.num_sm;
    // Splitting KV preserves all four interleaved compute groups. Only pay
    // for partial buffers and merge when wave waste is material and each
    // partition retains enough KV work to amortize its pipeline startup.
    double const useful_k = mha_kv_workload_bound(p, max_seqlen_k)
        - (double(p.total_q) / p.b - 1.0) * 0.5;
    constexpr int MinPartitionKeyTiles = 128;
    constexpr double MinWaveUtilization = 0.90;
    return blocks < MinWaveUtilization * waves * p.num_sm
        && (2 * blocks + p.num_sm - 1) / p.num_sm < 2 * waves
        && useful_k >= 2 * MinPartitionKeyTiles * MhaWiKeyColumns ? 2 : 1;
#endif
}

inline bool mha_qkv_layout_supported(Flash_fwd_params const& p) {
    for (auto ptr : {p.q_ptr, p.k_ptr, p.v_ptr}) {
        if (!ptr || (reinterpret_cast<uintptr_t>(ptr) & 127)) return false;
    }
    for (auto stride : {p.q_row_stride, p.k_row_stride, p.v_row_stride,
                        p.q_head_stride, p.k_head_stride, p.v_head_stride}) {
        if (stride <= 0 || stride > INT32_MAX || stride % 64) return false;
    }
    return true;
}

// Packed varlen and page64 KV contracts shared by BF16 and FP8 WI.
inline bool mha_kv_layout_supported(Flash_fwd_params const& p) {
    if (!p.page_table) return p.cu_seqlens_k && !p.seqused_k;
#ifdef FLASHATTENTION_DISABLE_PAGEDKV
    return false;
#else
    return p.page_size == 64 && !p.cu_seqlens_k && p.seqused_k
        && p.page_table_batch_stride > 0 && p.k_batch_stride > 0 && p.v_batch_stride > 0
        && p.k_batch_stride % 64 == 0 && p.v_batch_stride % 64 == 0;
#endif
}

// Fixed Q/K/V or packed Q with packed/page64 KV. Fixed batches must
// preserve the AIU alignment of each sequence base, including broadcast batches.
inline bool mha_sequence_layout_supported(Flash_fwd_params const& p) {
    if (p.cu_seqlens_q) return mha_kv_layout_supported(p);
    if (p.cu_seqlens_k || p.page_table || p.seqused_k) return false;
    for (auto stride : {p.q_batch_stride, p.k_batch_stride, p.v_batch_stride}) {
        if (stride < 0 || stride % 64) return false;
    }
    return true;
}

// BF16 WI layout and semantic requirements; profitability is checked below.
inline bool mha_bf16_wi_supported(Flash_fwd_params const& p) {
    if (p.arch != 89 || p.num_sm != 39 || !p.is_bf16 || p.is_e4m3 || p.is_fp32 || p.is_qsa
        || p.d != 192 || p.dv != 128 || p.b <= 0 || p.b > MhaMaxGridYZ || p.h != p.h_k
        || p.h <= 0 || p.num_splits != 1 || p.pack_gqa
        || !mha_sequence_layout_supported(p)
        || p.seqused_q || p.leftpad_k || p.kv_batch_idx
        || p.knew_ptr || p.vnew_ptr || p.qv_ptr
        || p.rotary_cos_ptr || p.rotary_sin_ptr || p.s_aux_ptr
        || !p.is_causal || p.is_local || p.window_size_left != p.seqlen_k - 1 || p.attention_chunk
        || p.softcap != 0.f || p.p_dropout != 1.f
        || p.v_dim_stride != 1 || !p.softmax_lse_ptr
        || !(p.scale_softmax > 0.f) || !std::isfinite(p.scale_softmax)) return false;
    if (p.window_size_right != 0) return false;
    return mha_qkv_layout_supported(p) && p.o_ptr && p.o_row_stride > 0 && p.o_head_stride > 0;
}

#if !defined(FLASHATTENTION_DISABLE_FP8)
// Amortize the extra FP8 V pack over useful 64-row query groups. A packed V
// page is shared by all query tiles and GQA heads. Use the same bottom-right
// causal reuse estimate for dense and paged WI.
inline bool mha_v_prepack_reuse_supported(Flash_fwd_params const& p, int min_waves, int max_seqlen_k) {
    int const work_k = mha_kv_workload_bound(p, max_seqlen_k);
    if (p.b <= 0 || p.h_k <= 0 || p.num_sm <= 0 || work_k <= 0) return false;
    constexpr int QueryRowsPerTile = 64;
    // A long query must not hide the packing cost paid for a shorter batch
    // member. Every query has at least total_q - (b - 1) * max_q rows;
    // uniform batches retain their exact length without reading device lengths.
    int64_t const min_q = std::max(int64_t(0), int64_t(p.total_q)
        - int64_t(p.b - 1) * p.seqlen_q);
    if (min_q == 0) return false;
    double const average_k = std::max(0.0, double(work_k) - (double(min_q) - 1.0) * 0.5);
    double const reuse_tiles = double(min_q) / QueryRowsPerTile * (double(p.h) / p.h_k)
        * (average_k / work_k);
    double const min_reuse = double(min_waves) * p.num_sm;
    if (reuse_tiles < min_reuse) return false;
    // Calibrated steady-state budget: reuse and causal loop length jointly
    // amortize packing and WI startup. An independent minimum KV length
    // rejects profitable long-query self-attention and shorter prefix loops.
    constexpr int SteadyStateKeyTileBudget = 56;
    return reuse_tiles * (average_k / MhaWiKeyColumns)
        >= min_reuse * SteadyStateKeyTileBudget;
}
#endif
}  // namespace detail

#ifdef FLASHATTENTION_MHA_WI
inline bool supports_mha_bf16_wi(Flash_fwd_params const& p, int max_seqlen_k = -1) {
    // Enough useful query blocks are required to fill the device. Count
    // actual rows so empty/ragged batch slots do not inflate the estimate.
    constexpr double MinWaveUtilization = 0.70;
    double const useful_blocks = double(p.total_q) * p.h / detail::MhaWiQueryRows;
    if (!detail::mha_bf16_wi_supported(p)) return false;
    // Shape-independent safety bounds: the loop's Q/K sum and launch grid
    // use 32-bit indices. Performance is gated by useful work, not head count.
    int64_t const query_blocks = (int64_t(p.seqlen_q) + detail::MhaWiQueryRows - 1)
        / detail::MhaWiQueryRows;
    int64_t const grid_tiles = detail::mha_wi_compact_grid(p)
        ? detail::mha_query_tiles_upper_bound(p, detail::MhaWiQueryRows) : query_blocks;
    if (p.seqlen_q < detail::MhaWiQueryRows || p.seqlen_k < p.seqlen_q
        || int64_t(p.seqlen_q) + p.seqlen_k + detail::MhaWiQueryRows > INT_MAX
        || grid_tiles * p.h > INT_MAX) return false;
    // Count rounded launch tiles as well as useful rows. This includes the
    // wasted query lanes of a partial tile and the last incomplete CU wave.
    double const work_blocks = double(detail::mha_query_tiles_upper_bound(p, detail::MhaWiQueryRows)) * p.h;
    double const query_waves = std::ceil(work_blocks / p.num_sm);
    // Short KV cannot amortize the three-stage pipeline. Estimate the
    // bottom-right causal work using actual query rows, as for V prepacking.
    // Page gathers need more steady-state work than contiguous K/V.
    constexpr int MinDenseKeyTiles = 24, MinPagedKeyTiles = 32;
    double min_key_tiles = p.page_table ? MinPagedKeyTiles : MinDenseKeyTiles;
    // M64 gives up steady-state throughput for smaller query tiles. With
    // several query waves, calibrated average KV work can amortize WI sooner.
    constexpr int MinQueryWaves = 4;
    constexpr double MinM64KeyTiles = 10.75;
    bool const m64_steady = p.page_table && !p.use_kblockm_128 && query_waves >= MinQueryWaves;
    if (m64_steady) min_key_tiles = MinM64KeyTiles;
    double const average_q = double(p.total_q) / p.b;
    int const work_k = detail::mha_kv_workload_bound(p, max_seqlen_k);
    if (work_k <= 0) return false;
    double const useful_k = p.is_causal ? work_k - (average_q - 1.0) * 0.5 : work_k;
    // Compare wave padding with the native M64/M128 schedule, rather than
    // only checking WI's own utilization. The qualified native kernels keep
    // 256 query rows resident per CU (four M64 or two M128 blocks). Account
    // for internal KV partitions rather than assuming a long KV hides tails.
    int const native_rows = p.use_kblockm_128 ? 128 : 64;
    bool const uniform_q = int64_t(p.total_q) == int64_t(p.b) * p.seqlen_q;
    // A ragged native scheduler visits actual query tiles. Its lower bound
    // must not be inflated to batch * maxQ when estimating relative waves.
    int64_t const native_tiles = uniform_q
        ? ((int64_t(p.seqlen_q) + native_rows - 1) / native_rows) * p.b
        : (int64_t(p.total_q) + native_rows - 1) / native_rows;
    double const native_waves = std::ceil(double(native_tiles) * p.h
        / (p.num_sm * (detail::MhaWiQueryRows / native_rows)));
    // Use the same partition decision as launch. A longer KV loop alone
    // does not remove the extra wave; two KV partitions can reduce its cost.
    int const splits = detail::mha_wi_num_splits(p, max_seqlen_k);
    double const scheduled_waves = std::ceil(work_blocks * splits / p.num_sm) / splits;
    constexpr double MaxPagedWaveExpansion = 1.20;
    // A short M64 loop cannot absorb an extra WI wave even if it has enough
    // arithmetic in aggregate. Long loops retain the qualified tail margin.
    double const max_wave_expansion = m64_steady && useful_k < 16 * detail::MhaWiKeyColumns
        ? 1.0 : MaxPagedWaveExpansion;
    bool const wave_tail_supported = !p.page_table
        || scheduled_waves <= max_wave_expansion * native_waves;
    return wave_tail_supported && useful_k >= min_key_tiles * detail::MhaWiKeyColumns
        && useful_blocks >= p.num_sm
        && useful_blocks >= MinWaveUtilization * query_waves * p.num_sm;
}

#if !defined(FLASHATTENTION_DISABLE_FP8)
inline bool supports_mha_fp8_wi(Flash_fwd_params const &p, int max_seqlen_k = -1) {
    // Preserve the generic N=128 FP8 rounding by reusing its tile decision.
    // Optional/broadcast descales use the same pointer and stride rules as FA3.
    // Fixed, packed varlen and page64 MHA share a V buffer with a region per
    // sequence. Preserve the generic implementation for short-Q N64.
    if (p.arch != 89 || !p.is_e4m3 || p.is_fp32 || p.is_qsa ||
        !p.use_kblockm_128 || p.use_kblockm_16 || p.d != 192 || p.dv != 128 ||
        p.b <= 0 || p.b > detail::MhaMaxGridYZ || p.h_k != p.h || p.pack_gqa || !detail::mha_sequence_layout_supported(p) ||
        p.seqused_q || p.leftpad_k || p.kv_batch_idx ||
        p.num_splits != 1 || p.knew_ptr || p.vnew_ptr || p.qv_ptr || p.rotary_cos_ptr ||
        p.rotary_sin_ptr || p.s_aux_ptr || p.is_local || p.attention_chunk ||
        !p.is_causal || p.window_size_left != p.seqlen_k - 1 || p.window_size_right != 0 ||
        p.softcap != 0.f || p.p_dropout != 1.f || p.v_dim_stride != 1 ||
        !p.softmax_lse_ptr || !p.o_ptr || p.o_row_stride <= 0 || p.o_head_stride <= 0 ||
        p.seqlen_k < p.seqlen_q || p.seqlen_q <= 0 || p.seqlen_q > INT_MAX - 255 ||
        p.seqlen_k > INT_MAX - 127 || p.seqlen_q > INT_MAX - p.seqlen_k - 255 ||
        !(p.scale_softmax > 0.f) || !std::isfinite(p.scale_softmax)) {
        return false;
    }
    // Each WI group handles 64 query rows. Both dense and paged KV pay for
    // V packing; require the same useful reuse before selecting either path.
    if (!detail::mha_v_prepack_reuse_supported(p, 2, max_seqlen_k)) return false;
    return detail::mha_qkv_layout_supported(p);
}
#endif



#endif  // FLASHATTENTION_MHA_WI

// Runtime dtype dispatch for the BF16 and E4M3 WI implementations.
// Individual strategies keep their semantic/layout/work guards.
// False leaves params untouched and queues no work; true handles the entire
// operation. Runtime errors are propagated.
bool try_run_mha_fast_path(Flash_fwd_params& params, hggcStream_t stream, int max_seqlen_k = -1);
#endif  // FLASHATTENTION_FWD_FASTPATHS

}  // namespace flash
