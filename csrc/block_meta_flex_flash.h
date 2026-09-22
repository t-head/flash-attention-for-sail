/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Slice iteration metadata for the flex flash attention forward kernel.
//
// The arbitrary mask is decomposed into slices, each covering a
// [q_start, q_end) Q range and a [k_start, k_end) K range, with one of four
// mask types (FULL / CAUSAL / INVCAUSAL / BICAUSAL).
//
// Slice Q ranges MAY OVERLAP (RangeMerge support, per MagiAttention FFA).
// The merge layout (vbatch_to_slice / row_to_vbatch_start/end) lists, for
// each m_block, all slices covering it.  The kernel processes them with one
// thread block per m_block, carrying the online-softmax state across slices
// so partial results are merged via LSE — exactly Magi's DenseBlockMeta
// batch-loop mechanism, adapted to our SM89 mainloop.
//
// Kernel control flow per m_block:
//   1. RangeMerge: loop over vbatches covering this m_block
//      (slice_idx = vbatch_to_slice[vb]); legacy: single slice from
//      row_to_slice[m_block * kBlockM]
//   2. Check K range is non-empty and within seqlen_k
//   3. Compute zone boundaries for mask dispatch optimization
//   4. Inner K-block loop with zone splitting:
//      - no-mask zone: skip mask.apply() entirely
//      - diagonal zone: apply mask.apply() with algebraic col limits
//      - boundary: apply mask.apply() + seqlenk_mask
//   5. If no covering slice has valid work → kernel writes zeros via epilogue
//
// Constraint: slice Q ranges must be aligned to kBlockM boundaries so that
// slice coverage is constant within each m_block.
//
// Methodology reference: MagiAttention DenseBlockMeta (NeedsBatchLoop /
// RangeMerge) + mask_dispatch zone splitting — adapted to the SM89 kernel
// with configurable iteration direction (MaxToMin / MinToMax).

#pragma once

#include <cute/tensor.hpp>

#include "attn_slice.h"
#include "block_range_flex_flash.h"

namespace flash {

using namespace cute;

// ─── Dispatch direction (matches MagiAttention mask.h) ──────────────────────

enum class DispatchDirection { MinToMax, MaxToMin };

/// Compile-time switch over DispatchDirection, used by the flex flash attention
/// launch templates.  Kept next to the enum (rather than in the shared
/// ../static_switch.h) so the macro can never be expanded in a context where
/// flash::DispatchDirection is undeclared, and so the flex flash attention module
/// stays free of reverse dependencies into the base FA3 headers.
#define DIRECTION_SWITCH(MIN_TO_MAX_COND, DIR_NAME, ...)                                          \
  [&] {                                                                                          \
    if (MIN_TO_MAX_COND) {                                                                       \
      constexpr static flash::DispatchDirection DIR_NAME = flash::DispatchDirection::MinToMax;   \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static flash::DispatchDirection DIR_NAME = flash::DispatchDirection::MaxToMin;   \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()

template <DispatchDirection Dir>
CUTLASS_DEVICE int init_block_cur(int lo, int hi) {
    if constexpr (Dir == DispatchDirection::MaxToMin) {
        return hi - 1;
    } else {
        return lo;
    }
}

template <DispatchDirection Dir>
CUTLASS_DEVICE void advance_block_cur(int& block_cur) {
    if constexpr (Dir == DispatchDirection::MaxToMin) {
        --block_cur;
    } else {
        ++block_cur;
    }
}

template <DispatchDirection Dir>
CUTLASS_DEVICE bool is_block_finish(int block_cur, int lo, int hi) {
    if constexpr (Dir == DispatchDirection::MaxToMin) {
        return block_cur < lo;
    } else {
        return block_cur >= hi;
    }
}

// ─── SliceBlockMetaFlexFlash ────────────────────────────────────────────────
//
// Template parameters:
//   kDir      : iteration direction (MaxToMin = default for FWD online softmax)
//
// One instance describes a single (m_block, slice) work entry.  Under
// RangeMerge the kernel constructs one instance per covering slice and
// merges their results via the persistent online-softmax state.
//
// Zone splitting (per MagiAttention mask_dispatch):
//   For a given (m_block, mask_type, k_start, k_end, diagonal_offset, band_width):
//     causal_no_mask_start  : first n_block where the diagonal crosses the
//                             tile (all blocks below it are fully visible)
//     causal_no_mask_end    : last n_block (exclusive) in the no-mask zone
//     invcausal_no_mask_start : first n_block where ALL cols are above the diagonal
//     bicausal_no_mask_start/end : band around diagonal where both constraints hold
//
//   The kernel uses these to skip mask.apply() in no-mask zones.

template <DispatchDirection kDir = DispatchDirection::MaxToMin>
struct SliceBlockMetaFlexFlash {
    // Slice metadata (device pointers, borrowed from params).
    AttnSliceParams const& slices;
    int const seqlen_q;
    int const seqlen_k;
    int const kBlockM;
    int const kBlockN;

    // Current iteration state.
    int slice_idx;          // resolved slice index for this m_block
    int inner_block_min;    // first valid n_block in current slice's K range
    int inner_block_cnt;    // one past last valid n_block (ceil_div)
    int inner_block_idx;    // current n_block cursor

    // Per-slice cached fields (loaded once to avoid repeated global memory reads).
    int cur_k_start;
    int cur_k_end;
    int cur_mask_type;
    int cur_diagonal_offset;
    int cur_band_width;
    int cur_q_start;        // slice Q range (effective, intersected with m_block)
    int cur_q_end;

    // True when the slice only partially covers this m_block's Q rows;
    // mask.apply() is then required for every K-block (rows outside the
    // slice's Q range must be masked), so no-mask zones are disabled.
    bool q_partial;

    // ── Zone boundaries for mask dispatch optimization ──
    // These partition the K-block iteration range [inner_block_min, inner_block_cnt)
    // into zones with different masking requirements.
    //
    // For CAUSAL (k <= q + offset):
    //   n_block <  causal_no_mask_start → all cols below the diagonal → no mask
    //   n_block >= causal_no_mask_start → diagonal crosses tile (or above) → mask
    //
    // For INVCAUSAL (k >= q + offset):
    //   n_block >= invcausal_no_mask_end → all cols above the diagonal → no mask
    //   n_block <  invcausal_no_mask_end → diagonal crosses tile (or below) → mask
    //
    // For BICAUSAL (|k - q - offset| <= band_width):
    //   bicausal_no_mask_start <= n_block < bicausal_no_mask_end → fully in band
    //
    // For FULL: all blocks are no-mask (except last block for seqlenk).
    int causal_no_mask_start;      // CAUSAL: first crossing (masked) n_block
    int invcausal_no_mask_end;     // INVCAUSAL: last fully-visible n_block (exclusive)
    int bicausal_no_mask_start;    // BICAUSAL: first fully-in-band n_block
    int bicausal_no_mask_end;      // BICAUSAL: last fully-in-band n_block (exclusive)

    // Flag: true if this m_block has no valid slice (dummy or out-of-range).
    bool no_valid_slice;

    // True when the K-range alignment guard disabled ALL no-mask zones for
    // this work entry: mask.apply() is then required on every block to clip
    // columns to [cur_k_start, cur_k_end).  Checked before any zone query —
    // the per-type zone sentinels have OPPOSITE polarities (CAUSAL uses a
    // `n < start` predicate, FULL/INVCAUSAL use `n >= x`), so no single
    // sentinel value can encode "empty zone" for all of them.
    bool zones_disabled;

    // This m_block's Q row range [block_q_start, block_q_end).
    int block_q_start;
    int block_q_end;

    // Initialize state for a given (m_block, slice) work entry.
    // slice_idx is resolved by the caller (vbatch_to_slice[vb] under
    // RangeMerge, or row_to_slice[m_block * kBlockM] in the legacy path).
    CUTLASS_DEVICE
    SliceBlockMetaFlexFlash(
        AttnSliceParams const& slices_,
        int seqlen_q_, int seqlen_k_,
        int kBlockM_, int kBlockN_,
        int slice_idx_, int m_block_)
        : slices(slices_)
        , seqlen_q(seqlen_q_)
        , seqlen_k(seqlen_k_)
        , kBlockM(kBlockM_)
        , kBlockN(kBlockN_)
        , slice_idx(slice_idx_)
        , inner_block_min(0)
        , inner_block_cnt(0)
        , inner_block_idx(0)
        , cur_k_start(0)
        , cur_k_end(0)
        , cur_mask_type(SLICE_FULL)
        , cur_diagonal_offset(0)
        , cur_band_width(0)
        , cur_q_start(0)
        , cur_q_end(0)
        , q_partial(false)
        , causal_no_mask_start(0)
        , invcausal_no_mask_end(0)
        , bicausal_no_mask_start(0x7FFFFFFF)
        , bicausal_no_mask_end(0)
        , no_valid_slice(true)
        , zones_disabled(false)
    {
        block_q_start = m_block_ * kBlockM;
        block_q_end   = min(block_q_start + kBlockM, seqlen_q);
    }

    // Resolve the slice, initialize cursor, and compute zone boundaries.
    //
    // Returns true if this work entry has no valid work (invalid slice index,
    // empty K range, or slice Q range disjoint from this m_block); the caller
    // should then skip to the next vbatch.
    CUTLASS_DEVICE bool skip_to_first_valid() {
        if (slice_idx < 0 || slice_idx >= slices.num_slices) {
            no_valid_slice = true;
            return true;
        }

        // Intersect the slice Q range with this m_block's rows.
        int const qs = max(block_q_start, slices.q_starts[slice_idx]);
        int const qe = min(block_q_end,   slices.q_ends[slice_idx]);
        if (qe <= qs) {
            no_valid_slice = true;
            return true;
        }

        // Read slice K range and intersect with [0, seqlen_k).
        int const ks = slices.k_starts[slice_idx];
        int const ke = slices.k_ends[slice_idx];
        int const eff_start = max(0, ks);
        int const eff_end   = min(seqlen_k, ke);

        if (eff_end <= eff_start) {
            no_valid_slice = true;
            return true;
        }

        // Valid slice found — cache per-slice fields.
        no_valid_slice      = false;
        cur_k_start         = eff_start;
        cur_k_end           = eff_end;
        cur_mask_type       = slices.mask_types[slice_idx];
        cur_diagonal_offset = slices.diagonal_offsets[slice_idx];
        cur_band_width      = slices.band_widths[slice_idx];
        cur_q_start         = qs;
        cur_q_end           = qe;
        // Partial coverage of this m_block → mask.apply() needed everywhere.
        q_partial           = (qs > block_q_start) || (qe < block_q_end);

        // K-block iteration range for this slice.
        inner_block_min = cur_k_start / kBlockN;
        inner_block_cnt = (cur_k_end + kBlockN - 1) / kBlockN;  // ceil_div
        // Pre-pruning block count: fwd_valid_n_range below tightens the range
        // to the mask's diagonal constraint, but the K-range ALIGNMENT (needed
        // by the no-mask-zone check below) is a property of the slice's own
        // [cur_k_start, cur_k_end), so capture it before the pruning.
        int const slice_block_min = inner_block_min;
        int const slice_block_cnt = inner_block_cnt;

        // Tile-level diagonal pruning: tighten the n_block range with the
        // mask's diagonal constraint so fully-masked tiles are never visited
        // (see block_range_flex_flash.h).  Runs BEFORE compute_zone_boundaries()
        // since zones use inner_block_cnt as the empty-zone sentinel.
        // Pass the CLAMPED row range (block_q_end already min'd with
        // seqlen_q) so the last partial m_block prunes identically to
        // fwd_n_cover_count's counting predicate.
        fwd_valid_n_range(cur_mask_type, cur_diagonal_offset, cur_band_width,
                          block_q_start, block_q_end,
                          kBlockN, inner_block_min, inner_block_cnt);
        if (inner_block_cnt <= inner_block_min) {
            no_valid_slice = true;
            return true;
        }

        // Initialize cursor based on direction.
        inner_block_idx = init_block_cur<kDir>(inner_block_min, inner_block_cnt);

        // Compute zone boundaries.
        compute_zone_boundaries();

        // SLICE_BITMASK has no algebraic structure: no K block is provably
        // fully visible, so mask.apply() must run on EVERY block to test the
        // element-wise bits.  (compute_zone_boundaries() falls through to the
        // FULL branch for it, which would declare the whole range no-mask.)
        if (cur_mask_type == SLICE_BITMASK) {
            zones_disabled = true;
        }

        // If the slice K range does not cover its iteration blocks completely
        // (unaligned k_start/k_end), mask.apply() is required on EVERY block
        // to clip columns to [cur_k_start, cur_k_end) — this applies to
        // FULL slices as well.  Disable all no-mask zones in that case.
        // NOTE: compare against the PRE-PRUNING block range; the diagonal
        // pruning (fwd_valid_n_range) only removes fully-masked tiles and
        // says nothing about the slice K-range alignment.
        if (cur_k_start != slice_block_min * kBlockN || cur_k_end != slice_block_cnt * kBlockN) {
            zones_disabled = true;
            causal_no_mask_start   = inner_block_min;  // empty zone (pred is n < start)
            invcausal_no_mask_end  = inner_block_cnt;  // empty zone
            bicausal_no_mask_start = inner_block_cnt;  // empty zone
            bicausal_no_mask_end   = inner_block_cnt;
        }

        return false;
    }

    // Compute zone boundaries for mask dispatch optimization.
    // Reference: MagiAttention mask_dispatch zone computation (mask.h L370-378).
    CUTLASS_DEVICE void compute_zone_boundaries() {
        // m_block's Q row range in GLOBAL coordinates.  diagonal_offset is a
        // global quantity (k = q + diagonal_offset, same convention as
        // MaskFlexFlash::apply), so the zone math must use global rows too —
        // block-local rows would misplace the diagonal for every m_block > 0.
        int const q_lo = block_q_start;                 // first row (global)
        int const q_hi = block_q_end - 1;               // last row (global)

        int const m_block_start = inner_block_min * kBlockN;  // approximate
        (void)m_block_start;

        if (cur_mask_type == SLICE_CAUSAL) {
            // CAUSAL: k <= q + diagonal_offset
            // A K-block at n_block is fully visible (no mask) when:
            //   ALL columns in [n_block*kBlockN, (n_block+1)*kBlockN) satisfy
            //   col <= q_max + diagonal_offset for ALL rows q in the m_block.
            // The most restrictive row is q_lo (smallest q), so:
            //   n_block * kBlockN + kBlockN - 1 <= q_lo + diagonal_offset
            //   → n_block <= (q_lo + diag_offset - kBlockN + 1) / kBlockN
            //
            // A K-block needs mask when diagonal crosses through it:
            //   some cols visible, some masked.
            //
            // A K-block is fully masked when ALL cols > q_hi + diag_offset.
            int const diag_lo = q_lo + cur_diagonal_offset;  // diagonal for first row
            int const diag_hi = q_hi + cur_diagonal_offset;  // diagonal for last row

            // First n_block where the diagonal crosses the tile (some cols
            // masked); all blocks BELOW it are fully visible.
            // Need: (n_block+1)*kBlockN - 1 <= diag_lo → n_block <= (diag_lo - kBlockN + 1)/kBlockN
            // → crossing block = ceil((diag_lo + 1) / kBlockN).
            causal_no_mask_start = (diag_lo >= kBlockN - 1)
                ? (diag_lo - kBlockN + 1) / kBlockN + 1
                : 0;  // diagonal past the whole tile range → every block visible

            // Clamp to iteration range.
            if (causal_no_mask_start < inner_block_min) causal_no_mask_start = inner_block_min;
            if (causal_no_mask_start > inner_block_cnt) causal_no_mask_start = inner_block_cnt;

        } else if (cur_mask_type == SLICE_INVCAUSAL) {
            // INVCAUSAL: k >= q + diagonal_offset
            // A K-block is fully visible when ALL cols >= q_hi + diag_offset:
            //   n_block * kBlockN >= q_hi + diag_offset
            //   → n_block >= (q_hi + diag_offset) / kBlockN
            int const diag_hi = q_hi + cur_diagonal_offset;

            invcausal_no_mask_end = (diag_hi + kBlockN - 1) / kBlockN;  // ceil_div
            // Blocks with n_block >= invcausal_no_mask_end are fully visible.
            // So no-mask zone is [invcausal_no_mask_end, inner_block_cnt).

            if (invcausal_no_mask_end < inner_block_min) invcausal_no_mask_end = inner_block_min;
            if (invcausal_no_mask_end > inner_block_cnt) invcausal_no_mask_end = inner_block_cnt;

        } else if (cur_mask_type == SLICE_BICAUSAL) {
            // BICAUSAL: q + offset - band_width <= k <= q + offset
            // A K-block is fully visible when:
            //   ALL cols >= q_hi + offset - bw  AND  ALL cols <= q_lo + offset
            int const diag_lo = q_lo + cur_diagonal_offset;
            int const diag_hi = q_hi + cur_diagonal_offset;
            int const lower_bound = diag_hi - cur_band_width;  // k >= lower
            int const upper_bound = diag_lo;                   // k <= upper

            bicausal_no_mask_start = (lower_bound >= 0)
                ? (lower_bound + kBlockN - 1) / kBlockN  // ceil_div
                : inner_block_min;   // band extends left of column 0 → every
                                     // block satisfies the left constraint
            // Number of blocks lying entirely at or left of the diagonal.  The
            // fallback must EMPTY the zone (the predicate is
            // start <= n < end): upper_bound < kBlockN-1 means the diagonal of
            // the block's first row already cuts through the very first K
            // tile, so NO block is fully visible.  Setting inner_block_cnt
            // here instead declared the whole range no-mask and skipped
            // mask.apply() on it — for a hand-built BICAUSAL slice whose band
            // reaches left of column 0 that silently exposed every column.
            bicausal_no_mask_end = (upper_bound >= kBlockN - 1)
                ? (upper_bound - kBlockN + 1) / kBlockN + 1
                : inner_block_min;

            if (bicausal_no_mask_start < inner_block_min) bicausal_no_mask_start = inner_block_min;
            if (bicausal_no_mask_start > inner_block_cnt) bicausal_no_mask_start = inner_block_cnt;
            if (bicausal_no_mask_end < inner_block_min) bicausal_no_mask_end = inner_block_min;
            if (bicausal_no_mask_end > inner_block_cnt) bicausal_no_mask_end = inner_block_cnt;

        } else {
            // FULL: all blocks are no-mask (zone spans entire range).
            causal_no_mask_start = inner_block_min;
        }
    }

    // ── Zone query: is n_block in a no-mask zone? ──
    // Returns true if the K-block at n_block is fully visible for ALL rows
    // in this m_block, meaning mask.apply() can be skipped entirely.
    CUTLASS_DEVICE bool is_no_mask_block(int n_block) const {
        // Partial Q coverage: rows outside the slice's Q range must be masked.
        if (q_partial) return false;
        // Unaligned slice K range: every block needs mask.apply() clipping.
        if (zones_disabled) return false;
        if (cur_mask_type == SLICE_FULL) {
            // FULL has no diagonal constraint, but mask.apply() is still needed
            // to clip columns to [cur_k_start, cur_k_end) when the K range is
            // not block-aligned (skip_to_first_valid() empties the zone then).
            return n_block >= causal_no_mask_start;
        } else if (cur_mask_type == SLICE_CAUSAL) {
            return n_block < causal_no_mask_start;
        } else if (cur_mask_type == SLICE_INVCAUSAL) {
            return n_block >= invcausal_no_mask_end;
        } else if (cur_mask_type == SLICE_BICAUSAL) {
            return n_block >= bicausal_no_mask_start && n_block < bicausal_no_mask_end;
        }
        return false;
    }

    // RangeMerge: vbatch iteration is driven by the kernel's outer loop;
    // nothing to prefetch within a single (m_block, slice) entry.
    CUTLASS_DEVICE bool prefetch() {
        return true;
    }

    // Check whether the K-block cursor is past the iteration range.
    CUTLASS_DEVICE bool is_inner_finish() const {
        return is_block_finish<kDir>(inner_block_idx, inner_block_min, inner_block_cnt);
    }

    // Advance the K-block cursor in the configured direction.
    CUTLASS_DEVICE void advance_inner() {
        advance_block_cur<kDir>(inner_block_idx);
    }
};

// ─── TrivialSliceMeta ───────────────────────────────────────────────────────
//
// Drop-in replacement of SliceBlockMetaFlexFlash for the trivial
// single-slice specialization (AttnSliceParams::trivial_mask): the host
// PROVED the mask is exactly one FULL or CAUSAL slice covering the whole
// (seqlen_q, seqlen_k) rectangle, so every field the mainloop consumes is
// derivable from host constants — zero descriptor global loads per work
// tile (vs ~9 for the generic path: row_to_vbatch_start/end + the seven
// per-slice fields of skip_to_first_valid) and no vbatch bookkeeping.
//
// The zone model mirrors SliceBlockMetaFlexFlash with one improvement:
// the generic K-range ALIGNMENT guard must disable ALL no-mask zones
// whenever seqlen_k % kBlockN != 0 (cur_k_end unaligned), forcing
// mask.apply() on every block; here only the single tail block crossing
// seqlen_k actually needs clipping (no_mask_end folds in the floor
// seqlen_k / kBlockN), because triviality guarantees k_start == 0 and
// k_end == seqlen_k.
//
// Bitwise equivalence with the generic path: identical block set (the
// per-m_block count mirrors fwd_valid_n_range's CAUSAL pruning), identical
// iteration direction and identical -inf masked inputs, so softmax/O/LSE
// come out bitwise equal.
template <DispatchDirection kDir = DispatchDirection::MaxToMin>
struct TrivialSliceMeta {
    // Cursor state (same names/roles as SliceBlockMetaFlexFlash).
    int inner_block_min;
    int inner_block_cnt;
    int inner_block_idx;

    // Per-slice cached fields consumed by step() / mask.apply().
    int cur_k_start;
    int cur_k_end;
    int cur_mask_type;
    int cur_diagonal_offset;
    int cur_band_width;
    int cur_q_start;
    int cur_q_end;

    // First n_block OUTSIDE the no-mask zone (polarity: n < no_mask_end is
    // fully visible).  Folds in BOTH the diagonal crossing (CAUSAL) and the
    // seqlen_k tail (blocks [floor(sk/kBlockN), cnt) cross the K end).
    int no_mask_end;

    CUTLASS_DEVICE
    TrivialSliceMeta(int trivial_mask, int diagonal_offset,
                     int seqlen_q, int seqlen_k,
                     int kBlockM_, int kBlockN_, int m_block)
        : inner_block_min(0)
        , inner_block_cnt(0)
        , inner_block_idx(0)
        , cur_k_start(0)
        , cur_k_end(seqlen_k)
        , cur_mask_type(trivial_mask == 2 ? SLICE_CAUSAL : SLICE_FULL)
        , cur_diagonal_offset(diagonal_offset)
        , cur_band_width(0)
        , cur_q_start(m_block * kBlockM_)
        , cur_q_end(min(m_block * kBlockM_ + kBlockM_, seqlen_q))
        , no_mask_end(0)
    {
        // K-block iteration range [0, cnt).  CAUSAL: mirror the generic
        // fwd_valid_n_range tile pruning — rows of this m_block see at most
        // column (cur_q_end - 1) + diagonal_offset, so blocks past that are
        // fully masked and never visited.
        int const k_visible_end = (cur_mask_type == SLICE_CAUSAL)
            ? min(seqlen_k, cur_q_end + cur_diagonal_offset)
            : seqlen_k;
        inner_block_cnt = (k_visible_end > 0)
            ? cute::ceil_div(k_visible_end, kBlockN_) : 0;
        inner_block_idx = init_block_cur<kDir>(inner_block_min, inner_block_cnt);

        // No-mask zone: every block fully inside [0, seqlen_k) AND (CAUSAL)
        // fully below the tightest diagonal of this m_block (row cur_q_start).
        // Same formula as compute_zone_boundaries()'s CAUSAL branch, with
        // diag_lo = cur_q_start + diagonal_offset.
        if (cur_mask_type == SLICE_FULL) {
            no_mask_end = seqlen_k / kBlockN_;       // tail block stays masked
        } else {
            int const diag_lo = cur_q_start + cur_diagonal_offset;
            int const cross = (diag_lo >= kBlockN_ - 1)
                ? (diag_lo - kBlockN_ + 1) / kBlockN_ + 1
                : 0;
            no_mask_end = min(min(cross, inner_block_cnt),
                              seqlen_k / kBlockN_);
        }
    }

    // False when this m_block has no work (CAUSAL whose diagonal lies left
    // of every column of the block); the caller then stores zeros, exactly
    // like the generic no_valid_slice outcome.
    CUTLASS_DEVICE bool has_work() const { return inner_block_cnt > 0; }

    CUTLASS_DEVICE bool is_no_mask_block(int n_block) const {
        return n_block < no_mask_end;
    }

    CUTLASS_DEVICE bool is_inner_finish() const {
        return is_block_finish<kDir>(inner_block_idx, inner_block_min, inner_block_cnt);
    }

    CUTLASS_DEVICE void advance_inner() {
        advance_block_cur<kDir>(inner_block_idx);
    }
};

} // namespace flash
