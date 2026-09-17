/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Independent algebraic mask class for flex flash attention forward kernel.
//
// Methodology reference: MagiAttention csrc/flexible_flash_attention/mask.h
//
// For each (m_block, n_block) tile, the mask is fully described by the
// tuple (attn_type, seqlen_q, seqlen_k). Per-element mask predicate reduces
// to two integer limits per row:
//
//   col < col_limit_left    OR    col >= col_limit_right   →  score = -inf
//
// where:
//   FULL      : col_limit_left = -INF, col_limit_right = +INF (no diagonal constraint)
//   CAUSAL    : col_limit_right = row + (seqlen_k - seqlen_q) + 1
//   INVCAUSAL : col_limit_left  = row + (seqlen_k - seqlen_q)
//   BICAUSAL  : both left and right limits apply
//
// Combined with Seqlenk_mask (for partial tiles at the end of K) and the
// per-slice [k_start, k_end) range from the slice metadata.

#pragma once

#include <cute/tensor.hpp>

#include "../hopper/utils.h"
#include "attn_slice.h"

namespace flash {

using namespace cute;

// MaskFlexFlash<kBlockM, kBlockN, TiledMma, Transposed>:
//   Applies an arbitrary attention mask (one of Full / Causal / InvCausal /
//   BiCausal / Bitmask) to the score tensor tSrS for a single (m_block,
//   n_block) tile.
//
//   attn_type : runtime AttnSliceType (passed as int)
//   k_start/k_end : per-slice K column range (global coords)
//   diagonal_offset : for CAUSAL/INVCAUSAL/BICAUSAL, the offset of the diagonal
//   band_width : for BICAUSAL, distance from the diagonal to the left edge
//   Transposed  : set true for the BWD kernel, where the S accumulator
//                 fragment is partitioned with SdP_swapAB (rows/cols swapped).
template <int kBlockM, int kBlockN, typename TiledMma, bool Transposed = false>
struct MaskFlexFlash {

    // Apply a per-slice algebraic mask to tSrS.
    //
    // Template parameters:
    //   Seqlenk_mask : if true, additionally mask columns >= seqlen_k (partial K tile)
    //
    // Runtime parameters:
    //   tSrS         : score tensor of shape (kBlockM, kBlockN) partitioned by TiledMma
    //   m_block      : Q block index (row block)
    //   n_block      : K block index (col block)
    //   slice_type   : one of SLICE_FULL / SLICE_CAUSAL / SLICE_INVCAUSAL /
    //                  SLICE_BICAUSAL / SLICE_BITMASK
    //   k_start      : slice K range lower bound (global, inclusive)
    //   k_end        : slice K range upper bound (global, exclusive)
    //   diagonal_offset : offset of the diagonal  (k = q + diagonal_offset is the boundary)
    //   band_width   : only used for BICAUSAL, left extent of the band
    //   q_start      : slice Q range lower bound (global, inclusive)
    //   q_end        : slice Q range upper bound (global, exclusive); rows
    //                  outside [q_start, q_end) are fully masked (RangeMerge:
    //                  a slice may only partially cover an m_block)
    //   thread_idx   : current thread id
    //   seqlen_q     : Q sequence length
    //   seqlen_k     : K sequence length
    //   mask_bits    : SLICE_BITMASK only — row-major packed bits, LSB-first
    //                  (ignored by the four algebraic types; they are always
    //                  called with the nullptr default, so their code path is
    //                  untouched apart from one extra slice_type compare).
    //                  Padding bits beyond seqlen_k are zero on the host, so
    //                  the K-tile tail needs no separate seqlenk handling.
    //   mask_row_stride : bytes per Q row in mask_bits
    template <bool Seqlenk_mask, typename Engine, typename Layout>
    CUTLASS_DEVICE void apply(
        Tensor<Engine, Layout>& tSrS,
        int m_block, int n_block,
        int slice_type,
        int k_start, int k_end,
        int diagonal_offset, int band_width,
        int q_start, int q_end,
        int thread_idx,
        int seqlen_q, int seqlen_k,
        unsigned char const* mask_bits = nullptr,
        int mask_row_stride = 0) const
    {
        static_assert(Layout::rank == 3, "Only support 3D tensor");

        // ── Fast path: tile provably UNMASKED → skip all element work ──────
        // The BWD kernel applies the mask on every computed block (no zone
        // splitting), so dense / fully-visible tiles would otherwise burn a
        // full row×col scan of compares.  A tile is untouched when its rows
        // are fully inside [q_start, q_end) ∩ [0, seqlen_q), its K range is
        // inside the slice range [k_start, k_end), no seqlenk tail crosses
        // it, and no diagonal bound cuts it:
        //   FULL      : never cuts
        //   CAUSAL    : right bound row+off+1, tightest at the smallest row
        //   INVCAUSAL : left bound  row+off,   tightest at the largest row
        //   BICAUSAL  : both (left bound shifted by -band_width)
        // BITMASK is excluded: only its per-element bits are authoritative.
        if (slice_type != SLICE_BITMASK) {
            int const tile_row_lo = m_block * kBlockM;
            int const tile_row_hi = tile_row_lo + kBlockM;
            int const tile_col_lo = n_block * kBlockN;
            int const tile_col_hi = tile_col_lo + kBlockN;
            bool const rows_covered = (tile_row_lo >= q_start) && (tile_row_hi <= q_end)
                                   && (tile_row_hi <= seqlen_q);
            bool const cols_covered = (tile_col_lo >= k_start) && (tile_col_hi <= k_end);
            bool const no_seqlenk_tail = !Seqlenk_mask || (tile_col_hi <= seqlen_k);
            if (rows_covered && cols_covered && no_seqlenk_tail) {
                bool no_diag = true;
                if (slice_type == SLICE_CAUSAL || slice_type == SLICE_BICAUSAL) {
                    no_diag = no_diag &&
                        (tile_col_hi <= tile_row_lo + diagonal_offset + 1);
                }
                if (slice_type == SLICE_INVCAUSAL || slice_type == SLICE_BICAUSAL) {
                    int const left_adj = (slice_type == SLICE_BICAUSAL) ? -band_width : 0;
                    no_diag = no_diag &&
                        (tile_col_lo >= (tile_row_hi - 1) + diagonal_offset + left_adj);
                }
                if (no_diag) { return; }
            }
        }

        auto thread_mma  = TiledMma{}.get_slice(thread_idx);

        static constexpr int Row = 0;
        static constexpr int Col = 1;

        // Identity tensor gives us block-local (row, col) coordinates.
        Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS        = thread_mma.partition_C(cS);

        // Reinterpret the 3D score fragment with (row, col, group) layout.
        // Transposed=true (BWD, SdP_swapAB): the fragment's logical rows are
        // K columns and vice versa; convert_layout_acc_rowcol swaps them back.
        Tensor tSrS_rowcol = make_tensor(
            tSrS.data(),
            flash::convert_layout_acc_rowcol<Transposed>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(
            tScS.data(),
            flash::convert_layout_acc_rowcol<Transposed>(tScS.layout()));

        // NOTE: all coordinate lookups use the CURRENT thread's own identity
        // fragment (tScS_rowcol).  Do NOT mix thread-0 coordinates with a
        // per-thread offset: the PPU 16x16 MMA atom's thread->column mapping
        // is not additive across threads for every kBlockN tiling, which
        // would blank the wrong score columns (silently corrupting LSE).

        // ────────────────────────────────────────────────────────────────────────
        // 1) Seqlenk boundary (partial K tile): rightmost columns beyond seqlen_k.
        //    Block-local limit (element coords are block-local).
        // ────────────────────────────────────────────────────────────────────────
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN;

        if constexpr (Seqlenk_mask) {
            if (slice_type == SLICE_FULL || slice_type == SLICE_INVCAUSAL) {
                // Only right boundary from seqlen; no diagonal constraint.
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        if (int(get<Col>(tScS_rowcol(m, n))) >= seqlenk_col_limit) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    }
                }
            }
        }

        // ────────────────────────────────────────────────────────────────────────
        // 2) Per-row algebraic mask: reduce mask type to (col_limit_left, col_limit_right).
        //    Slice-type dispatch is resolved ONCE here (warp-uniform flags) so
        //    the per-row loop carries no if/else-if type chain.
        // ────────────────────────────────────────────────────────────────────────
        bool const is_bitmask = (slice_type == SLICE_BITMASK);
        bool const has_left   = (slice_type == SLICE_INVCAUSAL) || (slice_type == SLICE_BICAUSAL);
        bool const has_right  = (slice_type == SLICE_CAUSAL) || (slice_type == SLICE_BICAUSAL);
        int const left_diag_adj = (slice_type == SLICE_BICAUSAL) ? -band_width : 0;
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const physical_row = int(get<Row>(tScS_rowcol(m, _0{}))) + m_block * kBlockM;

            // Row out of Q bounds (seqlen or the slice's own Q range) → mask
            // everything.  The Q-range check lets RangeMerge assign a slice to
            // an m_block it only partially covers.
            if (physical_row >= seqlen_q || physical_row < q_start || physical_row >= q_end) {
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    tSrS_rowcol(m, n) = -INFINITY;
                }
                continue;
            }

            // SLICE_BITMASK: exact per-element predicate from the packed bit
            // array.  Columns >= seqlen_k fall into the host's zero padding
            // bits, so the Seqlenk_mask tail is covered without extra checks.
            if (is_bitmask) {
                unsigned char const* row_bits =
                    mask_bits + physical_row * mask_row_stride;
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    int const global_col = int(get<Col>(tScS_rowcol(m, n))) + n_block * kBlockN;
                    if (!((row_bits[global_col >> 3] >> (global_col & 7)) & 1)) {
                        tSrS_rowcol(m, n) = -INFINITY;
                    }
                }
                continue;
            }

            // Compute per-row (col_limit_left, col_limit_right) in GLOBAL K coords.
            //
            // col < col_limit_left  → mask  (INVCAUSAL / BICAUSAL lower bound)
            // col >= col_limit_right → mask (CAUSAL / BICAUSAL upper bound)
            // col < k_start  OR  col >= k_end → mask (slice K range)
            //
            // Branchless assembly: the slice K range is the base interval and
            // the diagonal bounds (BICAUSAL: asymmetric band diag-band_width
            // .. diag, right edge pinned at the diagonal) tighten it via
            // selects on the warp-uniform has_left / has_right flags.
            int const diag = physical_row + diagonal_offset;  // "diagonal" column for this row
            int col_limit_left  = cute::max(k_start,
                has_left ? diag + left_diag_adj : -0x7FFFFFFF);
            int col_limit_right = cute::min(k_end,
                has_right ? diag + 1 : 0x7FFFFFFF);

            // If Seqlenk_mask and we have a Causal / BiCausal right boundary, also
            // clamp by seqlen_k (handles the case where both the diagonal and the
            // K-sequence end fall within the tile).
            if constexpr (Seqlenk_mask) {
                col_limit_right = has_right ? cute::min(col_limit_right, seqlen_k)
                                            : col_limit_right;
            }

            // Apply: mask where column (global) is outside [col_limit_left, col_limit_right).
            #pragma unroll
            for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                int const global_col = int(get<Col>(tScS_rowcol(m, n))) + n_block * kBlockN;
                if (global_col < col_limit_left || global_col >= col_limit_right) {
                    tSrS_rowcol(m, n) = -INFINITY;
                }
            }
        }
    }
};

} // namespace flash
