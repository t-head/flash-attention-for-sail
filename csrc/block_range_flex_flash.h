/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Tile-level valid-range pruning for arbitrary mask slices.
//
// Methodology reference: MagiAttention DenseBlockMeta first/last valid block.
// Given a slice (mask type + diagonal offset + band width) and one tile on
// the "outer" axis, these helpers tighten the iteration range on the "inner"
// axis so that tiles which are COMPLETELY masked are never computed.
//
// IMPORTANT: this only tightens loop BOUNDS.  MaskFlexFlash::apply() still
// runs on every computed tile and keeps all of its responsibilities
// (diagonal constraint, per-slice K/Q range clipping, seqlen tail) — see the
// no-mask-zone pitfall where skipping mask.apply() also skipped K clipping.
//
// Mask semantics (matching attn_slice.h):
//   SLICE_FULL:      no diagonal constraint
//   SLICE_CAUSAL:    k <= q + o
//   SLICE_INVCAUSAL: k >= q + o
//   SLICE_BICAUSAL:  q + o - w <= k <= q + o

#pragma once

#include <cutlass/cutlass.h>

#include "attn_slice.h"

namespace flash {

// ─── BWD: valid m_block range for a given (slice, n_block) ─────────────────
//
// The K tile covers columns [n_lo, n_hi).  Returns in [m_lo, m_hi) the
// m_block range that may contain at least one unmasked element, intersected
// with the slice Q range [q_start, q_end), [0, seqlen_q) and the caller's
// coarse bounds.  Empty result (m_lo >= m_hi) means the (slice, n_block)
// pair has no work.
//
// Derivation (tile intersects the mask region):
//   CAUSAL:    exists k <= q + o with k >= n_lo        → q >= n_lo - o
//   INVCAUSAL: exists k >= q + o with k <= n_hi - 1    → q <= n_hi - 1 - o
//   BICAUSAL:  exists q+o-w <= k <= q+o              → q in [n_lo-o, n_hi-1-o+w]
CUTLASS_DEVICE
void bwd_valid_m_range(int mask_type, int diagonal_offset, int band_width,
                       int q_start, int q_end,
                       int n_lo, int n_hi,
                       int kBlockM, int seqlen_q,
                       int &m_lo, int &m_hi) {
    int const q_max = min(q_end, seqlen_q);
    int q_first = q_start;
    int q_last_excl = q_max;  // exclusive

    if (mask_type == SLICE_CAUSAL) {
        q_first = max(q_first, n_lo - diagonal_offset);
    } else if (mask_type == SLICE_INVCAUSAL) {
        q_last_excl = min(q_last_excl, n_hi - diagonal_offset);  // q <= n_hi-1-o
    } else if (mask_type == SLICE_BICAUSAL) {
        q_first = max(q_first, n_lo - diagonal_offset);          // k <= q+o, k >= n_lo
        q_last_excl = min(q_last_excl, n_hi - diagonal_offset + band_width);  // q <= n_hi-1-o+w
    }
    // FULL: no diagonal constraint → keep the caller's Q-range bounds.

    // m_lo from an INCLUSIVE first row: the first m_block containing row
    // q_first is floor(q_first / kBlockM) (NOT ceil_div — e.g. row 64 with
    // kBlockM=128 lives in m_block 0).  m_hi from the EXCLUSIVE last row:
    // one past the m_block containing row q_last_excl-1 = ceil_div.
    m_lo = max(m_lo, max(0, q_first) / kBlockM);
    m_hi = min(m_hi, cute::ceil_div(q_last_excl, kBlockM));
}

// ─── FWD: valid n_block range for a given (slice, m_block) ─────────────────
//
// Dual of bwd_valid_m_range: the Q tile covers rows [m_lo_row, m_hi_row).
// Tightens the n_block iteration range [n_lo_blk, n_hi_blk) (initialized by
// the caller from the slice K range).
//
// Derivation:
//   CAUSAL:    exists k <= q + o with k in tile → k <= m_hi_row-1+o
//   INVCAUSAL: exists k >= q + o with k in tile → k >= m_lo_row+o
//   BICAUSAL:  k in [m_lo_row+o-w, m_hi_row-1+o]
CUTLASS_DEVICE
void fwd_valid_n_range(int mask_type, int diagonal_offset, int band_width,
                       int m_lo_row, int m_hi_row,
                       int kBlockN,
                       int &n_lo_blk, int &n_hi_blk) {
    if (mask_type == SLICE_CAUSAL) {
        // Fully-masked when the tile's smallest k > m_hi_row-1+o.
        int const k_max_valid = m_hi_row - 1 + diagonal_offset;   // inclusive
        n_hi_blk = min(n_hi_blk, cute::ceil_div(k_max_valid + 1, kBlockN));
    } else if (mask_type == SLICE_INVCAUSAL) {
        // Fully-masked when the tile's largest k < m_lo_row+o.
        int const k_min_valid = m_lo_row + diagonal_offset;       // inclusive
        n_lo_blk = max(n_lo_blk, k_min_valid / kBlockN);
    } else if (mask_type == SLICE_BICAUSAL) {
        int const k_min_valid = m_lo_row + diagonal_offset - band_width;
        int const k_max_valid = m_hi_row - 1 + diagonal_offset;
        n_lo_blk = max(n_lo_blk, max(0, k_min_valid) / kBlockN);
        n_hi_blk = min(n_hi_blk, cute::ceil_div(k_max_valid + 1, kBlockN));
    }
    // FULL: no diagonal constraint → unchanged.
}

// ─── BWD determinism helper: contiguous m_block cover of one n_block ───────
//
// For a fixed (slice, n_block), the set of m_blocks whose tile intersects
// the mask region is a CONTIGUOUS interval (FULL/CAUSAL/INVCAUSAL produce
// half-bounded intervals, BICAUSAL a band).  Returns that interval
// [m_lo, m_hi), intersected with the slice Q range, the slice K range
// (clamped to [0, seqlen_k)) and [0, seqlen_q).
// Empty interval (m_lo >= m_hi) ⇒ no (slice, n_block) contributor.
//
// Used by the deterministic backward paths to assign each contributor
// (slice, n_block/m_block, head) a gap-free "turn" in a fixed total order.
// The K-range clamp is REQUIRED for that invariant: the actual participation
// gates in both bwd mainloops skip slices whose K range does not reach the
// n_block tile, so counting them here would inflate the turn past the real
// arrival count (semaphore deadlock under LOOPK × DETERMINISTIC).
CUTLASS_DEVICE
void bwd_m_cover_interval(int mask_type, int diagonal_offset, int band_width,
                          int q_start, int q_end,
                          int k_start, int k_end,
                          int n_lo, int n_hi,
                          int kBlockM, int seqlen_q, int seqlen_k,
                          int &m_lo, int &m_hi) {
    int const q_max = min(q_end, seqlen_q);
    if (q_max <= q_start) { m_lo = 0; m_hi = 0; return; }
    // Clamp the K tile to the slice's own K range: contributors exist only
    // for columns the slice actually covers (dual of the mainloop gates,
    // which intersect [ks_eff, ke_eff) with the tile before pruning).
    n_lo = max(n_lo, max(k_start, 0));
    n_hi = min(n_hi, min(k_end, seqlen_k));
    if (n_hi <= n_lo) { m_lo = 0; m_hi = 0; return; }
    m_lo = q_start / kBlockM;
    m_hi = cute::ceil_div(q_max, kBlockM);
    bwd_valid_m_range(mask_type, diagonal_offset, band_width,
                      q_start, q_end, n_lo, n_hi, kBlockM, seqlen_q,
                      m_lo, m_hi);
    if (m_hi < m_lo) { m_hi = m_lo; }
}

// ─── FWD determinism helper: how many n_blocks of a slice cover an m_block ─
//
// Dual of bwd_m_cover_interval: for a fixed (slice, m_block), the covering
// n_blocks also form a contiguous interval; returns its length.  Rows are
// clamped to [0, seqlen_q) so the count matches the bwd-side contributor
// set exactly (duality of the tile ∩ mask-region predicate).
CUTLASS_DEVICE
int fwd_n_cover_count(int mask_type, int diagonal_offset, int band_width,
                      int q_start, int q_end,
                      int k_start, int k_end,
                      int m_block, int kBlockM, int kBlockN,
                      int seqlen_q, int seqlen_k) {
    int const m_lo_row = m_block * kBlockM;
    int const m_hi_row = min(m_lo_row + kBlockM, seqlen_q);
    int const qs_eff = max(q_start, m_lo_row);
    int const qe_eff = min(min(q_end, seqlen_q), m_hi_row);
    if (qe_eff <= qs_eff) { return 0; }
    int const ks_eff = max(k_start, 0);
    int const ke_eff = min(k_end, seqlen_k);
    if (ke_eff <= ks_eff) { return 0; }
    int n_lo_blk = ks_eff / kBlockN;
    int n_hi_blk = cute::ceil_div(ke_eff, kBlockN);
    fwd_valid_n_range(mask_type, diagonal_offset, band_width,
                      m_lo_row, m_hi_row, kBlockN, n_lo_blk, n_hi_blk);
    return max(0, n_hi_blk - n_lo_blk);
}

// ─── BWD determinism helper (LoopK): m_block cover count of one (slice, n_block)
//
// Number of m_blocks whose tile intersects the mask region for a fixed
// (slice, n_block) — simply the length of bwd_m_cover_interval (including
// its K-range clamp).  Used by the deterministic LoopK mainloop to build
// gap-free dK/dV turns; the dual fwd_valid_n_range used for LoopK's
// inner-range pruning encodes the SAME tile ∩ mask-region predicate (with
// identical seqlen and K-range clamping), so every contributing
// (slice, m_block) CTA is counted exactly once.
CUTLASS_DEVICE
int bwd_m_cover_count(int mask_type, int diagonal_offset, int band_width,
                      int q_start, int q_end,
                      int k_start, int k_end,
                      int n_block, int kBlockM, int kBlockN,
                      int seqlen_q, int seqlen_k) {
    int const n_lo = n_block * kBlockN;
    int const n_hi = n_lo + kBlockN;
    int m_lo, m_hi;
    bwd_m_cover_interval(mask_type, diagonal_offset, band_width,
                         q_start, q_end, k_start, k_end,
                         n_lo, n_hi, kBlockM, seqlen_q, seqlen_k,
                         m_lo, m_hi);
    return max(0, m_hi - m_lo);
}

} // namespace flash
