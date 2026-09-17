/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Flex flash attention support: attention slice data structures.
//
// An arbitrary mask is decomposed into "slices", each covering a contiguous
// [q_start, q_end) range of Q rows and a contiguous [k_start, k_end) range of
// K columns, with one of four mask types
// (FULL / CAUSAL / INVCAUSAL / BICAUSAL).
//
// Slice Q ranges MAY overlap (RangeMerge support, per MagiAttention FFA):
// a single Q row can be covered by multiple slices.  The merge layout
// (vbatch_to_slice / row_to_vbatch_start / row_to_vbatch_end) lists, for each
// m_block, all slices covering it; the kernel iterates over them with a
// persistent online-softmax state so partial results are merged via LSE.
//
// Dispatch: with merge layout present, the kernel loops
//   for vb in [row_to_vbatch_start[row], row_to_vbatch_end[row])
//       process slice vbatch_to_slice[vb]
// Legacy path (no merge layout): row_to_slice[m_block * kBlockM] gives the
// single owning slice; slice Q ranges must then be non-overlapping and
// aligned to kBlockM boundaries.

#pragma once

#include "../hopper/flash.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// Mask type for a single attention slice.  Values match MagiAttention's
// AttnType enum for conceptual correspondence.
enum AttnSliceType : int {
    SLICE_FULL      = 0,  // Dense rectangle: no mask within [q_start,q_end) x [k_start,k_end)
    SLICE_CAUSAL    = 1,  // Lower triangle: k <= q + offset
    SLICE_INVCAUSAL = 2,  // Upper triangle: k >= q + offset
    SLICE_BICAUSAL  = 3,  // Band left of the diagonal: q + offset - band_width <= k <= q + offset
    SLICE_BITMASK   = 4,  // Element-wise bitmask fallback: per-(q,k) bit from
                          // AttnSliceParams::mask_bits.  Exact for ANY boolean
                          // mask; used when the layered interval decomposition
                          // exceeds its cap (unstructured sparsity).  No
                          // no-mask zones, no diagonal tile pruning.
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GPU-side parameters describing the slice decomposition.
// All pointer arrays have length num_slices (except row_to_slice which has
// length seqlen_q).  Memory is owned by the caller; pointers are borrowed.
struct AttnSliceParams {
    // Per-slice range descriptors [num_slices].
    int *__restrict__ q_starts;       // Q row start (inclusive)
    int *__restrict__ q_ends;         // Q row end   (exclusive)
    int *__restrict__ k_starts;       // K col start (inclusive)
    int *__restrict__ k_ends;         // K col end   (exclusive)

    // Per-slice mask type [num_slices], values from AttnSliceType.
    int *__restrict__ mask_types;

    // Q-row → slice index mapping [seqlen_q] (legacy non-overlapping path).
    // row_to_slice[q_row] = index into the per-slice arrays above.
    // Rows not covered by any slice hold -1; the kernel treats negative (or
    // out-of-range) entries as "no work" and skips them (see
    // SliceBlockMetaFlexFlash::skip_to_first_valid).
    // When a merge layout is provided this field is still required but only
    // used as a fallback reference (first covering slice per row).
    int *__restrict__ row_to_slice;

    // Per-slice diagonal offset [num_slices].
    // For CAUSAL:   mask condition is  k <= q + diagonal_offsets[slice]
    // For INVCAUSAL: mask condition is k >= q + diagonal_offsets[slice]
    // For BICAUSAL:  q + diagonal_offsets[slice] - band_widths[slice] <= k <= q + diagonal_offsets[slice]
    // For FULL:      ignored.
    int *__restrict__ diagonal_offsets;

    // Per-slice band width [num_slices], only used for BICAUSAL.
    int *__restrict__ band_widths;

    int num_slices;

    // ── RangeMerge layout (Q ranges may overlap) ──────────────────────────
    // A "vbatch" (virtual batch, per MagiAttention terminology) is one
    // (m_block, covering slice) work entry.  Coverage is constant within an
    // m_block (slice Q boundaries are kBlockM-aligned), so the arrays below
    // are indexed by Q row but hold identical values for all rows of the
    // same m_block.
    //
    // When vbatch_to_slice == nullptr the kernel uses the legacy single-slice
    // dispatch via row_to_slice[].
    int *__restrict__ vbatch_to_slice;      // [num_vbatches] vbatch → slice idx
    int *__restrict__ row_to_vbatch_start;  // [seqlen_q] first covering vbatch
    int *__restrict__ row_to_vbatch_end;    // [seqlen_q] one past last covering
    int num_vbatches;

    // Optional perf-debug counters (nullptr = disabled).  Layout:
    //   [0] total inner-loop steps, [1] masked steps, [2] slice iterations,
    //   [3] skipped (empty) slice iterations, [4] accumulated clock64 cycles
    // The kernel atomicAdds into these; the host zeroes them before launch.
    unsigned long long *__restrict__ dbg_counters;

    // ── Bitmask fallback (SLICE_BITMASK slices only) ──────────────────────
    // Row-major packed bits, LSB-first within a byte:
    //   visible(q, k) == (mask_bits[q * mask_row_stride + (k >> 3)]
    //                       >> (k & 7)) & 1
    // The row stride pads seqlen_k up to a multiple of 128 bits so the
    // kernel's last K tile (kBlockN <= 128) never reads past the buffer;
    // padding bits are zero, which doubles as the seqlen_k tail mask.
    // nullptr when no SLICE_BITMASK slice is present — the existing four
    // mask types never touch these fields.
    unsigned char const *__restrict__ mask_bits;
    int mask_row_stride;   // bytes per Q row (>= ceil(seqlen_k / 8))

    // ── Per-(batch, head) heterogeneous masks (P3) ────────────────────────
    // bh_to_group == nullptr → one shared layout for all (batch, head)
    // pairs: every field above is used as-is (zero-overhead legacy path).
    //
    // Otherwise the per-slice and RangeMerge arrays above are the FLAT
    // concatenation of `num_groups` distinct layouts, and each (batch,
    // head) pair picks its layout via
    //     group = bh_to_group[batch * num_heads + head]
    // Group g occupies
    //     slice arrays  : [group_slice_offsets[g], group_slice_offsets[g+1])
    //     vbatch_to_slice: [group_vb_offsets[g],  group_vb_offsets[g+1])
    //     row_to_slice / row_to_vbatch_start / row_to_vbatch_end:
    //                     [g * seqlen_q, (g + 1) * seqlen_q)
    //     mask_bits     : rows [g * seqlen_q, (g + 1) * seqlen_q)
    // Index convention in the flat tables (set by build_bh_layout):
    // row_to_slice / vbatch_to_slice entries are FLAT (global) slice
    // indices and row_to_vbatch_* entries are FLAT vbatch indices.
    // CTAs rebase a local copy of this struct with rebase_slices() — which
    // shifts the array BASES and narrows the counts but keeps the vbatch
    // tables globally based — resolve slice_idx from the flat entry minus
    // group_slice_offsets[g], and then use it unchanged everywhere
    // downstream.
    int const *__restrict__ bh_to_group;         // [num_batch * num_heads]
    int const *__restrict__ group_slice_offsets; // [num_groups + 1] prefix sum
    int const *__restrict__ group_vb_offsets;    // [num_groups + 1] prefix sum
    int num_groups;

    // ── Trivial single-slice specialization (host-proven) ─────────────────
    // Set by the SDPA glue when it can PROVE on the host that the mask is
    // exactly one slice covering the whole (seqlen_q, seqlen_k) rectangle:
    //   0 = generic slice mechanism (descriptor arrays consulted as usual)
    //   1 = one FULL slice over the whole rectangle (dense attention)
    //   2 = one CAUSAL slice over the whole rectangle
    // When non-zero the fwd kernel runs the zone-split mainloop with
    // host-constant metadata (TrivialSliceMeta) and NEVER dereferences the
    // descriptor/row arrays above — zero global loads for slice resolution,
    // and the no-mask zone can be proven from seqlens alone (the generic
    // alignment guard would otherwise force mask.apply() on every block
    // whenever seqlen_k % kBlockN != 0).
    // Default 0: the bwd impl never sets these (the bwd trivial fast path
    // was removed — its LoopQ lambda refactor regressed the generic path),
    // so unwritten params stay safely on the generic machinery.
    int trivial_mask = 0;
    // Diagonal offset of the trivial CAUSAL slice (visible iff k <= q + d);
    // unused for FULL.
    int trivial_diagonal = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Extended forward parameters for flex flash attention.
// Inherits all standard Flash_fwd_params fields (QKV pointers, strides,
// dimensions, etc.) and adds slice-specific fields.
struct FlexFlashAttentionFwdParams : public Flash_fwd_params {
    // Slice decomposition (borrows AttnSliceParams pointers).
    AttnSliceParams slices;

    // Flag checked by launch template to select the flex flash attention path.
    bool is_flex_flash_attention;

    // ── Feature flags (runtime dispatch in the launch template) ──
    // Inner-loop direction: false = MaxToMin (default), true = MinToMax.
    bool inner_min_to_max;
    // Dynamic persistent tile scheduler (FlexFlashTileSchedulerSM80).
    bool persistent_scheduler;

    // kBlockM=128 hint from the caller (which knows the mask structure on
    // the host).  has_hint=false → launch template falls back to its own
    // shape-based heuristic.  has_hint=true → use kblockm128_value.
    // Large-span diagonal slices run pathologically slow at kBlockM=128, so
    // the Python layer sets value=false for those masks.
    bool kblockm128_has_hint;
    bool kblockm128_value;

    // kBlockN=64 override for short-K-span dense slice sets (block-diagonal
    // document masks): kBlockN=96 pads the last K-block of each slice by up
    // to 32 cols.  Host-computed hint, only consulted when kBlockM=64.
    bool kblockn64;

    // ── CUDA graph capture RNG (see the dropout section of
    // flex_flash_attention_api.cpp) ──
    // During capture the host cannot draw a Philox (seed, offset) pair, so
    // the kernel resolves its stream from these device pointers at every
    // replay (PyTorch refreshes them before each replay) and publishes the
    // pair it used into rng_state_dev for the captured bwd graph.  All
    // nullptr/0 in eager (by-value rng_state stays in effect).
    const unsigned long long* philox_seed_ptr;
    const unsigned long long* philox_offset_ptr;
    unsigned long long intragraph_offset;
    unsigned long long* rng_state_dev;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Extended backward parameters for flex flash attention.
// Inherits all Flash_bwd_params fields (dO/dQKV pointers, dQ/dK/dV fp32
// accum buffers, dsoftmax_sum, softmax_lse_log2, etc.) and adds the slice
// decomposition.  dK/dV always use the fp32 accum + atomicAdd path (the
// GQA-style epilogue) so Q-overlapping slices (RangeMerge) and GQA are
// both race-free.
struct FlexFlashAttentionBwdParams : public Flash_bwd_params {
    // Slice decomposition (borrows AttnSliceParams pointers).
    AttnSliceParams slices;

    // Flag for symmetry with FlexFlashAttentionFwdParams (bwd dispatch is a
    // separate entry point, so this is informational only).
    bool is_flex_flash_attention;

    // ── Feature flags (runtime dispatch in the launch template) ──
    // Inner-loop direction: false = MaxToMin (default), true = MinToMax.
    bool inner_min_to_max;
    // Dynamic persistent tile scheduler (FlexFlashTileSchedulerSM80).
    bool persistent_scheduler;
    // Bwd mainloop variant: false = LoopQ (CTA over n_block), true = LoopK
    // (CTA over m_block, dQ accumulated in registers across slices).
    bool use_loop_k;
    // Route dK/dV through an fp32 workspace + separate reduce kernel.
    // Only supported together with use_loop_k.
    bool reduce_kv;

    // ── CUDA graph capture RNG (see the dropout section of
    // flex_flash_attention_api.cpp) ──
    // Captured fwd published its (seed, offset) pair into a device
    // rng_state tensor; bwd replays the same stream by pointing
    // philox_seed_ptr/philox_offset_ptr at its two elements (intragraph
    // offset 0).  nullptr in eager (params.rng_state host values instead).
    const unsigned long long* philox_seed_ptr;
    const unsigned long long* philox_offset_ptr;
    unsigned long long intragraph_offset;
    unsigned long long* rng_state_dev;   // unused in bwd, kept for symmetry
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper: compute the K-column range [k_min, k_max) for a given slice.
// This is the range of K columns that can contribute non-zero attention for
// any Q row belonging to this slice.
inline __host__ __device__
void slice_k_range(AttnSliceParams const& s, int slice_idx,
                   int &k_min, int &k_max) {
    k_min = s.k_starts[slice_idx];
    k_max = s.k_ends[slice_idx];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper: rebase a COPY of the slice params onto one (batch, head) pair's
// layout group (P3 heterogeneous masks).  Returns the input unchanged when
// bh_to_group == nullptr (single shared layout).  The returned struct owns
// no memory; it only rebases borrowed pointers, so it is valid for exactly
// as long as the original params.
//
// NOTE: vbatch_to_slice is deliberately NOT pointer-shifted: its entries
// (and those of row_to_vbatch_start/end) are FLAT vbatch indices, so the
// lookup stays globally based and only the group's [v_off, v_off+num_vb)
// window is ever addressed.  Slice entries in row_to_slice / vbatch_to_slice
// are FLAT too — the caller converts them to group-local indices by
// subtracting group_slice_offsets[g] when resolving slice_idx.
inline __device__
AttnSliceParams rebase_slices(AttnSliceParams p, int bh_idx, int seqlen_q) {
    if (p.bh_to_group == nullptr) {
        return p;
    }
    int const g = p.bh_to_group[bh_idx];
    int const s_off = p.group_slice_offsets[g];
    int const v_off = p.group_vb_offsets[g];
    long long const r_off = (long long)g * seqlen_q;

    p.q_starts           += s_off;
    p.q_ends             += s_off;
    p.k_starts           += s_off;
    p.k_ends             += s_off;
    p.mask_types         += s_off;
    p.diagonal_offsets   += s_off;
    p.band_widths        += s_off;
    p.row_to_slice       += r_off;
    if (p.vbatch_to_slice != nullptr) {
        // Keep the GLOBAL base (entries are flat vbatch indices); only the
        // row tables move to this group's rows.
        p.row_to_vbatch_start  += r_off;
        p.row_to_vbatch_end    += r_off;
    }
    if (p.mask_bits != nullptr) {
        p.mask_bits += r_off * p.mask_row_stride;
    }
    p.num_slices   = p.group_slice_offsets[g + 1] - s_off;
    p.num_vbatches = p.group_vb_offsets[g + 1] - v_off;
    return p;
}
