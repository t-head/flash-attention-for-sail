// Layer-2 decomposition, phase 3: merge adjacent slices, tile-align, and
// build the RangeMerge (vbatch) layout + row tables.
//
// Device-side equivalents of mask_decomp.py merge_adjacent_slices,
// align_slices_to_tiles, and build_merge_layout.  Operates on the
// accumulated pre-align slice list (post-classify, all layers) and fills
// the final desc arrays.
//
// Determinism notes (mirroring the Python orderings exactly):
//  - merge is a sequential compaction in slice-index order;
//  - aligned fragments are emitted grouped by source slice in slice-index
//    order (align_slices_to_tiles order), so fragment index == final
//    slice index;
//  - vbatches within an m_block are ordered by slice index (the cov
//    matrix's row-major (m, s) order), assigned by a sequential scatter;
//  - row_to_slice[row] = the LOWEST-index slice covering that row (the
//    attn_slice.h fallback reference "first covering slice per row"), -1
//    for uncovered rows; computed from the block's vbatch list, so it
//    stays well-defined even for multi-layer (Q-overlapping) descs.
//
// The sequential kernels (merge / block-count scan / vb scatter) are cheap
// for structured masks (slice counts in the hundreds); their worst case is
// the degenerate fallback-explosion masks that approach the slice cap,
// documented as acceptable (decomposition of such masks is bandwidth-bound
// upstream anyway).
//
// PPU discipline: naturally-aligned 4-byte accesses only; no vector loads.
// Torch-header-free; op wrapper lives in flex_flash_attention_api.cpp.

#include <hggc_runtime.h>
#include <cstdint>

namespace {

// ── stage 1: merge adjacent slices (single thread, index order) ──────────
__global__ void merge_kernel(int const* __restrict__ in_q0,
                             int const* __restrict__ in_q1,
                             int const* __restrict__ in_k0,
                             int const* __restrict__ in_k1,
                             int const* __restrict__ in_ty,
                             int const* __restrict__ in_df,
                             int const* __restrict__ in_bw,
                             int const* __restrict__ n_in_p,
                             int* __restrict__ out_q0, int* __restrict__ out_q1,
                             int* __restrict__ out_k0, int* __restrict__ out_k1,
                             int* __restrict__ out_ty, int* __restrict__ out_df,
                             int* __restrict__ out_bw,
                             int* __restrict__ n_out_p) {
  int const n = *n_in_p;
  int m = 0;
  for (int i = 0; i < n; ++i) {
    if (m > 0 && out_q1[m - 1] == in_q0[i] && out_ty[m - 1] == in_ty[i] &&
        out_k0[m - 1] == in_k0[i] && out_k1[m - 1] == in_k1[i] &&
        out_df[m - 1] == in_df[i] && out_bw[m - 1] == in_bw[i]) {
      out_q1[m - 1] = in_q1[i];  // extend the previous slice
    } else {
      out_q0[m] = in_q0[i]; out_q1[m] = in_q1[i];
      out_k0[m] = in_k0[i]; out_k1[m] = in_k1[i];
      out_ty[m] = in_ty[i]; out_df[m] = in_df[i]; out_bw[m] = in_bw[i];
      ++m;
    }
  }
  *n_out_p = m;
}

// ── stage 2: tile alignment (fragment per kBlockM tile, sequential) ──────
// Fragments are emitted grouped by source slice in index order; fragment
// index is therefore the final slice index.  K bounds / offset / band are
// copied verbatim (no K-side alignment), matching align_slices_to_tiles.
__global__ void frag_emit_kernel(int const* __restrict__ mq0,
                                 int const* __restrict__ mq1,
                                 int const* __restrict__ mk0,
                                 int const* __restrict__ mk1,
                                 int const* __restrict__ mty,
                                 int const* __restrict__ mdf,
                                 int const* __restrict__ mbw,
                                 int const* __restrict__ n_merged_p,
                                 int const kblock_m, int const cap,
                                 int* __restrict__ q0, int* __restrict__ q1,
                                 int* __restrict__ k0, int* __restrict__ k1,
                                 int* __restrict__ ty, int* __restrict__ df,
                                 int* __restrict__ bw,
                                 int* __restrict__ n_frags_p,
                                 int* __restrict__ supported) {
  int const n = *n_merged_p;
  int w = 0;
  bool overflow = false;
  for (int i = 0; i < n; ++i) {
    int qs = mq0[i];
    int const qe = mq1[i];
    while (qs < qe) {
      int const nb = ((qs / kblock_m) + 1) * kblock_m;
      int const fe = (nb < qe) ? nb : qe;
      if (w >= cap) {
        overflow = true;
        break;
      }
      q0[w] = qs; q1[w] = fe; k0[w] = mk0[i]; k1[w] = mk1[i];
      ty[w] = mty[i]; df[w] = mdf[i]; bw[w] = mbw[i];
      ++w;
      qs = fe;
    }
    if (overflow) break;
  }
  *n_frags_p = w;
  if (overflow) *supported = 0;
}

// ── stage 3: vbatch layout ───────────────────────────────────────────────
// 3a: per-block covering-fragment counts (parallel atomicAdd; the count
//     value itself is order-independent).
__global__ void block_count_kernel(int const* __restrict__ q0,
                                   int const* __restrict__ n_frags_p,
                                   int const kblock_m, int const n_frags_cap,
                                   int* __restrict__ block_counts) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  int const n = *n_frags_p;
  if (i >= n || i >= n_frags_cap) return;
  atomicAdd(&block_counts[q0[i] / kblock_m], 1);
}

// 3b: exclusive prefix sum of block counts + total (single thread; the
//     block count is seqlen_q/kBlockM, tiny).  Also fills the per-row
//     vbatch ranges: coverage is constant within an m_block.
__global__ void vb_scan_rows_kernel(int* __restrict__ block_counts,
                                    int const num_blocks,
                                    int const seqlen_q, int const kblock_m,
                                    int* __restrict__ row_to_vb_start,
                                    int* __restrict__ row_to_vb_end,
                                    int* __restrict__ n_vb_p) {
  int acc = 0;
  for (int b = 0; b < num_blocks; ++b) {
    int const c = block_counts[b];
    block_counts[b] = acc;  // now the exclusive base (vb_first)
    acc += c;
  }
  for (int r = 0; r < seqlen_q; ++r) {
    int b = r / kblock_m;
    if (b >= num_blocks) b = num_blocks - 1;
    row_to_vb_start[r] = block_counts[b];
    row_to_vb_end[r] = (b + 1 < num_blocks) ? block_counts[b + 1] : acc;
  }
  *n_vb_p = acc;
}

// 3c: deterministic ordered scatter — fragment i takes the next free slot
//     of its block, so within-block order == slice index order (matches
//     build_merge_layout's row-major (m, s) enumeration).
__global__ void vb_scatter_kernel(int const* __restrict__ q0,
                                  int const* __restrict__ n_frags_p,
                                  int const kblock_m,
                                  int const* __restrict__ vb_first,
                                  int* __restrict__ rank_ctr,
                                  int* __restrict__ vbatch_to_slice) {
  int const n = *n_frags_p;
  for (int i = 0; i < n; ++i) {
    int const b = q0[i] / kblock_m;
    int const vb = vb_first[b] + rank_ctr[b];
    rank_ctr[b] += 1;
    vbatch_to_slice[vb] = i;
  }
}

// 3d: row_to_slice = lowest-index slice covering the row, -1 when nothing
//     covers it.  All covering slices of a row are among the row's block
//     vbatches, so scan that (short) list instead of all slices.
__global__ void row_to_slice_kernel(int const* __restrict__ row_to_vb_start,
                                    int const* __restrict__ row_to_vb_end,
                                    int const* __restrict__ vbatch_to_slice,
                                    int const* __restrict__ q0,
                                    int const* __restrict__ q1,
                                    int const seqlen_q,
                                    int* __restrict__ row_to_slice) {
  int const r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= seqlen_q) return;
  int const lo = row_to_vb_start[r], hi = row_to_vb_end[r];
  int best = -1;
  for (int vb = lo; vb < hi; ++vb) {
    int const s = vbatch_to_slice[vb];
    if (q0[s] <= r && r < q1[s] && (best < 0 || s < best)) best = s;
  }
  row_to_slice[r] = best;
}

constexpr int kBlock = 256;

}  // namespace

size_t decomp_layout_scratch_size(int seqlen_q, int kblock_m);
hggcError_t launch_decomp_layout(
    int const* pre_q0, int const* pre_q1, int const* pre_k0,
    int const* pre_k1, int const* pre_ty, int const* pre_df,
    int const* pre_bw, int const* n_pre_p, int seqlen_q, int kblock_m,
    int cap, int* m_q0, int* m_q1, int* m_k0, int* m_k1, int* m_ty,
    int* m_df, int* m_bw, int* n_merged_p, int* q0, int* q1, int* k0,
    int* k1, int* ty, int* df, int* bw, int* n_frags_p,
    int* vbatch_to_slice, int* row_to_slice, int* row_to_vb_start,
    int* row_to_vb_end, int* n_vb_p, int* supported_p, void* scratch,
    size_t scratch_bytes, hggcStream_t stream);

size_t decomp_layout_scratch_size(int seqlen_q, int kblock_m) {
  int const nb = (seqlen_q + kblock_m - 1) / kblock_m;
  return (size_t)(2 * nb) * sizeof(int) + 256;  // block_counts + rank_ctr
}

hggcError_t launch_decomp_layout(
    int const* pre_q0, int const* pre_q1, int const* pre_k0,
    int const* pre_k1, int const* pre_ty, int const* pre_df,
    int const* pre_bw, int const* n_pre_p, int seqlen_q, int kblock_m,
    int cap, int* m_q0, int* m_q1, int* m_k0, int* m_k1, int* m_ty,
    int* m_df, int* m_bw, int* n_merged_p, int* q0, int* q1, int* k0,
    int* k1, int* ty, int* df, int* bw, int* n_frags_p,
    int* vbatch_to_slice, int* row_to_slice, int* row_to_vb_start,
    int* row_to_vb_end, int* n_vb_p, int* supported_p, void* scratch,
    size_t scratch_bytes, hggcStream_t stream) {
  if (seqlen_q <= 0 || kblock_m <= 0) return cudaErrorInvalidValue;
  if (scratch_bytes < decomp_layout_scratch_size(seqlen_q, kblock_m))
    return cudaErrorInvalidValue;
  int const num_blocks = (seqlen_q + kblock_m - 1) / kblock_m;
  uintptr_t a = ((uintptr_t)scratch + 255) & ~(uintptr_t)255;
  int* block_counts = reinterpret_cast<int*>(a);
  int* rank_ctr = block_counts + num_blocks;

  hggcError_t err;
  merge_kernel<<<1, 1, 0, stream>>>(pre_q0, pre_q1, pre_k0, pre_k1, pre_ty,
                                    pre_df, pre_bw, n_pre_p, m_q0, m_q1,
                                    m_k0, m_k1, m_ty, m_df, m_bw,
                                    n_merged_p);
  frag_emit_kernel<<<1, 1, 0, stream>>>(m_q0, m_q1, m_k0, m_k1, m_ty, m_df,
                                        m_bw, n_merged_p, kblock_m, cap, q0,
                                        q1, k0, k1, ty, df, bw, n_frags_p,
                                        supported_p);
  err = cudaMemsetAsync(block_counts, 0,
                        (size_t)(2 * num_blocks) * sizeof(int), stream);
  if (err != hggcSuccess) return err;
  int const nb_grid = (cap + kBlock - 1) / kBlock;
  block_count_kernel<<<nb_grid, kBlock, 0, stream>>>(q0, n_frags_p, kblock_m,
                                                     cap, block_counts);
  vb_scan_rows_kernel<<<1, 1, 0, stream>>>(block_counts, num_blocks,
                                           seqlen_q, kblock_m,
                                           row_to_vb_start, row_to_vb_end,
                                           n_vb_p);
  vb_scatter_kernel<<<1, 1, 0, stream>>>(q0, n_frags_p, kblock_m,
                                         block_counts, rank_ctr,
                                         vbatch_to_slice);
  int const r_grid = (seqlen_q + kBlock - 1) / kBlock;
  row_to_slice_kernel<<<r_grid, kBlock, 0, stream>>>(
      row_to_vb_start, row_to_vb_end, vbatch_to_slice, q0, q1, seqlen_q,
      row_to_slice);
  return hggcGetLastError();
}
