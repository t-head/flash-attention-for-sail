// Layer-2 decomposition, phase 2: segment classification → slice emission.
//
// Device-side equivalent of mask_decomp.py _classify_segments +
// _single_row_fallback + the per-layer empty-slice policy of
// _decompose_with_holes.  Input: the segment descriptors produced by
// launch_decomp_segment (actual count in n_segs_p, device-side); output:
// slice 7-tuples APPENDED to the desc arrays starting at the device
// counter *n_slices_p (which this launcher advances by the number of
// slices emitted).
//
// Per-segment rules (mirror _classify_segments exactly):
//   ks0 == 0 && ke0 == 0  → zero-K FULL slice; emitted on layer 0 only
//                           (deeper layers drop it: k_end <= k_start)
//   (dks,dke) == (0,0)    → FULL  [ks0, ke0)
//   (dks,dke) == (0,1)    → CAUSAL,    offset = ke0-1-s, k_end = ke0+(len-1)
//   (dks,dke) == (1,0)    → INVCAUSAL, offset = ks0-s
//   (dks,dke) == (1,1)    → BICAUSAL,  offset = ke0-1-s, bw = ke0-1-ks0,
//                                      k_end = ke0+(len-1)
//   anything else         → fallback expansion: one single-row FULL slice
//                           per row of the segment, k bounds gathered from
//                           the peel output (len slices emitted)
// Layer > 0 additionally drops every slice with k_end <= k_start (the
// _decompose_with_holes filter): peeled deeper-layer rows can carry
// degenerate intervals (ke <= ks while ks != ke, e.g. a trailing row of a
// CAUSAL segment), which classify would otherwise keep.
//
// Variable-length output (fallback) is handled by an int-value scan of the
// per-segment emission counts followed by a scatter.  If the cumulative
// slice count would exceed `cap`, *supported_p is set to 0 (strategy A:
// honest "not supported" instead of silent truncation) and the overflowed
// slices are not written.
//
// PPU discipline: only naturally-aligned 4-byte accesses; no vector loads.
// Torch-header-free (repo convention); op wrapper lives in
// flex_flash_attention_api.cpp.

#include <hggc_runtime.h>
#include <cstdint>

namespace {

constexpr int kBlock = 256;
constexpr int kMaxScanBlocks = 256;  // single-block phase-B capacity
// Cascaded phase B lifts the cap to 256*256 blocks (seqlen_q <= 16777216);
// nb <= kMaxScanBlocks keeps the original one-block phase B verbatim so
// every existing shape compiles and allocates identically.
constexpr int kMaxCascadeBlocks = kMaxScanBlocks * kMaxScanBlocks;

// Tail ints after the scan array: block totals (+ group sums when
// cascading).  Equal to the old kMaxScanBlocks constant for nb <= 256.
inline size_t scan_tail_ints(int nb) {
  if (nb <= kMaxScanBlocks) return (size_t)kMaxScanBlocks;
  return (size_t)nb + (size_t)((nb + kBlock - 1) / kBlock);
}

// Slice type values — must match AttnSliceType in attn_slice.h.
constexpr int SLICE_FULL = 0;
constexpr int SLICE_CAUSAL = 1;
constexpr int SLICE_INVCAUSAL = 2;
constexpr int SLICE_BICAUSAL = 3;

// ── block-level inclusive int scan (additive) ────────────────────────────
__device__ __forceinline__ int block_scan_incl(int val, int* smem,
                                               int& block_total) {
  int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int const nwarp = blockDim.x >> 5;
  int x = val;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    int const y = __shfl_up_sync(0xffffffffu, x, off);
    if (lane >= off) x += y;
  }
  if (lane == 31) smem[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = (lane < nwarp) ? smem[lane] : 0;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int const y = __shfl_up_sync(0xffffffffu, w, off);
      if (lane >= off) w += y;
    }
    if (lane < nwarp) smem[lane] = w;
  }
  __syncthreads();
  int const base = warp ? smem[warp - 1] : 0;
  block_total = smem[nwarp - 1];
  return x + base;
}

// ── multi-block int-value scan (3 kernels) ───────────────────────────────
__global__ void scan_int_phase_a(int const* __restrict__ vals, int const n,
                                 int* __restrict__ out,
                                 int* __restrict__ totals) {
  __shared__ int smem[kBlock / 32];
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  int const v = (i < n) ? vals[i] : 0;
  int total;
  int const incl = block_scan_incl(v, smem, total);
  if (threadIdx.x == 0) totals[blockIdx.x] = total;
  if (i < n) out[i] = incl;
}

__global__ void scan_int_phase_b(int* __restrict__ totals, int const nb) {
  __shared__ int smem[kBlock / 32];
  int const x = threadIdx.x;
  int const v = (x < nb) ? totals[x] : 0;
  int total;
  int const incl = block_scan_incl(v, smem, total);
  if (x < nb) totals[x] = incl - v;  // exclusive base for block x
}

__global__ void scan_int_phase_c(int* __restrict__ out, int const n,
                                 int const* __restrict__ totals) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] += totals[blockIdx.x];
}

// ── cascaded phase B for nb > kMaxScanBlocks (seqlen_q > 65536) ─────────
__global__ void scan_int_phase_b1(int* __restrict__ totals, int const nb,
                                  int* __restrict__ gtotals) {
  __shared__ int smem[kBlock / 32];
  int const x = threadIdx.x;
  int const gi = blockIdx.x * kBlock + x;
  int const v = (gi < nb) ? totals[gi] : 0;
  int total;
  int const incl = block_scan_incl(v, smem, total);
  if (x == 0) gtotals[blockIdx.x] = total;
  if (gi < nb) totals[gi] = incl - v;  // in-group exclusive base
}

__global__ void scan_int_phase_b2(int* __restrict__ gtotals,
                                  int const ngroups) {
  __shared__ int smem[kBlock / 32];
  int const x = threadIdx.x;
  int const v = (x < ngroups) ? gtotals[x] : 0;
  int total;
  int const incl = block_scan_incl(v, smem, total);
  if (x < ngroups) gtotals[x] = incl - v;
}

__global__ void scan_int_phase_b3(int* __restrict__ totals, int const nb,
                                  int const* __restrict__ gtotals) {
  int const gi = blockIdx.x * kBlock + threadIdx.x;
  if (gi >= nb) return;
  totals[gi] += gtotals[blockIdx.x];
}

void scan_ints(int const* vals, int n, int* out, int* totals,
               hggcStream_t stream) {
  int const nb = (n + kBlock - 1) / kBlock;
  scan_int_phase_a<<<nb, kBlock, 0, stream>>>(vals, n, out, totals);
  if (nb <= kMaxScanBlocks) {
    scan_int_phase_b<<<1, kBlock, 0, stream>>>(totals, nb);
  } else {
    int const ngroups = (nb + kBlock - 1) / kBlock;
    scan_int_phase_b1<<<ngroups, kBlock, 0, stream>>>(totals, nb, totals + nb);
    scan_int_phase_b2<<<1, kBlock, 0, stream>>>(totals + nb, ngroups);
    scan_int_phase_b3<<<ngroups, kBlock, 0, stream>>>(totals, nb, totals + nb);
  }
  scan_int_phase_c<<<nb, kBlock, 0, stream>>>(out, n, totals);
}

// ── per-segment classified descriptor (host/device shared logic) ─────────
struct ClSlice {
  int k_start, k_end, type, offset, bw;
};

__device__ __forceinline__ ClSlice classify_seg(int s, int e, int ks0,
                                                int ke0, int dks, int dke) {
  ClSlice r;
  if (ks0 == 0 && ke0 == 0) {
    r.k_start = 0; r.k_end = 0; r.type = SLICE_FULL; r.offset = 0; r.bw = 0;
  } else if (dks == 0 && dke == 0) {
    r.k_start = ks0; r.k_end = ke0; r.type = SLICE_FULL; r.offset = 0;
    r.bw = 0;
  } else if (dks == 0 && dke == 1) {
    r.k_start = ks0; r.k_end = ke0 + (e - s - 1); r.type = SLICE_CAUSAL;
    r.offset = ke0 - 1 - s; r.bw = 0;
  } else if (dks == 1 && dke == 0) {
    r.k_start = ks0; r.k_end = ke0; r.type = SLICE_INVCAUSAL;
    r.offset = ks0 - s; r.bw = 0;
  } else if (dks == 1 && dke == 1) {
    r.k_start = ks0; r.k_end = ke0 + (e - s - 1); r.type = SLICE_BICAUSAL;
    r.offset = ke0 - 1 - s; r.bw = ke0 - 1 - ks0;
  } else {
    r.type = -1;  // fallback expansion marker
    r.k_start = 0; r.k_end = 0; r.offset = 0; r.bw = 0;
  }
  return r;
}

// ── pass 1: per-segment emission count ───────────────────────────────────
__global__ void emit_count_kernel(int const* __restrict__ seg_q_start,
                                  int const* __restrict__ seg_q_end,
                                  int const* __restrict__ seg_ks0,
                                  int const* __restrict__ seg_ke0,
                                  int const* __restrict__ seg_dks,
                                  int const* __restrict__ seg_dke,
                                  int const* __restrict__ n_segs_p,
                                  int const* __restrict__ ks,
                                  int const* __restrict__ ke,
                                  int const layer, int const seqlen_q,
                                  int const* __restrict__ done,
                                  int* __restrict__ cnt) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= seqlen_q) return;
  if (done && *done) {
    cnt[i] = 0;  // gated layer emits nothing; keep the scan input zeroed
    return;
  }
  int const n_segs = *n_segs_p;
  if (i >= n_segs) {
    cnt[i] = 0;
    return;
  }
  int const s = seg_q_start[i], e = seg_q_end[i];
  ClSlice const c = classify_seg(s, e, seg_ks0[i], seg_ke0[i], seg_dks[i],
                                 seg_dke[i]);
  if (c.type >= 0) {
    // Layer > 0 drops degenerate (k_end <= k_start) slices, matching the
    // _decompose_with_holes filter on the Python side.
    cnt[i] = (layer > 0 && c.k_end <= c.k_start) ? 0 : 1;
    return;
  }
  // Fallback expansion: one slice per row.  Layer > 0 keeps only rows
  // with ke > ks (the k_end > k_start filter; peel convention makes this
  // a no-op on real data, but equivalence with the Python filter demands
  // it).
  if (layer == 0) {
    cnt[i] = e - s;
  } else {
    int m = 0;
    for (int q = s; q < e; ++q) m += (ke[q] > ks[q]) ? 1 : 0;
    cnt[i] = m;
  }
}

// ── pass 2: capacity gate ────────────────────────────────────────────────
// cnt slots beyond n_segs are zero, so the inclusive scan's last element
// holds the layer total regardless of the device-side count.
__global__ void cap_check_kernel(int const* __restrict__ cnt_scan,
                                 int const n, int const* __restrict__ base_p,
                                 int const cap, int* __restrict__ supported,
                                 int const* __restrict__ done) {
  if (done && *done) return;
  if (*base_p + cnt_scan[n - 1] > cap) *supported = 0;
}

// ── pass 3: scatter the slices ───────────────────────────────────────────
__global__ void scatter_kernel(int const* __restrict__ seg_q_start,
                               int const* __restrict__ seg_q_end,
                               int const* __restrict__ seg_ks0,
                               int const* __restrict__ seg_ke0,
                               int const* __restrict__ seg_dks,
                               int const* __restrict__ seg_dke,
                               int const* __restrict__ n_segs_p,
                               int const* __restrict__ ks,
                               int const* __restrict__ ke,
                               int const* __restrict__ cnt_scan,
                               int const* __restrict__ base_p,
                               int const cap, int const layer,
                               int const seqlen_q,
                               int const* __restrict__ done,
                               int* __restrict__ q_starts,
                               int* __restrict__ q_ends,
                               int* __restrict__ k_starts,
                               int* __restrict__ k_ends,
                               int* __restrict__ mask_types,
                               int* __restrict__ diag_offsets,
                               int* __restrict__ band_widths) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= seqlen_q) return;
  if (done && *done) return;
  int const n_segs = *n_segs_p;
  if (i >= n_segs) return;
  int const s = seg_q_start[i], e = seg_q_end[i];
  int const cnt = cnt_scan[i] - (i ? cnt_scan[i - 1] : 0);
  if (cnt == 0) return;
  int const off = *base_p + cnt_scan[i] - cnt;  // exclusive write base
  if (off + cnt > cap) return;  // overflow: supported already cleared

  ClSlice const c = classify_seg(s, e, seg_ks0[i], seg_ke0[i], seg_dks[i],
                                 seg_dke[i]);
  if (c.type >= 0) {
    q_starts[off] = s; q_ends[off] = e; k_starts[off] = c.k_start;
    k_ends[off] = c.k_end; mask_types[off] = c.type;
    diag_offsets[off] = c.offset; band_widths[off] = c.bw;
    return;
  }
  // Unknown pattern: expand to single-row FULL slices with per-row bounds
  // (layer > 0 skips degenerate rows, matching emit_count_kernel).
  int w = off;
  for (int q = s; q < e; ++q) {
    if (layer > 0 && ke[q] <= ks[q]) continue;  // degenerate-row filter
    q_starts[w] = q; q_ends[w] = q + 1;
    k_starts[w] = ks[q]; k_ends[w] = ke[q];
    mask_types[w] = SLICE_FULL; diag_offsets[w] = 0; band_widths[w] = 0;
    ++w;
  }
}

// ── pass 4: advance the global slice counter ────────────────────────────
__global__ void add_total_kernel(int const* __restrict__ cnt_scan,
                                 int const n, int* __restrict__ base_p,
                                 int const* __restrict__ done) {
  if (done && *done) return;
  *base_p += cnt_scan[n - 1];
}

struct ClassifyScratch {
  int* cnt;
  int* cnt_scan;  // + kMaxScanBlocks tail ints for block totals
};

size_t decomp_classify_scratch_bytes(int seqlen_q) {
  size_t const n = (size_t)seqlen_q;
  int const nb = (seqlen_q + kBlock - 1) / kBlock;
  return n * sizeof(int) + (n + scan_tail_ints(nb)) * sizeof(int) + 512;
}

}  // namespace

size_t decomp_classify_scratch_size(int seqlen_q);
hggcError_t launch_decomp_classify(
    int const* seg_q_start, int const* seg_q_end, int const* seg_ks0,
    int const* seg_ke0, int const* seg_dks, int const* seg_dke,
    int const* n_segs_p, int const* ks, int const* ke, int layer,
    int seqlen_q, int* n_slices_p, int* q_starts, int* q_ends,
    int* k_starts, int* k_ends, int* mask_types, int* diag_offsets,
    int* band_widths, int cap, int* supported_p, int const* done,
    void* scratch, size_t scratch_bytes, hggcStream_t stream);

size_t decomp_classify_scratch_size(int seqlen_q) {
  return decomp_classify_scratch_bytes(seqlen_q > 0 ? seqlen_q : 1);
}

// `done` (nullable): layered-decompose gate; when *done != 0 this launch
// emits zero slices and leaves *n_slices_p untouched.
hggcError_t launch_decomp_classify(
    int const* seg_q_start, int const* seg_q_end, int const* seg_ks0,
    int const* seg_ke0, int const* seg_dks, int const* seg_dke,
    int const* n_segs_p, int const* ks, int const* ke, int layer,
    int seqlen_q, int* n_slices_p, int* q_starts, int* q_ends,
    int* k_starts, int* k_ends, int* mask_types, int* diag_offsets,
    int* band_widths, int cap, int* supported_p, int const* done,
    void* scratch, size_t scratch_bytes, hggcStream_t stream) {
  if (seqlen_q <= 0) return hggcSuccess;
  if (scratch_bytes < decomp_classify_scratch_bytes(seqlen_q))
    return hggcErrorInvalidValue;
  if ((seqlen_q + kBlock - 1) / kBlock > kMaxCascadeBlocks)
    return hggcErrorInvalidValue;

  char* p = reinterpret_cast<char*>(scratch);
  uintptr_t a = (uintptr_t)p;
  a = (a + 255) & ~(uintptr_t)255;
  ClassifyScratch sc;
  sc.cnt = reinterpret_cast<int*>(a);
  sc.cnt_scan = sc.cnt + seqlen_q;

  int const n = seqlen_q;
  int const nb = (n + kBlock - 1) / kBlock;
  emit_count_kernel<<<nb, kBlock, 0, stream>>>(
      seg_q_start, seg_q_end, seg_ks0, seg_ke0, seg_dks, seg_dke, n_segs_p,
      ks, ke, layer, n, done, sc.cnt);
  scan_ints(sc.cnt, n, sc.cnt_scan, sc.cnt_scan + n, stream);
  cap_check_kernel<<<1, 1, 0, stream>>>(sc.cnt_scan, n, n_slices_p, cap,
                                        supported_p, done);
  scatter_kernel<<<nb, kBlock, 0, stream>>>(
      seg_q_start, seg_q_end, seg_ks0, seg_ke0, seg_dks, seg_dke, n_segs_p,
      ks, ke, sc.cnt_scan, n_slices_p, cap, layer, n, done, q_starts,
      q_ends, k_starts, k_ends, mask_types, diag_offsets, band_widths);
  add_total_kernel<<<1, 1, 0, stream>>>(sc.cnt_scan, n, n_slices_p, done);
  return hggcGetLastError();
}
