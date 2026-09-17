// Layer-2 decomposition, phase 1: GPU run-length segmentation.
//
// Device-side equivalent of mask_decomp.py _segment_rows_gpu + the greedy
// anchor filter: rows are grouped into segments of constant (d_ks, d_ke)
// slope, with empty rows (ks == ke) poisoned into standalone segments and
// the "admit row s+1 unconditionally" greedy rule reproduced by dropping
// candidate boundaries immediately after each kept boundary (kept
// boundaries are >= 2 rows apart), EXEMPTING boundaries adjacent to an
// empty row (see the Python docstring for the full rationale).
//
// All intermediate state stays on device; only segment descriptors are
// produced (no CPU round trip), which is what makes the decomposition a
// single C++ op usable inside torch.compile graphs.
//
// Upper bound for the segment count: seqlen_q (every row its own segment).
// All output arrays are allocated to that bound; n_segs_out[0] holds the
// actual count on device, read by downstream device kernels.
//
// PPU discipline (probe-validated): every global access here is a 4-byte
// int or 1-byte byte at its natural alignment — no vector/uint4 loads.
// This TU stays torch-header-free (repo convention); the op wrapper lives
// in flex_flash_attention_api.cpp and calls the launcher below.

#include <hggc_runtime.h>
#include <cstdint>

namespace {

constexpr int kBlock = 256;
constexpr int kMaxScanBlocks = 256;  // single-block phase-B capacity
// Cascaded phase B (group scan + scan of group sums) lifts the block-count
// cap to 256*256, i.e. seqlen_q <= 16777216.  n <= kMaxScanBlocks keeps the
// original one-block phase B verbatim (codegen-identical for every existing
// shape); the cascade kernels below are separate instantiations that only
// compile into the binary, never shared with the short path.
constexpr int kMaxCascadeBlocks = kMaxScanBlocks * kMaxScanBlocks;

// Tail ints reserved after each scan array for block totals (+ group sums
// when cascading).  For nb <= kMaxScanBlocks this is exactly the old
// constant, so scratch sizes of all existing shapes are byte-identical.
inline size_t scan_tail_ints(int nb) {
  if (nb <= kMaxScanBlocks) return (size_t)kMaxScanBlocks;
  return (size_t)nb + (size_t)((nb + kBlock - 1) / kBlock);
}

// ── block-level inclusive scans (additive / max, int) ────────────────────
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

__device__ __forceinline__ int block_scan_max_incl(int val, int* smem,
                                                   int& block_max) {
  int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int const nwarp = blockDim.x >> 5;
  int x = val;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    int const y = __shfl_up_sync(0xffffffffu, x, off);
    if (lane >= off && y > x) x = y;
  }
  if (lane == 31) smem[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = (lane < nwarp) ? smem[lane] : -1;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int const y = __shfl_up_sync(0xffffffffu, w, off);
      if (lane >= off && y > w) w = y;
    }
    if (lane < nwarp) smem[lane] = w;
  }
  __syncthreads();
  int const base = warp ? smem[warp - 1] : -1;
  block_max = smem[nwarp - 1];
  return x > base ? x : base;
}

// ── pass 1: deltas + changed flags ───────────────────────────────────────
// changed[i] marks a candidate segment start at row i:
//   i == 0 : always
//   i == 1 : emptiness flip between rows 0 and 1 (the unconditional admit
//            of row 1 never crosses an empty/non-empty transition)
//   i >= 2 : (d_ks, d_ke) changed entering row i
// Deltas entering an EMPTY row are poisoned (> any real delta) so empty
// rows never join their neighbours' segments.
//
// Race-free formulation: changed[i] compares the delta entering row i with
// the delta entering row i-1, and BOTH are recomputed locally from ks/ke.
// (The earlier form read d_ks[i-2] — written by another thread of the same
// kernel — which is unordered across warps and reads uninitialized scratch;
// it passed probes only by timing luck.)  The d_ks/d_ke stores remain for
// seg_desc_kernel, which runs in a later stream-ordered kernel.
__global__ void delta_changed_kernel(int const* __restrict__ ks,
                                     int const* __restrict__ ke,
                                     int const seqlen_q, int const seqlen_k,
                                     int const* __restrict__ done,
                                     int* __restrict__ d_ks,
                                     int* __restrict__ d_ke,
                                     unsigned char* __restrict__ changed) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= seqlen_q) return;
  if (done && *done) return;  // residual emptied by an earlier peel layer
  if (i == 0) {
    changed[0] = 1;
    return;
  }
  bool const empty_prev = ks[i - 1] == ke[i - 1];
  bool const empty_cur = ks[i] == ke[i];
  int const poison = seqlen_k + seqlen_q + 1;
  int const dks = empty_cur ? poison : ks[i] - ks[i - 1];
  int const dke = empty_cur ? poison : ke[i] - ke[i - 1];
  d_ks[i - 1] = dks;
  d_ke[i - 1] = dke;
  if (i == 1) {
    changed[1] = (empty_prev != empty_cur) ? 1 : 0;
    return;
  }
  int const pdks = empty_prev ? poison : ks[i - 1] - ks[i - 2];
  int const pdke = empty_prev ? poison : ke[i - 1] - ke[i - 2];
  changed[i] = (dks != pdks || dke != pdke) ? 1 : 0;
}

// ── generic byte-flag inclusive scan, three kernels on one stream ────────
// Phase A: per-block inclusive scan + publish block total (lane 31).
__global__ void scan_phase_a(unsigned char const* __restrict__ flags,
                             int const n, int const* __restrict__ done,
                             int* __restrict__ out,
                             int* __restrict__ totals) {
  __shared__ int smem[kBlock / 32];
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  int const v = (i < n && !(done && *done)) ? (int)flags[i] : 0;
  int total;
  int const incl = block_scan_incl(v, smem, total);
  // Converged point: every thread holds the same `total`.
  if (threadIdx.x == 0) totals[blockIdx.x] = total;
  if (i < n) out[i] = incl;  // block-local for now; base added in phase C
}

// Phase B: single block scans the (<= kMaxScanBlocks) block totals into
// INCLUSIVE prefix sums of the PREVIOUS blocks: totals[x] becomes the sum
// of block totals [0, x), i.e. the add-back base for block x.
__global__ void scan_phase_b(int* __restrict__ totals, int const nb) {
  __shared__ int smem[kBlock / 32];
  int const x = threadIdx.x;
  int const v = (x < nb) ? totals[x] : 0;
  int total;
  int const incl = block_scan_incl(v, smem, total);
  if (x < nb) totals[x] = incl - v;  // exclusive base for block x
}

// Phase C: add the block base back.
__global__ void scan_phase_c(int* __restrict__ out, int const n,
                             int const* __restrict__ totals,
                             int const* __restrict__ done) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (done && *done) return;
  out[i] += totals[blockIdx.x];
}

// ── cascaded phase B for nb > kMaxScanBlocks (seqlen_q > 65536) ─────────
// B1: one block per group of 256 totals; in-group exclusive base written
//     back, group sum published to gtotals (= totals + nb).
__global__ void scan_phase_b1(int* __restrict__ totals, int const nb,
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

// B2: single block scans the (<= kMaxScanBlocks) group sums into exclusive
//     group bases.
__global__ void scan_phase_b2(int* __restrict__ gtotals, int const ngroups) {
  __shared__ int smem[kBlock / 32];
  int const x = threadIdx.x;
  int const v = (x < ngroups) ? gtotals[x] : 0;
  int total;
  int const incl = block_scan_incl(v, smem, total);
  if (x < ngroups) gtotals[x] = incl - v;
}

// B3: add the group base back into every in-group exclusive base.
__global__ void scan_phase_b3(int* __restrict__ totals, int const nb,
                              int const* __restrict__ gtotals) {
  int const gi = blockIdx.x * kBlock + threadIdx.x;
  if (gi >= nb) return;
  totals[gi] += gtotals[blockIdx.x];
}

void scan_flags(unsigned char const* flags, int n, int const* done,
                int* out, int* totals, hggcStream_t stream) {
  int const nb = (n + kBlock - 1) / kBlock;  // capped by caller allocation
  scan_phase_a<<<nb, kBlock, 0, stream>>>(flags, n, done, out, totals);
  if (nb <= kMaxScanBlocks) {
    scan_phase_b<<<1, kBlock, 0, stream>>>(totals, nb);
  } else {
    int const ngroups = (nb + kBlock - 1) / kBlock;
    scan_phase_b1<<<ngroups, kBlock, 0, stream>>>(totals, nb, totals + nb);
    scan_phase_b2<<<1, kBlock, 0, stream>>>(totals + nb, ngroups);
    scan_phase_b3<<<ngroups, kBlock, 0, stream>>>(totals, nb, totals + nb);
  }
  scan_phase_c<<<nb, kBlock, 0, stream>>>(out, n, totals, done);
}

// ── max-index scan: out[i] = max{ j <= i : flags[j] }, -1 if none ────────
__global__ void scan_max_phase_a(unsigned char const* __restrict__ flags,
                                 int const n, int const* __restrict__ done,
                                 int* __restrict__ out,
                                 int* __restrict__ totals) {
  __shared__ int smem[kBlock / 32];
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  int const v = (i < n && !(done && *done) && flags[i]) ? i : -1;
  int bmax;
  int const incl = block_scan_max_incl(v, smem, bmax);
  if (threadIdx.x == 0) totals[blockIdx.x] = bmax;
  if (i < n) out[i] = incl;
}

// Inclusive prefix max over the block maxima (single block, nb <= 256).
__global__ void scan_max_phase_b(int* __restrict__ totals, int const nb) {
  __shared__ int smem[kBlock / 32];
  int const x = threadIdx.x;
  int const v = (x < nb) ? totals[x] : -1;
  int bmax;
  int const incl = block_scan_max_incl(v, smem, bmax);
  if (x < nb) totals[x] = incl;
}

// ── cascaded phase B for the max-index scan (nb > kMaxScanBlocks) ────────
// B1: per-group inclusive prefix max; publish the group max.
__global__ void scan_max_phase_b1(int* __restrict__ totals, int const nb,
                                  int* __restrict__ gmax) {
  __shared__ int smem[kBlock / 32];
  int const x = threadIdx.x;
  int const gi = blockIdx.x * kBlock + x;
  int const v = (gi < nb) ? totals[gi] : -1;
  int bmax;
  int const incl = block_scan_max_incl(v, smem, bmax);
  if (x == 0) gmax[blockIdx.x] = bmax;
  if (gi < nb) totals[gi] = incl;  // in-group inclusive max
}

// B2: single-block inclusive prefix max over the group maxima (ngroups
//     <= kMaxScanBlocks); same inclusive semantics as scan_max_phase_b.
__global__ void scan_max_phase_b2(int* __restrict__ gmax, int const ngroups) {
  __shared__ int smem[kBlock / 32];
  int const x = threadIdx.x;
  int const v = (x < ngroups) ? gmax[x] : -1;
  int bmax;
  int const incl = block_scan_max_incl(v, smem, bmax);
  if (x < ngroups) gmax[x] = incl;
}

// B3: fold the previous groups' max into every in-group entry.
__global__ void scan_max_phase_b3(int* __restrict__ totals, int const nb,
                                  int const* __restrict__ gmax_incl) {
  int const g = blockIdx.x;
  if (g == 0) return;
  int const gi = g * kBlock + threadIdx.x;
  if (gi >= nb) return;
  int const base = gmax_incl[g - 1];
  if (base > totals[gi]) totals[gi] = base;
}

__global__ void scan_max_phase_c(int* __restrict__ out, int const n,
                                 int const* __restrict__ totals,
                                 int const* __restrict__ done) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || blockIdx.x == 0) return;
  if (done && *done) return;
  int const base = totals[blockIdx.x - 1];
  if (base > out[i]) out[i] = base;
}

void scan_flag_pos(unsigned char const* flags, int n, int const* done,
                   int* out, int* totals, hggcStream_t stream) {
  int const nb = (n + kBlock - 1) / kBlock;
  scan_max_phase_a<<<nb, kBlock, 0, stream>>>(flags, n, done, out, totals);
  if (nb <= kMaxScanBlocks) {
    scan_max_phase_b<<<1, kBlock, 0, stream>>>(totals, nb);
  } else {
    int const ngroups = (nb + kBlock - 1) / kBlock;
    scan_max_phase_b1<<<ngroups, kBlock, 0, stream>>>(totals, nb, totals + nb);
    scan_max_phase_b2<<<1, kBlock, 0, stream>>>(totals + nb, ngroups);
    scan_max_phase_b3<<<ngroups, kBlock, 0, stream>>>(totals, nb, totals + nb);
  }
  scan_max_phase_c<<<nb, kBlock, 0, stream>>>(out, n, totals, done);
}

// ── pass 2: compact candidates + exemption flag ──────────────────────────
// cand[] holds the changed rows ascending; exempt[j] marks candidate j as
// adjacent to an empty row (boundary kept regardless of the min-gap rule).
__global__ void cand_flags_kernel(unsigned char const* __restrict__ changed,
                                  int const* __restrict__ ks,
                                  int const* __restrict__ ke,
                                  int const seqlen_q,
                                  int const* __restrict__ done,
                                  int const* __restrict__ scan,
                                  int* __restrict__ cand,
                                  unsigned char* __restrict__ exempt) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= seqlen_q) return;
  if (done && *done) return;
  if (!changed[i]) return;
  int const pos = scan[i] - 1;  // inclusive scan → 1-based rank
  cand[pos] = i;
  if (i == 0) {
    exempt[pos] = 0;
    return;
  }
  bool const near_empty = (ks[i - 1] == ke[i - 1]) || (ks[i] == ke[i]);
  exempt[pos] = near_empty ? 1 : 0;
}

// ── pass 3: restart flags of the greedy filter ───────────────────────────
// Sequential rule: kept=[c0]; keep x iff x >= kept[-1]+2 or exempt(x).
// One-step reduction: if c_{j-1} was kept, c_j survives iff gap_j >= 2 or
// exempt_j; if c_{j-1} was dropped then c_{j-1} == kept[-1]+1 and c_j is
// kept unconditionally — hence keep_j = a_j || !keep_{j-1} with
// a_j = (gap_j >= 2) || exempt_j and keep_0 = true.  Positions with
// a_j = true "restart" the forced alternation, so keep_j iff
// (j - last_restart(j)) is even.  The restart flags are scanned below with
// a max-index scan to obtain last_restart(j) per candidate.
__global__ void restart_flags_kernel(
    unsigned char const* __restrict__ exempt, int const* __restrict__ cand,
    int const n, int const* __restrict__ done,
    int const* __restrict__ n_cand_p,
    unsigned char* __restrict__ rflag) {
  int const j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  if (done && *done) {
    rflag[j] = 0;
    return;
  }
  if (j >= *n_cand_p) {  // slots beyond the real candidates hold junk
    rflag[j] = 0;
    return;
  }
  bool const r = (j == 0) || (cand[j] - cand[j - 1] >= 2) || exempt[j];
  rflag[j] = r ? 1 : 0;
}

__global__ void keep_flags_kernel(int const* __restrict__ last_restart,
                                  int const n,
                                  int const* __restrict__ done,
                                  int const* __restrict__ n_cand_p,
                                  unsigned char* __restrict__ keep) {
  int const j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  if (done && *done) {
    keep[j] = 0;
    return;
  }
  if (j >= *n_cand_p) {
    keep[j] = 0;
    return;
  }
  keep[j] = (((j - last_restart[j]) & 1) == 0) ? 1 : 0;
}

// ── pass 5: compact kept candidates → final boundary list b ─────────────
__global__ void compact_b_kernel(unsigned char const* __restrict__ keep,
                                 int const* __restrict__ cand,
                                 int const n,
                                 int const* __restrict__ done,
                                 int const* __restrict__ n_cand_p,
                                 int const* __restrict__ keep_scan,
                                 int* __restrict__ b) {
  int const j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  if (done && *done) return;
  if (j >= *n_cand_p || !keep[j]) return;
  b[keep_scan[j] - 1] = cand[j];
}

// Copy scan[n-1] (device) into a counter output.
__global__ void copy_last_scan(int const* __restrict__ scan, int const n,
                               int const* __restrict__ done,
                               int* __restrict__ out) {
  if (done && *done) {
    *out = 0;
    return;
  }
  *out = scan[n - 1];
}

__global__ void write_int(int* p, int v) { *p = v; }

// write_int that honors the layered-decompose gate (seqlen_q == 1 path):
// gated ⇒ write 0 so downstream classify sees zero segments, never a
// stale count from the previous layer.
__global__ void write_int_gated(int* p, int v, int const* __restrict__ done) {
  *p = (done && *done) ? 0 : v;
}

// ── pass 6: per-segment descriptors (one thread per segment) ────────────
// Segment i covers rows [b[i], b[i+1]).  ks0/ke0 come from the segment's
// first row; slopes from its first row-pair (single-row segments get
// (0, 0), matching _segment_and_classify).  Grid launched at the seqlen_q
// upper bound; threads beyond n_segs exit after one uniform device read.
__global__ void seg_desc_kernel(int const* __restrict__ b,
                                int const* __restrict__ n_segs_p,
                                int const seqlen_q,
                                int const* __restrict__ done,
                                int const* __restrict__ ks,
                                int const* __restrict__ ke,
                                int const* __restrict__ d_ks,
                                int const* __restrict__ d_ke,
                                int* __restrict__ seg_q_start,
                                int* __restrict__ seg_q_end,
                                int* __restrict__ seg_ks0,
                                int* __restrict__ seg_ke0,
                                int* __restrict__ seg_dks,
                                int* __restrict__ seg_dke) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (done && *done) return;
  int const n_segs = *n_segs_p;
  if (i >= n_segs) return;
  int const s = b[i];
  int const e = (i + 1 < n_segs) ? b[i + 1] : seqlen_q;
  seg_q_start[i] = s;
  seg_q_end[i] = e;
  seg_ks0[i] = ks[s];
  seg_ke0[i] = ke[s];
  if (e - s >= 2) {
    seg_dks[i] = d_ks[s];
    seg_dke[i] = d_ke[s];
  } else {
    seg_dks[i] = 0;
    seg_dke[i] = 0;
  }
}

// ── scratch layout ───────────────────────────────────────────────────────
struct SegScratch {
  int* d_ks;
  int* d_ke;
  unsigned char* changed;
  int* scan0;      // inclusive scan of changed; tail room = block totals
  int* cand;
  unsigned char* exempt;
  unsigned char* rflag;
  int* rmax;       // max-index scan of rflag; tail room = block maxima
  unsigned char* keep;
  int* scan2;      // inclusive scan of keep
};

size_t decomp_segment_scratch_bytes(int seqlen_q) {
  size_t const n = (size_t)seqlen_q;
  int const nb = (seqlen_q + kBlock - 1) / kBlock;
  size_t bytes = 0;
  bytes += 2 * n * sizeof(int);   // d_ks, d_ke (length n-1)
  bytes += n;                     // changed
  bytes += 3 * (n + scan_tail_ints(nb)) * sizeof(int);  // scan0/rmax/scan2 (+totals)
  bytes += n * sizeof(int);       // cand
  bytes += 3 * n;                 // exempt, rflag, keep
  // layout_scratch() re-aligns EACH of its 10 allocations to 256 bytes —
  // worst case 255 padding bytes per take — so reserve 11 * 256, not a
  // small constant (the earlier +512 silently overflowed into adjacent
  // allocations and corrupted b[] on some allocator layouts).
  return bytes + 11 * 256;
}

SegScratch layout_scratch(void* scratch, int seqlen_q) {
  SegScratch s;
  char* p = reinterpret_cast<char*>(scratch);
  size_t const n = (size_t)seqlen_q;
  size_t const scan_n = n + scan_tail_ints((seqlen_q + kBlock - 1) / kBlock);
  auto take = [&](size_t bytes) {
    uintptr_t a = (uintptr_t)p;
    a = (a + 255) & ~(uintptr_t)255;
    p = reinterpret_cast<char*>(a) + bytes;
    return reinterpret_cast<void*>(a);
  };
  s.d_ks = reinterpret_cast<int*>(take((n - 1) * sizeof(int)));
  s.d_ke = reinterpret_cast<int*>(take((n - 1) * sizeof(int)));
  s.changed = reinterpret_cast<unsigned char*>(take(n));
  s.scan0 = reinterpret_cast<int*>(take(scan_n * sizeof(int)));
  s.cand = reinterpret_cast<int*>(take(n * sizeof(int)));
  s.exempt = reinterpret_cast<unsigned char*>(take(n));
  s.rflag = reinterpret_cast<unsigned char*>(take(n));
  s.rmax = reinterpret_cast<int*>(take(scan_n * sizeof(int)));
  s.keep = reinterpret_cast<unsigned char*>(take(n));
  s.scan2 = reinterpret_cast<int*>(take(scan_n * sizeof(int)));
  return s;
}

}  // namespace

size_t decomp_segment_scratch_size(int seqlen_q);
hggcError_t launch_decomp_segment(int const* ks, int const* ke,
                                  int seqlen_q, int seqlen_k, int* b,
                                  int* n_segs_out, int* seg_q_start,
                                  int* seg_q_end, int* seg_ks0,
                                  int* seg_ke0, int* seg_dks, int* seg_dke,
                                  int const* done, void* scratch,
                                  size_t scratch_bytes, hggcStream_t stream);

size_t decomp_segment_scratch_size(int seqlen_q) {
  return decomp_segment_scratch_bytes(seqlen_q);
}

// `done` (nullable): layered-decompose gate; when *done != 0 the residual
// was emptied by an earlier peel layer and this launch is a device-side
// no-op that zeroes n_segs_out so classify emits nothing.
hggcError_t launch_decomp_segment(int const* ks, int const* ke,
                                  int seqlen_q, int seqlen_k, int* b,
                                  int* n_segs_out, int* seg_q_start,
                                  int* seg_q_end, int* seg_ks0,
                                  int* seg_ke0, int* seg_dks, int* seg_dke,
                                  int const* done, void* scratch,
                                  size_t scratch_bytes, hggcStream_t stream) {
  if (scratch_bytes < decomp_segment_scratch_bytes(seqlen_q > 1 ? seqlen_q : 2))
    return hggcErrorInvalidValue;
  if (seqlen_q <= 0) {
    write_int<<<1, 1, 0, stream>>>(n_segs_out, 0);
    return hggcGetLastError();
  }
  if (seqlen_q == 1) {
    // Single row: one segment; slopes trivially (0, 0).
    write_int_gated<<<1, 1, 0, stream>>>(n_segs_out, 1, done);
    write_int_gated<<<1, 1, 0, stream>>>(b, 0, done);
    seg_desc_kernel<<<1, 1, 0, stream>>>(b, n_segs_out, seqlen_q, done, ks,
                                         ke, nullptr, nullptr, seg_q_start,
                                         seg_q_end, seg_ks0, seg_ke0,
                                         seg_dks, seg_dke);
    return hggcGetLastError();
  }
  int const n = seqlen_q;
  if ((n + kBlock - 1) / kBlock > kMaxCascadeBlocks) return hggcErrorInvalidValue;

  SegScratch sc = layout_scratch(scratch, seqlen_q);
  int const nb = (n + kBlock - 1) / kBlock;
  // Block totals reuse each scan array's tail room (scan_n = n + kMaxScanBlocks).
  hggcError_t err;

  // Pass 1: deltas + changed flags.
  delta_changed_kernel<<<nb, kBlock, 0, stream>>>(ks, ke, seqlen_q, seqlen_k,
                                                  done, sc.d_ks, sc.d_ke,
                                                  sc.changed);
  // Pass 2: candidates + exemption.
  scan_flags(sc.changed, n, done, sc.scan0, sc.scan0 + n, stream);
  err = hggcMemsetAsync(sc.exempt, 0, n, stream);
  if (err != hggcSuccess) return err;
  cand_flags_kernel<<<nb, kBlock, 0, stream>>>(sc.changed, ks, ke, seqlen_q,
                                               done, sc.scan0, sc.cand,
                                               sc.exempt);
  // n_cand lives on device (scan0[n-1]); copy it out so passes 3-5 can gate
  // their slot loops — slots [n_cand, n) hold junk cand entries that must
  // never be kept or counted.
  copy_last_scan<<<1, 1, 0, stream>>>(sc.scan0, n, done,
                                      n_segs_out);  // temp: n_cand
  // Pass 3: greedy filter via restart flags + max-index scan:
  // rflag[j] = restart position, rmax[j] = last restart at or before j,
  // keep_j iff (j - rmax[j]) even.  Then scan keep[] for compaction and
  // the segment count.
  restart_flags_kernel<<<nb, kBlock, 0, stream>>>(sc.exempt, sc.cand, n,
                                                  done, n_segs_out,
                                                  sc.rflag);
  scan_flag_pos(sc.rflag, n, done, sc.rmax, sc.rmax + n, stream);
  keep_flags_kernel<<<nb, kBlock, 0, stream>>>(sc.rmax, n, done, n_segs_out,
                                               sc.keep);
  scan_flags(sc.keep, n, done, sc.scan2, sc.scan2 + n, stream);
  compact_b_kernel<<<nb, kBlock, 0, stream>>>(sc.keep, sc.cand, n, done,
                                              n_segs_out, sc.scan2, b);
  copy_last_scan<<<1, 1, 0, stream>>>(sc.scan2, n, done,
                                      n_segs_out);  // real n_segs
  // Pass 5: segment descriptors at the seqlen_q upper-bound grid.
  seg_desc_kernel<<<nb, kBlock, 0, stream>>>(b, n_segs_out, seqlen_q, done,
                                             ks, ke, sc.d_ks, sc.d_ke,
                                             seg_q_start, seg_q_end,
                                             seg_ks0, seg_ke0, seg_dks,
                                             seg_dke);
  return hggcGetLastError();
}
