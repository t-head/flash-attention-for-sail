// Fused mask-decomposition ops: hand-written CUDA replacements for the
// torch.compile'd sweeps in mask_decomp.py (_row_stats_compiled /
// _peel_first_interval_compiled).  Measured 2.66x faster than the
// inductor-fused versions at s=25286 (3.13ms -> 1.18ms, 543 GB/s) and they
// remove the dynamo dependency from the decomposition path entirely — a
// prerequisite for a C++-callable decomposition op (SDPA backend port).
//
// Semantics are 1:1 with mask_decomp.py:
//   row_stats(mask bool [Sq,Sk]) ->
//     ks  int32[Sq] : first True col, 0 if none
//     ke  int32[Sq] : last True col + 1, 0 if none
//     hh  bool [Sq] : any False inside [ks,ke)
//   peel_first_interval(blk bool [Sq,Sk], mutated in place) ->
//     ks/ke int32[Sq] of each row's FIRST contiguous True interval
//     (ks==ke==0 for empty rows); blk is left with that interval cleared.
//
// One block per row; warp + smem reductions.  Row accesses use a scalar
// prologue to the next 16B boundary, a uint4 body, and a scalar tail; that
// exact pattern was probe-validated on PPU for every row offset.  Raw
// misaligned uint4 loads must never be issued: PPU compiles them to
// 16B-aligned vector loads and a misaligned address kills the device
// context ("invalid global mem access alignment").
//
// This TU stays torch-header-free (repo convention: only .cpp translation
// units may include torch headers); the op wrappers and TORCH_LIBRARY
// registration live in flex_flash_attention_api.cpp and call the launchers below.

#include <hggc_runtime.h>
#include <cstdint>

namespace {

constexpr int kBlock = 256;
constexpr int kWarp = 32;

// ── row_stats ────────────────────────────────────────────────────────────
__global__ void row_stats_kernel(unsigned char const* __restrict__ m,
                                 int const seqlen_q, int const seqlen_k,
                                 int* __restrict__ ks_out,
                                 int* __restrict__ ke_out,
                                 unsigned char* __restrict__ hh_out) {
  int const row = blockIdx.x;
  if (row >= seqlen_q) return;
  unsigned char const* r = m + (size_t)row * seqlen_k;

  int first = seqlen_k, last = -1, cnt = 0;

  // Scalar prologue until the row cursor is 16B-aligned.  Clamped to
  // seqlen_k: for short rows the next boundary can lie PAST the row end,
  // and an unclamped pre would drive vec_end negative so the scalar tail
  // loop started at a negative index (OOB read, garbage stats).
  int const pre_raw = (int)((16 - ((uintptr_t)r & 15)) & 15);
  int const pre = pre_raw < seqlen_k ? pre_raw : seqlen_k;
  int const vec_end = seqlen_k - ((seqlen_k - pre) & 15);
  for (int i = threadIdx.x; i < pre && i < seqlen_k; i += kBlock) {
    if (r[i]) { if (i < first) first = i; if (i > last) last = i; ++cnt; }
  }
  // Body: 16 cols per uint4 load per iteration.
  uint4 const* rv = reinterpret_cast<uint4 const*>(r + pre);
  int const n_vec = (vec_end - pre) >> 4;
  for (int i = threadIdx.x; i < n_vec; i += kBlock) {
    uint4 const v = rv[i];
    unsigned char const* bytes = reinterpret_cast<unsigned char const*>(&v);
    int const base = pre + (i << 4);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (bytes[j]) {
        int const col = base + j;
        if (col < first) first = col;
        if (col > last) last = col;
        ++cnt;
      }
    }
  }
  // Scalar tail.
  for (int i = vec_end + threadIdx.x; i < seqlen_k; i += kBlock) {
    if (r[i]) { if (i < first) first = i; if (i > last) last = i; ++cnt; }
  }

  // Block reduction: min(first), max(last), sum(cnt).
  __shared__ int s_first[kBlock / kWarp], s_last[kBlock / kWarp],
      s_cnt[kBlock / kWarp];
  int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    int const o_first = __shfl_down_sync(0xffffffffu, first, off);
    int const o_last = __shfl_down_sync(0xffffffffu, last, off);
    int const o_cnt = __shfl_down_sync(0xffffffffu, cnt, off);
    if (o_first < first) first = o_first;
    if (o_last > last) last = o_last;
    cnt += o_cnt;
  }
  if (lane == 0) { s_first[warp] = first; s_last[warp] = last; s_cnt[warp] = cnt; }
  __syncthreads();
  if (threadIdx.x == 0) {
    int f = seqlen_k, l = -1, n = 0;
#pragma unroll
    for (int w = 0; w < kBlock / kWarp; ++w) {
      if (s_first[w] < f) f = s_first[w];
      if (s_last[w] > l) l = s_last[w];
      n += s_cnt[w];
    }
    bool const any = (l >= 0);
    int const ks = any ? f : 0;
    int const ke = any ? l + 1 : 0;
    ks_out[row] = ks;
    ke_out[row] = ke;
    hh_out[row] = any && (n != (ke - ks));
  }
}

// ── peel_first_interval ──────────────────────────────────────────────────
// Three sweeps over the row: (1) global first True, (2) first False after
// it, (3) write-back with [ks, ke) cleared.  Kept separate because
// "first False after the GLOBAL first" needs the reduced result before it
// can be tested — a single-pass per-thread scan misses False columns owned
// by threads that never saw a True locally (empirically verified bug).
//
// WITH_FLAGS variant (layered-decompose orchestration): phase 3 also
// reduces two per-row booleans into device-wide flags —
//   nonempty : any row has ks != ke  (== Python (ks_l != ke_l).any())
//   leftover : any row still has a True strictly after its peeled ke
// and sets *done_p when the residual is empty so every later layer's
// launch short-circuits on device (no host round trip over 16 layers).
template <bool WITH_FLAGS>
__global__ void peel_kernel(unsigned char* __restrict__ blk,
                            int const seqlen_q, int const seqlen_k,
                            int* __restrict__ ks_out,
                            int* __restrict__ ke_out,
                            int* __restrict__ nonempty_p,
                            int* __restrict__ leftover_p,
                            int const* __restrict__ done_p) {
  int const row = blockIdx.x;
  if (row >= seqlen_q) return;
  if (WITH_FLAGS && *done_p) return;  // residual emptied by an earlier layer
  unsigned char* r = blk + (size_t)row * seqlen_k;

  // Prologue/uint4/tail split; prologue makes every uint4 load 16B-aligned
  // (misaligned vector loads kill the PPU context; see file header note).
  // pre is clamped to seqlen_k (same guard as row_stats_kernel: without it
  // short rows produce a negative vec_end and an OOB scalar tail).
  int const pre_raw = (int)((16 - ((uintptr_t)r & 15)) & 15);
  int const pre = pre_raw < seqlen_k ? pre_raw : seqlen_k;
  int const vec_end = seqlen_k - ((seqlen_k - pre) & 15);
  int const n_vec = (vec_end - pre) >> 4;
  uint4 const* rv = reinterpret_cast<uint4 const*>(r + pre);

  // Phase 1: global first True.
  int first = seqlen_k;
  for (int i = threadIdx.x; i < pre && i < seqlen_k; i += kBlock)
    if (r[i] && i < first) first = i;
  for (int vi = threadIdx.x; vi < n_vec; vi += kBlock) {
    uint4 const v = rv[vi];
    unsigned char const* bytes = reinterpret_cast<unsigned char const*>(&v);
    int const base = pre + (vi << 4);
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if (bytes[j]) { int const col = base + j; if (col < first) first = col; }
  }
  for (int i = vec_end + threadIdx.x; i < seqlen_k; i += kBlock)
    if (r[i] && i < first) first = i;

  __shared__ int s_first_w[kBlock / kWarp];
  int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    int const o = __shfl_down_sync(0xffffffffu, first, off);
    if (o < first) first = o;
  }
  if (lane == 0) s_first_w[warp] = first;
  __syncthreads();
  __shared__ int s_ks, s_ke;
  __shared__ bool s_any;
  if (threadIdx.x == 0) {
    int f = seqlen_k;
#pragma unroll
    for (int w = 0; w < kBlock / kWarp; ++w)
      if (s_first_w[w] < f) f = s_first_w[w];
    s_any = (f != seqlen_k);
    s_ks = s_any ? f : 0;
  }
  __syncthreads();
  if (!s_any) {
    // Empty row: uniform early-out (no pending barriers beyond this point).
    if (threadIdx.x == 0) {
      ks_out[row] = 0;
      ke_out[row] = 0;
      // WITH_FLAGS: this row contributes neither nonempty nor leftover;
      // no flag write needed (flag kernels OR-reduce per-row votes).
    }
    return;
  }
  int const ks = s_ks;

  // Phase 2: first False column strictly after the global first.
  int ffa = seqlen_k;
  for (int i = threadIdx.x; i < pre && i < seqlen_k; i += kBlock)
    if (i > ks && !r[i] && i < ffa) ffa = i;
  for (int vi = threadIdx.x; vi < n_vec; vi += kBlock) {
    int const base = pre + (vi << 4);
    if (base + 16 <= ks) continue;  // entirely at/before ks
    uint4 const v = rv[vi];
    unsigned char const* bytes = reinterpret_cast<unsigned char const*>(&v);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      int const col = base + j;
      if (col > ks && !bytes[j] && col < ffa) ffa = col;
    }
  }
  for (int i = vec_end + threadIdx.x; i < seqlen_k; i += kBlock)
    if (i > ks && !r[i] && i < ffa) ffa = i;

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    int const o = __shfl_down_sync(0xffffffffu, ffa, off);
    if (o < ffa) ffa = o;
  }
  if (lane == 0) s_first_w[warp] = ffa;
  __syncthreads();
  if (threadIdx.x == 0) {
    int g = seqlen_k;
#pragma unroll
    for (int w = 0; w < kBlock / kWarp; ++w)
      if (s_first_w[w] < g) g = s_first_w[w];
    s_ke = g;
    ks_out[row] = ks;
    ke_out[row] = g;
  }
  __syncthreads();
  int const ke = s_ke;

  // Phase 3: write-back: clear [ks, ke).  peeled = blk & !(ks<=col<ke).
  // WITH_FLAGS: the same sweep votes "leftover" when any True strictly
  // after ke survives (the row still needs another layer).
  int left = 0;
  if (ke > ks) {
    for (int i = threadIdx.x; i < pre && i < seqlen_k; i += kBlock)
      if (i >= ks && i < ke) r[i] = 0;
    for (int vi = threadIdx.x; vi < n_vec; vi += kBlock) {
      int const base = pre + (vi << 4);
      if (base + 16 <= ks || base >= ke) continue;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        int const col = base + j;
        if (col >= ks && col < ke) r[col] = 0;
      }
    }
    for (int i = vec_end + threadIdx.x; i < seqlen_k; i += kBlock)
      if (i >= ks && i < ke) r[i] = 0;
  }
  if (WITH_FLAGS) {
    // Leftover detection: True columns strictly after ke.
    if (ke < seqlen_k) {
      for (int i = vec_end + threadIdx.x; i < seqlen_k; i += kBlock)
        if (i > ke && r[i]) left = 1;
      for (int vi = threadIdx.x; vi < n_vec; vi += kBlock) {
        int const base = pre + (vi << 4);
        if (base + 16 <= ke) continue;  // entirely at/before ke
        uint4 const v = rv[vi];
        unsigned char const* bytes =
            reinterpret_cast<unsigned char const*>(&v);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          int const col = base + j;
          if (col > ke && bytes[j]) left = 1;
        }
      }
      for (int i = threadIdx.x; i < pre && i < seqlen_k; i += kBlock)
        if (i > ke && r[i]) left = 1;
    }
    // OR-reduce `left` across the block, then vote into the global flags.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      left |= __shfl_xor_sync(0xffffffffu, left, off);
    if (lane == 0) s_first_w[warp] = left;
    __syncthreads();
    if (threadIdx.x == 0) {
      int lo = 0;
#pragma unroll
      for (int w = 0; w < kBlock / kWarp; ++w) lo |= s_first_w[w];
      atomicOr(nonempty_p, 1);          // this row has ks != ke
      if (lo) atomicOr(leftover_p, 1);
    }
  }
}

// Grid-wide gate after a peel layer: the residual is empty iff NO row
// voted nonempty; flip done so every later layer's launch short-circuits.
// Also resets the per-layer votes: peel only atomicOr's into them, so
// without a reset layer 0's "nonempty" would stick and the gate would
// never fire on a later empty peel.  A gated peel early-outs without
// voting, so the reset keeps the flag at 0 and the gate propagates.
__global__ void decomp_gate_done_kernel(int* __restrict__ nonempty_p,
                                        int* __restrict__ leftover_p,
                                        int* __restrict__ done_p) {
  if (!*nonempty_p) *done_p = 1;
  *nonempty_p = 0;
  *leftover_p = 0;
}

// ── layered-decompose finalizer ──────────────────────────────────────────
// Layer-cap verdict, mirroring _decompose_with_holes exactly: the Python
// loop raises whenever ALL 16 peels returned nonempty — even when the
// residual happened to empty on the 16th peel (it never peels a 17th time
// to check).  gate_cap (= gate[16]) is set iff some peel inside the cap
// returned an empty residual, hence supported ⇔ gate_cap.  Also publishes
// the working supported flag into counters[2], the desc pack's slot (the
// flag itself lives in the scratch flags tensor).
__global__ void decomp_finalize_kernel(int const* __restrict__ gate_cap_p,
                                       int* __restrict__ supported_p,
                                       int* __restrict__ counters_sup_p,
                                       int* __restrict__ n_layers_p,
                                       int const n_layers) {
  if (!*gate_cap_p) *supported_p = 0;
  *counters_sup_p = *supported_p;
  *n_layers_p = n_layers;
}

// Layout guard: when the desc is unsupported (layer cap / slice cap), the
// accumulated slice counter may hold garbage — zero it so the layout
// stage emits an empty, well-defined desc instead of reading junk.
__global__ void decomp_guard_zero_kernel(int const* __restrict__ supported_p,
                                         int* __restrict__ n_pre_p) {
  if (!*supported_p) *n_pre_p = 0;
}

}  // namespace

// ── launchers (called from flex_flash_attention_api.cpp) ─────────────────────
void launch_decomp_row_stats(unsigned char const* mask, int seqlen_q,
                             int seqlen_k, int* ks, int* ke,
                             unsigned char* hh, hggcStream_t stream) {
  if (seqlen_q > 0 && seqlen_k > 0) {
    row_stats_kernel<<<seqlen_q, kBlock, 0, stream>>>(mask, seqlen_q,
                                                      seqlen_k, ks, ke, hh);
  }
}

void launch_decomp_peel_first_interval(unsigned char* blk, int seqlen_q,
                                       int seqlen_k, int* ks, int* ke,
                                       hggcStream_t stream) {
  if (seqlen_q > 0 && seqlen_k > 0) {
    peel_kernel<false><<<seqlen_q, kBlock, 0, stream>>>(
        blk, seqlen_q, seqlen_k, ks, ke, nullptr, nullptr, nullptr);
  }
}

void launch_decomp_peel_flags(unsigned char* blk, int seqlen_q,
                              int seqlen_k, int* ks, int* ke,
                              int* nonempty_p, int* leftover_p,
                              int const* done_p, hggcStream_t stream) {
  if (seqlen_q > 0 && seqlen_k > 0) {
    peel_kernel<true><<<seqlen_q, kBlock, 0, stream>>>(
        blk, seqlen_q, seqlen_k, ks, ke, nonempty_p, leftover_p, done_p);
  }
}

void launch_decomp_gate_done(int* nonempty_p, int* leftover_p,
                             int* done_p, hggcStream_t stream) {
  decomp_gate_done_kernel<<<1, 1, 0, stream>>>(nonempty_p, leftover_p,
                                               done_p);
}

void launch_decomp_finalize(int const* gate_cap_p, int* supported_p,
                            int* counters_sup_p, int* n_layers_p,
                            int n_layers, hggcStream_t stream) {
  decomp_finalize_kernel<<<1, 1, 0, stream>>>(gate_cap_p, supported_p,
                                              counters_sup_p, n_layers_p,
                                              n_layers);
}

void launch_decomp_guard_zero(int const* supported_p, int* n_pre_p,
                              hggcStream_t stream) {
  decomp_guard_zero_kernel<<<1, 1, 0, stream>>>(supported_p, n_pre_p);
}
