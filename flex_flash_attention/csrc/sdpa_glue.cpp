// Library-side implementation of the SDPA contract declared in
// include/flex_flash_attention_sdpa.h.
//
// The pytorch fork links this TU (as part of libflex_flash_attention.so) and
// calls flex_flash_attention::sdpa_fwd/bwd from the CUDA dispatch of the aten
// ops `_scaled_dot_product_flex_flash_attention(+_backward)`; autograd
// is wired by tools/autograd/derivatives.yaml in the torch fork.
//
// Orchestration mirrors interface.py::FlexFlashAttentionSDPABackend:
//   * SDPA arrives BHSD; kernels consume BSHD views (last-dim-contiguous
//     is all they need — no copies).
//   * attn_mask None / is_causal synthesize a dense / causal mask on
//     device and run the SAME decomposition path.
//   * masks outside the layered-interval envelope are rejected BEFORE
//     backend selection by sdpa_mask_decomposable (called from
//     can_use_flex_flash_attention), so SDPA dispatch routes them
//     to another backend; the checks in sdpa_fwd/bwd are defense only.
//   * bwd re-decomposes on the 768-row bwd tile grid, same as the Python
//     autograd.Function it replaces.
//
// The kernel entry points (flex_flash_attention::fwd/bwd/decompose_mask) are
// registered by flex_flash_attention_api.cpp in this same shared library, so
// the dispatcher lookups below always resolve once the so is loaded.
#include "flex_flash_attention_sdpa.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/EmptyTensor.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/zeros.h>

#include <cstdlib>
#include <mutex>
#include <tuple>
#include <unordered_map>

namespace flex_flash_attention {
namespace {

constexpr int64_t kFwdKblockM = 128;   // fwd Q-tile (interface.py FWD_KBLOCK_M)
constexpr int64_t kBwdKblockM = 768;   // bwd Q-tile (interface.py _BWD_KBLOCK_M)

// Must match the registered functor signatures (mha_flex_flash_fwd/bwd in
// flex_flash_attention_api.cpp) EXACTLY — by value, not const&: typed<>() throws
// "wrong signature" otherwise.
using FlexFlashAttentionFwd = std::tuple<at::Tensor, at::Tensor, at::Tensor>(
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<double>, std::optional<double>, bool, bool,
    std::optional<bool>, bool, double,
    std::optional<at::Generator>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, int64_t, int64_t);

using FlexFlashAttentionBwd = std::tuple<at::Tensor, at::Tensor, at::Tensor>(
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<double>, bool, std::optional<double>,
    bool, bool, bool, bool, double,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>);

const c10::TypedOperatorHandle<FlexFlashAttentionFwd>* flex_flash_attention_fwd() {
  static const auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("flex_flash_attention::fwd", "")
          .typed<FlexFlashAttentionFwd>();
  return &op;
}

const c10::TypedOperatorHandle<FlexFlashAttentionBwd>* flex_flash_attention_bwd() {
  static const auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("flex_flash_attention::bwd", "")
          .typed<FlexFlashAttentionBwd>();
  return &op;
}

// ISOLATED fp32 kernel set: fwd_f32/bwd_f32 have schemas identical to
// fwd/bwd, so the typed signatures above are reused verbatim.  Only the
// float32 branch of sdpa_fwd/bwd touches these handles — the fp16/bf16
// dispatch stays exactly as before.
const c10::TypedOperatorHandle<FlexFlashAttentionFwd>* flex_flash_attention_fwd_f32() {
  static const auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("flex_flash_attention::fwd_f32", "")
          .typed<FlexFlashAttentionFwd>();
  return &op;
}

const c10::TypedOperatorHandle<FlexFlashAttentionBwd>* flex_flash_attention_bwd_f32() {
  static const auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("flex_flash_attention::bwd_f32", "")
          .typed<FlexFlashAttentionBwd>();
  return &op;
}

using DecomposeMask = std::tuple<at::Tensor, at::Tensor, at::Tensor,
                                 at::Tensor>(const at::Tensor&, int64_t);

const c10::TypedOperatorHandle<DecomposeMask>* flex_flash_attention_decompose_mask() {
  static const auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("flex_flash_attention::decompose_mask", "")
          .typed<DecomposeMask>();
  return &op;
}

// Cache-bypassing variant recorded into CUDA graphs (see graph mode in
// sdpa_fwd/bwd): replays must re-run the pipeline, never read a cache.
const c10::TypedOperatorHandle<DecomposeMask>*
flex_flash_attention_decompose_mask_nocache() {
  static const auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("flex_flash_attention::decompose_mask_nocache", "")
          .typed<DecomposeMask>();
  return &op;
}

// SDPA accepts (sq, sk) or broadcastable 4-D masks; the decomposition only
// understands a single (sq, sk) pattern, and
// can_use_flex_flash_attention already restricted the 4-D form to
// singleton batch/head dims.
at::Tensor mask_to_2d(const at::Tensor& mask) {
  if (mask.dim() == 2) {
    return mask;
  }
  TORCH_CHECK(mask.dim() == 4 && mask.size(0) == 1 && mask.size(1) == 1,
              "flex_flash_attention::sdpa: attn_mask must be (seqlen_q, "
              "seqlen_k) or (1, 1, seqlen_q, seqlen_k)");
  return mask.select(0, 0).select(0, 0);
}

// Decompose results are cached inside flex_flash_attention::decompose_mask,
// keyed by (TensorImpl*, version, kblock_m).  When resolve_mask materializes
// a fresh (sq, sk) copy (4-D select / is_causal / None), that identity is
// per-call and would miss the cache between the selection probe and the
// actual fwd/bwd run — so cache the resolved 2-D mask on the ORIGINAL mask
// identity (or an is_causal tag) and hand the same tensor to every stage.
// Entries store the original mask, which pins its TensorImpl (same ABA
// defense as the library's decomp cache); the data pointer in the key
// additionally narrows hits across the clear-on-overflow eviction window.
// Inference tensors carry NO version counter (reading it raises), and they
// can never be mutated in place (in-place ops reject them), so keying on a
// constant version is sound and keeps inference_mode workloads usable.
static int64_t safe_version(at::TensorImpl* impl) {
  return impl->is_inference()
             ? 0
             : static_cast<int64_t>(impl->version_counter().current_version());
}

struct ResolvedMaskKey {
  at::TensorImpl* impl;  // nullptr for the is_causal synthesis path
  int64_t version;
  const void* data;  // storage address: narrows TensorImpl-recycle (ABA) hits
  int64_t seqlen_q;
  int64_t seqlen_k;
  bool is_causal;
  bool operator==(const ResolvedMaskKey& o) const {
    return impl == o.impl && version == o.version && data == o.data &&
        seqlen_q == o.seqlen_q &&
        seqlen_k == o.seqlen_k && is_causal == o.is_causal;
  }
};
struct ResolvedMaskKeyHash {
  size_t operator()(const ResolvedMaskKey& k) const {
    return std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(k.impl)) ^
        std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(k.data)) ^
        (std::hash<int64_t>{}(k.seqlen_q) << 1) ^
        (std::hash<int64_t>{}(k.seqlen_k) << 2) ^
        (k.is_causal ? size_t{1} << 3 : 0);
  }
};
std::mutex g_resolved_mtx;
// Heap-allocated, never destroyed: entries hold CUDA tensors, and a static
// destructor running after CUDA context teardown crashes while freeing
// them (static-destruction-order fiasco).
std::unordered_map<ResolvedMaskKey, at::Tensor, ResolvedMaskKeyHash>&
    g_resolved_cache =
        *new std::unordered_map<ResolvedMaskKey, at::Tensor,
                                ResolvedMaskKeyHash>();
constexpr size_t kResolvedCacheMax = 16;

// ── Specialized fast-path desc cache (FA2-grade host overhead) ────────────
// The decomposition op returns GPU counters, so the legacy path paid three
// device->host syncs (.item()) on EVERY fwd/bwd call even when the result
// was an internal cache hit.  This cache stores the trimmed desc pack plus
// the HOST-side counters, keyed by resolved-mask identity + Q-tile grid, so
// repeat calls pay zero syncs and zero kernel launches for the mask path.
// None-mask / is_causal entries are SYNTHETIC (no mask tensor is ever
// materialized or decomposed): one FULL / CAUSAL slice covering the whole
// (sq, sk) rectangle — valid on any Q-tile grid since the slice Q range
// spans all rows.
struct DecompKey {
  ResolvedMaskKey mask;
  int64_t kblock_m;
  bool operator==(const DecompKey& o) const {
    return mask == o.mask && kblock_m == o.kblock_m;
  }
};
struct DecompKeyHash {
  size_t operator()(const DecompKey& k) const {
    return ResolvedMaskKeyHash{}(k.mask) ^
        (std::hash<int64_t>{}(k.kblock_m) << 4);
  }
};
struct DecompEntry {
  // Trimmed slice arrays (views into the cached desc pack).
  at::Tensor q_starts, q_ends, k_starts, k_ends, mask_types;
  at::Tensor diagonal_offsets, band_widths;
  at::Tensor row_to_slice, row_to_vb_start, row_to_vb_end, vbatch;
  int64_t n = 0;
  int64_t nvb = 0;
  bool supported = false;
  // True when any slice is diagonal (CAUSAL/INVCAUSAL/BICAUSAL): those yield
  // the triangular per-Q-tile workload that the static round-robin scheduler
  // load-balances poorly on low-SM parts (see tile_scheduler_flex_flash.h).
  bool has_diagonal = false;
  // Trivial single-slice specialization (0 = off, 1 = FULL, 2 = CAUSAL):
  // host-proven one-slice-over-the-rectangle mask, forwarded to the fwd op
  // so the kernel skips all descriptor global loads (trivial_enabled()).
  int64_t trivial_mask = 0;
  int64_t trivial_diagonal = 0;
};
std::mutex g_decomp_mtx;
std::unordered_map<DecompKey, DecompEntry, DecompKeyHash>&
    g_decomp_cache =
        *new std::unordered_map<DecompKey, DecompEntry, DecompKeyHash>();
constexpr size_t kDecompCacheMax = 16;

// Dynamic persistent tile-scheduler gate (work-stealing via a global
// semaphore; cures the causal tail-wave imbalance, see
// tile_scheduler_flex_flash.h).  FLASH_ATTENTION_ARB_DYN_SCHED: "1" force on,
// "0" force off, unset = auto (on iff the slice set has a diagonal slice).
// Both scheduler variants are compiled into every fwd/bwd instantiation
// (BOOL_SWITCH on params.persistent_scheduler), so this is a host-side-only
// choice with no kernel rebuild.
bool use_dyn_sched(bool has_diagonal) {
  static int const env = [] {
    char const* e = getenv("FLASH_ATTENTION_ARB_DYN_SCHED");
    if (e != nullptr && e[0] == '1') return 1;
    if (e != nullptr && e[0] == '0') return 0;
    return -1;  // auto
  }();
  if (env >= 0) return env == 1;
  return has_diagonal;
}

// Trivial single-slice fast path gate: when the mask provably is ONE FULL /
// CAUSAL slice over the whole rectangle (every None-mask / is_causal call,
// plus explicit masks whose decomposition comes out that way), the fwd
// kernel runs with host-constant slice metadata and touches no descriptor
// arrays at all (see AttnSliceParams::trivial_mask / TrivialSliceMeta).
// FLASH_ATTENTION_ARB_NO_TRIVIAL=1 forces the generic slice mechanism —
// same binary, for A/B benchmarking.
bool trivial_enabled() {
  static bool const env = [] {
    char const* e = getenv("FLASH_ATTENTION_ARB_NO_TRIVIAL");
    return !(e != nullptr && e[0] == '1');
  }();
  return env;
}

// Synthetic desc pack: one FULL (dense) or CAUSAL slice over the whole
// matrix.  Counters are host constants, so consumers never sync.
DecompEntry synth_desc(bool causal, int64_t seqlen_q, int64_t seqlen_k,
                       const at::Tensor& like) {
  auto i32 = like.options().dtype(at::kInt);
  DecompEntry e;
  e.n = 1;
  e.nvb = 1;
  e.supported = true;
  e.has_diagonal = causal;
  e.q_starts = at::zeros({1}, i32);
  e.q_ends = at::full({1}, seqlen_q, i32);
  e.k_starts = at::zeros({1}, i32);
  e.k_ends = at::full({1}, seqlen_k, i32);
  e.mask_types = at::full({1}, causal ? int64_t{1} : int64_t{0}, i32);
  e.diagonal_offsets = at::zeros({1}, i32);
  e.band_widths = at::zeros({1}, i32);
  e.row_to_slice = at::zeros({seqlen_q}, i32);
  e.row_to_vb_start = at::zeros({seqlen_q}, i32);
  e.row_to_vb_end = at::ones({seqlen_q}, i32);
  e.vbatch = at::zeros({1}, i32);
  // Synthetic descs ARE the trivial single-slice case by construction: the
  // kernel receives host-constant slice metadata and never reads the (all
  // constant anyway) descriptor tensors.  Synthetic causal is tril(diag=0),
  // so the trivial diagonal is 0.
  if (trivial_enabled()) {
    e.trivial_mask = causal ? 2 : 1;
    e.trivial_diagonal = 0;
  }
  return e;
}

// Decompose with host-side counter caching: the first call per (mask,
// kblock_m) pays the op + the counter readbacks; every later call is pure
// host-side bookkeeping (cache lookup, no syncs, no launches).
DecompEntry decomp_cached(const at::Tensor& mask, int64_t kblock_m) {
  DecompKey key{{mask.unsafeGetTensorImpl(), safe_version(
                    mask.unsafeGetTensorImpl()),
                 mask.const_data_ptr(), mask.size(-2), mask.size(-1),
                 /*is_causal=*/false},
                kblock_m};
  {
    std::lock_guard<std::mutex> lk(g_decomp_mtx);
    auto it = g_decomp_cache.find(key);
    if (it != g_decomp_cache.end()) return it->second;
  }
  auto [slices, rows, vbatch, counters] =
      flex_flash_attention_decompose_mask()->call(mask, kblock_m);
  DecompEntry e;
  e.n = counters[0].item<int64_t>();
  e.nvb = counters[1].item<int64_t>();
  e.supported = counters[2].item<bool>();
  if (e.supported) {
    e.q_starts = slices[0].narrow(0, 0, e.n);
    e.q_ends = slices[1].narrow(0, 0, e.n);
    e.k_starts = slices[2].narrow(0, 0, e.n);
    e.k_ends = slices[3].narrow(0, 0, e.n);
    e.mask_types = slices[4].narrow(0, 0, e.n);
    e.diagonal_offsets = slices[5].narrow(0, 0, e.n);
    e.band_widths = slices[6].narrow(0, 0, e.n);
    e.row_to_slice = rows[0];
    e.row_to_vb_start = rows[1];
    e.row_to_vb_end = rows[2];
    e.vbatch = vbatch.narrow(0, 0, e.nvb);
    // Diagonal-slice probe for the scheduler gate: small one-shot DtoH on
    // cache miss only (the miss already pays the counter readbacks).
    // SLICE_CAUSAL=1 / SLICE_INVCAUSAL=2 / SLICE_BICAUSAL=3.
    auto mt = e.mask_types.cpu();
    const int* mt_p = mt.const_data_ptr<int>();
    for (int64_t i = 0; i < e.n; ++i) {
      if (mt_p[i] >= 1 && mt_p[i] <= 3) {
        e.has_diagonal = true;
        break;
      }
    }
    // Trivial single-slice probe on the SAME miss window: an explicit mask
    // that decomposes to exactly one FULL / CAUSAL slice covering the whole
    // rectangle rides the same kernel fast path as None-mask / is_causal.
    // One packed DtoH of the five scalars; only ever paid once per
    // (mask, kblock_m) identity.
    //
    // bwd note: the decomposer splits explicit masks by the Q-tile grid
    // (kblock_m), so a dense/causal mask usually arrives as n>1 SAME-TYPE
    // contiguous segments (e.g. 2048 rows @ kblock 768 -> 3 FULL segments).
    // The kernel's trivial path derives slice geometry purely from
    // (trivial_mask, trivial_diagonal) and never indexes the slice arrays,
    // so such uniform coverings also ride the fast path — the segment
    // tensors stay as passed and are simply not loaded.
    if (trivial_enabled() && e.n >= 1 && mt_p[0] <= 1) {
      // Uniform-type check across ALL segments (first/last alone would miss
      // a type change in the middle).
      const bool uniform_type =
          e.mask_types.eq(int64_t(mt_p[0])).all().item<bool>();
      if (uniform_type) {
        const int64_t sq = mask.size(-2), sk = mask.size(-1);
        auto ends = at::cat({e.q_ends.narrow(0, e.n - 1, 1),
                             e.k_starts.narrow(0, 0, 1),
                             e.k_ends.narrow(0, e.n - 1, 1)}).cpu();
        const int* ep = ends.const_data_ptr<int>();
        // Run-length segmentation covers rows contiguously, so q coverage is
        // proven by first q_start==0 and last q_end==sq; the per-segment k
        // range must agree with ONE global FULL / CAUSAL slice.
        const bool covers = ep[0] == sq && ep[1] == 0 && ep[2] == sk;
        // SOUNDNESS: "covers" only checks the FIRST k_start / LAST k_end; the
        // stair mask decomposes to same-type segments with varying k ranges
        // (e.g. ks=[0,256,256] ke=[512,1024,1024]) yet passes it.  The exact
        // check is geometric: a segment's k range is the column UNION over
        // its rows, so one global slice implies a closed form per segment.
        if (covers) {
          if (mt_p[0] == 0) {
            // FULL: every segment must span all columns (any hole layer
            // shrinks some segment's range and fails here).
            const bool kspan =
                e.k_starts.eq(0).logical_and(e.k_ends.eq(sk))
                    .all().item<bool>();
            if (kspan) {
              e.trivial_mask = 1;
              e.trivial_diagonal = 0;
            }
          } else {
            // CAUSAL(d): row q sees k in [0, q+d), so segment [a, b) spans
            // [0, min(sk, b+d)); require ONE uniform diagonal and that exact
            // range for EVERY segment (bands/holes/stairs break it).
            auto d0 = e.diagonal_offsets.narrow(0, 0, 1);
            const bool uniform =
                e.diagonal_offsets.narrow(0, 0, e.n).eq(d0).all().item<bool>();
            if (uniform) {
              const int d = d0.cpu().item<int>();
              auto expect_ke = e.q_ends.narrow(0, 0, e.n)
                                   .add(int64_t(d)).clamp(int64_t(0), sk);
              const bool kgeom =
                  e.k_starts.narrow(0, 0, e.n).eq(0)
                      .logical_and(e.k_ends.narrow(0, 0, e.n).eq(expect_ke))
                      .all().item<bool>();
              if (kgeom) {
                e.trivial_mask = 2;
                e.trivial_diagonal = d;
              }
            }
          }
        }
      }
    }
  }
  std::lock_guard<std::mutex> lk(g_decomp_mtx);
  if (g_decomp_cache.size() >= kDecompCacheMax) g_decomp_cache.clear();
  g_decomp_cache.emplace(key, e);
  return e;
}

// ── CUDA graph support ──────────────────────────────────────────────────
// Graph replay executes NO host code, so (a) routing verdicts needed while
// capturing must come from a host-side cache populated by earlier eager
// probes (any device readback would break stream capture), and (b) mask
// content changes between replays can only be tracked by re-running the
// decomposition INSIDE the graph (graph mode in sdpa_fwd/bwd below).

// Host-side verdict cache: eager sdpa_mask_decomposable probes record their
// per-(mask identity, kblock_m) supported flags here.  Same identity tuple
// as DecompKey, so warmup and capture agree on what "the same mask" means.
struct VerdictKey {
  at::TensorImpl* impl;
  int64_t version;
  const void* data;
  int64_t sq;
  int64_t sk;
  int64_t kblock_m;
  bool operator==(const VerdictKey& o) const {
    return impl == o.impl && version == o.version && data == o.data &&
           sq == o.sq && sk == o.sk && kblock_m == o.kblock_m;
  }
};
struct VerdictKeyHash {
  size_t operator()(const VerdictKey& k) const {
    auto h = std::hash<const void*>{}(k.impl);
    h ^= std::hash<int64_t>{}(k.version) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.kblock_m) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};
std::unordered_map<VerdictKey, bool, VerdictKeyHash> g_verdict_cache;
std::mutex g_verdict_mtx;
constexpr size_t kVerdictCacheMax = 4096;

void record_verdict(const at::Tensor& resolved, int64_t kblock_m,
                    bool supported) {
  VerdictKey key{resolved.unsafeGetTensorImpl(),
                 safe_version(resolved.unsafeGetTensorImpl()),
                 resolved.const_data_ptr(), resolved.size(-2),
                 resolved.size(-1), kblock_m};
  std::lock_guard<std::mutex> lk(g_verdict_mtx);
  if (g_verdict_cache.size() >= kVerdictCacheMax) g_verdict_cache.clear();
  g_verdict_cache[key] = supported;
}

std::optional<bool> lookup_verdict(const at::Tensor& resolved,
                                   int64_t kblock_m) {
  VerdictKey key{resolved.unsafeGetTensorImpl(),
                 safe_version(resolved.unsafeGetTensorImpl()),
                 resolved.const_data_ptr(), resolved.size(-2),
                 resolved.size(-1), kblock_m};
  std::lock_guard<std::mutex> lk(g_verdict_mtx);
  auto it = g_verdict_cache.find(key);
  if (it == g_verdict_cache.end()) return std::nullopt;
  return it->second;
}

// Pinned envelope flag: the graph-mode decomposition publishes its
// device-side `supported` verdict here via a captured D2H memcpy node, so
// every replay refreshes it without running host code.  PPU device-side
// faults (assert / illegal access) are silently swallowed inside graph
// replays (probe-verified), so this host-readable flag is the ONLY way a
// mid-replay envelope violation can surface.  Initialized to supported so a
// check between capture and first replay does not false-alarm.
//
// Leaked-singleton on purpose: this pinned tensor is first allocated DURING
// stream capture, so its storage is recorded on the capture-private stream
// (which is destroyed right after capture ends).  A plain global would be
// destroyed in the static-destruction phase, where the allocator walks that
// dead stream -> "Unrecognized stream" abort at exit.
at::Tensor* g_graph_flag = nullptr;  // leaked on purpose, see graph_flag()
at::Tensor& graph_flag() {
  if (!g_graph_flag) {
    g_graph_flag =
        new at::Tensor(at::full({1}, 1, at::TensorOptions()
                                            .dtype(at::kInt)
                                            .pinned_memory(true)));
  }
  return *g_graph_flag;
}

// Graph-mode decomposition: run the cache-bypassing pipeline on the
// capture stream so it is recorded into the graph, then publish the
// supported verdict to the pinned flag with a capturable async D2H copy.
// Returns the FULL-capacity desc pack; the zero tail is inert by the
// compile-friendly contract of decompose_mask_impl.
struct GraphDecomp {
  at::Tensor slices;  // [7, cap]
  at::Tensor rows;    // [3, sq]
  at::Tensor vbatch;  // [cap]
};
GraphDecomp graph_mode_decompose(const at::Tensor& mask, int64_t kblock_m) {
  auto [slices, rows, vbatch, counters] =
      flex_flash_attention_decompose_mask_nocache()->call(mask, kblock_m);
  graph_flag().copy_(counters.narrow(0, 2, 1), /*non_blocking=*/true);
  return {slices, rows, vbatch};
}

// None mask -> dense (full attention); is_causal -> causal — both become
// explicit masks so a single decomposition path serves every shape.
at::Tensor resolve_mask(const std::optional<at::Tensor>& attn_mask,
                        bool is_causal, int64_t seqlen_q, int64_t seqlen_k,
                        const at::Tensor& like);

at::Tensor resolve_mask_cached(
    const std::optional<at::Tensor>& attn_mask_in, bool is_causal,
    int64_t seqlen_q, int64_t seqlen_k, const at::Tensor& like) {
  // Autograd may save a None mask as an UNDEFINED tensor (optional holding
  // a tensor without storage); treat it exactly like nullopt or downstream
  // data_ptr access crashes.
  std::optional<at::Tensor> attn_mask = attn_mask_in;
  if (attn_mask.has_value() && !attn_mask->defined()) {
    attn_mask = std::nullopt;
  }
  ResolvedMaskKey key{attn_mask.has_value()
                          ? attn_mask->unsafeGetTensorImpl()
                          : nullptr,
                      attn_mask.has_value()
                          ? safe_version(attn_mask->unsafeGetTensorImpl())
                          : int64_t{0},
                      attn_mask.has_value() ? attn_mask->const_data_ptr()
                                            : nullptr,
                      seqlen_q, seqlen_k, is_causal};
  {
    std::lock_guard<std::mutex> lk(g_resolved_mtx);
    auto it = g_resolved_cache.find(key);
    if (it != g_resolved_cache.end()) return it->second;
  }
  auto m = resolve_mask(attn_mask, is_causal, seqlen_q, seqlen_k, like);
  std::lock_guard<std::mutex> lk(g_resolved_mtx);
  if (g_resolved_cache.size() >= kResolvedCacheMax) g_resolved_cache.clear();
  g_resolved_cache.emplace(key, m);
  return m;
}

// None mask -> dense (full attention); is_causal -> causal — both become
// explicit masks so a single decomposition path serves every shape.
at::Tensor resolve_mask(const std::optional<at::Tensor>& attn_mask,
                        bool is_causal, int64_t seqlen_q, int64_t seqlen_k,
                        const at::Tensor& like) {
  if (attn_mask.has_value()) {
    auto m = mask_to_2d(attn_mask.value());
    TORCH_CHECK(m.scalar_type() == at::kBool,
                "flex_flash_attention::sdpa: attn_mask must be boolean");
    if (!m.is_cuda()) {
      m = m.to(like.device());
    }
    if (!m.is_contiguous()) {
      m = m.contiguous();
    }
    return m;
  }
  if (is_causal) {
    // PyTorch is_causal semantics: tril(diagonal=0) — top-left aligned
    // causal mask, row i attends to columns j <= i.  Upstream synthesizes
    // exactly this for non-square shapes too (attention.cpp .tril() and
    // the functional.py pseudocode both use diagonal 0), so the arb
    // backend must match the MATH backend bit-for-bit here.
    return at::tril(
        at::ones({seqlen_q, seqlen_k}, like.options().dtype(at::kBool)));
  }
  return at::ones({seqlen_q, seqlen_k}, like.options().dtype(at::kBool));
}

} // namespace

bool sdpa_available() {
  // The kernel ops are registered by this library's own static
  // initializers; probe defensively so a partially-linked so reports
  // unavailable instead of crashing at first use.  The fp32 kernel set
  // (fwd_f32/bwd_f32) is a mandatory part of the library, so its schemas
  // belong in the same availability contract.
  auto& dispatcher = c10::Dispatcher::singleton();
  return dispatcher.findSchema({"flex_flash_attention::fwd", ""}).has_value() &&
         dispatcher.findSchema({"flex_flash_attention::bwd", ""}).has_value() &&
         dispatcher.findSchema({"flex_flash_attention::fwd_f32", ""}).has_value() &&
         dispatcher.findSchema({"flex_flash_attention::bwd_f32", ""}).has_value() &&
         dispatcher.findSchema({"flex_flash_attention::decompose_mask", ""})
             .has_value() &&
         dispatcher.findSchema({"flex_flash_attention::decompose_mask_nocache", ""})
             .has_value();
}

static void envelope_check_folded();  // defined below, after sdpa_graph_mask_check

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_fwd(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, bool is_causal,
    double dropout_p, std::optional<double> scale) {
  envelope_check_folded();
  TORCH_CHECK(!attn_mask.has_value() || !is_causal,
              "flex_flash_attention::sdpa: attn_mask and is_causal are "
              "mutually exclusive");
  TORCH_CHECK(key.size(-3) == value.size(-3) &&
              query.size(-3) % key.size(-3) == 0,
              "flex_flash_attention::sdpa GQA requires k/v head counts equal and "
              "q heads an integer multiple of them");
  const auto seqlen_q = query.size(-2);
  const auto seqlen_k = key.size(-2);

  // Specialized fast path (FA2-grade host overhead + ARB slice kernels):
  //   * None mask / is_causal: synthetic single-slice desc — no mask
  //     materialization, no decomposition, zero device syncs.
  //   * explicit mask: decomp_cached — the counter readbacks happen once per
  //     mask identity; repeat calls are pure host bookkeeping.
  std::optional<at::Tensor> attn_mask_eff = attn_mask;
  if (attn_mask_eff.has_value() && !attn_mask_eff->defined()) {
    attn_mask_eff = std::nullopt;  // autograd may save None as undefined
  }
  // CUDA graph capture mode: the replay path runs no host code, so the
  // decomposition must be RECORDED into the graph and re-executed on every
  // replay — this is what lets replays track in-place mask content
  // changes (mem-efficient-grade semantics).  Everything below is
  // capture-safe: cache-bypassing decomposition kernels, a capturable D2H
  // publish of the supported verdict into the pinned envelope flag, and
  // full-capacity descriptor tensors whose zero tail is inert.  Eager
  // warmup (the can_use probe ran before capture) is the contract that
  // the mask fits the envelope; post-replay enforcement is automatic via
  // envelope_check_folded() at every sdpa_fwd/bwd entry (the optional
  // flex_flash_attention::sdpa_graph_mask_check() gives immediate detection).
  // None-mask / is_causal
  // shapes ride the synthetic descs below — host constants, already
  // capture-safe.
  const bool capturing = at::cuda::currentStreamCaptureStatus() !=
                         at::cuda::CaptureStatus::None;
  if (capturing && attn_mask_eff.has_value()) {
    auto mask = resolve_mask_cached(attn_mask_eff, /*is_causal=*/false,
                                    seqlen_q, seqlen_k, query);
    auto gd = graph_mode_decompose(mask, kFwdKblockM);
    auto q = query.transpose(1, 2);
    auto k = key.transpose(1, 2);
    auto v = value.transpose(1, 2);
    const auto fwd_op = query.scalar_type() == at::kFloat
                            ? flex_flash_attention_fwd_f32()
                            : flex_flash_attention_fwd();
    auto [out, lse, rng] = fwd_op->call(
        q, k, v,
        gd.slices[0], gd.slices[1], gd.slices[2], gd.slices[3],
        gd.slices[4], gd.rows[0], gd.slices[5], gd.slices[6],
        gd.vbatch, gd.rows[1], gd.rows[2],
        std::nullopt,          // out
        scale,
        std::nullopt,          // softcap
        false,                 // inner_min_to_max (fwd: max->min is faster)
        true,                  // persistent_scheduler: content-independent
                               // choice — graph mode cannot know
                               // has_diagonal without a device readback
        std::nullopt,          // use_kblockm128: kernel-side heuristic
        false,                 // kblockn64
        dropout_p,
        std::nullopt,          // gen (kernel uses the default CUDA generator)
        std::nullopt,          // mask_bits
        std::nullopt, std::nullopt, std::nullopt,
        0, 0);                 // trivial path needs host-proven content:
                               // disabled in graph mode
    return {out.transpose(1, 2), lse, rng};
  }
  DecompEntry ent;
  if (!attn_mask_eff.has_value()) {
    DecompKey key{{nullptr, 0, nullptr, seqlen_q, seqlen_k, is_causal},
                  kFwdKblockM};
    {
      std::lock_guard<std::mutex> lk(g_decomp_mtx);
      auto it = g_decomp_cache.find(key);
      if (it != g_decomp_cache.end()) {
        ent = it->second;
      }
    }
    if (ent.n == 0) {
      ent = synth_desc(is_causal, seqlen_q, seqlen_k, query);
      std::lock_guard<std::mutex> lk(g_decomp_mtx);
      if (g_decomp_cache.size() >= kDecompCacheMax) g_decomp_cache.clear();
      g_decomp_cache.emplace(key, ent);
    }
  } else {
    auto mask = resolve_mask_cached(attn_mask_eff, /*is_causal=*/false,
                                    seqlen_q, seqlen_k, query);
    ent = decomp_cached(mask, kFwdKblockM);
  }
  // can_use_flex_flash_attention already probed sdpa_mask_decomposable
  // with this exact mask identity, so an unsupported mask reaching here is
  // defense only.
  TORCH_CHECK(ent.supported,
              "flex_flash_attention::sdpa_fwd: mask is outside the decomposition "
              "envelope (too many hole layers/slices); it should have been "
              "rejected by can_use_flex_flash_attention");

  // BHSD -> BSHD views (last dim stays contiguous; kernels are happy).
  auto q = query.transpose(1, 2);
  auto k = key.transpose(1, 2);
  auto v = value.transpose(1, 2);
  // fp32 rides the isolated fwd_f32 kernel set; fp16/bf16 keep fwd.
  const auto fwd_op = query.scalar_type() == at::kFloat
                          ? flex_flash_attention_fwd_f32()
                          : flex_flash_attention_fwd();
  auto [out, lse, rng] = fwd_op->call(
      q, k, v,
      ent.q_starts, ent.q_ends, ent.k_starts, ent.k_ends, ent.mask_types,
      ent.row_to_slice, ent.diagonal_offsets, ent.band_widths,
      ent.vbatch, ent.row_to_vb_start, ent.row_to_vb_end,
      std::nullopt,          // out
      scale,
      std::nullopt,          // softcap
      false,                 // inner_min_to_max (fwd: max->min is faster)
      use_dyn_sched(ent.has_diagonal),  // persistent_scheduler
      std::nullopt,          // use_kblockm128: kernel-side heuristic
      false,                 // kblockn64
      dropout_p,             // dropout_p
      std::nullopt,          // gen (kernel uses the default CUDA generator)
      std::nullopt,          // mask_bits (desc path has no bitmask slices)
      std::nullopt, std::nullopt, std::nullopt,
      ent.trivial_mask,      // trivial single-slice fast path (0 = generic)
      ent.trivial_diagonal);
  return {out.transpose(1, 2), lse, rng};
}

// Content-level envelope probe used by can_use_flex_flash_attention
// BEFORE backend selection.  Decomposes against both the fwd and the bwd
// Q-tile grids (training needs both to be within the envelope); the
// results are cached by mask identity inside flex_flash_attention::decompose_mask,
// so sdpa_fwd/bwd re-decompose for free.  None mask / is_causal are never
// probed: dense and causal masks always fit.
bool sdpa_mask_decomposable(const at::Tensor& attn_mask) {
  // Resolve through the same identity cache sdpa_fwd/bwd use, so the
  // decompositions below share cache keys with the execution path.
  auto mask = resolve_mask_cached(attn_mask, /*is_causal=*/false,
                                  attn_mask.size(-2), attn_mask.size(-1),
                                  attn_mask);
  auto fwd_counters =
      std::get<3>(flex_flash_attention_decompose_mask()->call(mask, kFwdKblockM));
  const bool fwd_ok = fwd_counters[2].item<bool>();
  // Record host-side verdicts for the capture-time gate: while a CUDA
  // graph is being captured the gate must decide WITHOUT touching the
  // device, using the verdict of an earlier eager warmup probe.
  record_verdict(mask, kFwdKblockM, fwd_ok);
  if (!fwd_ok) {
    return false;
  }
  auto bwd_counters =
      std::get<3>(flex_flash_attention_decompose_mask()->call(mask, kBwdKblockM));
  const bool bwd_ok = bwd_counters[2].item<bool>();
  record_verdict(mask, kBwdKblockM, bwd_ok);
  return bwd_ok;
}

// Capture-time envelope verdict: pure host-side lookup of the verdicts
// recorded by eager sdpa_mask_decomposable probes (typically the warmup
// runs preceding a CUDA graph capture).  Returns 1 = decomposable on BOTH
// Q-tile grids, -1 = some probe said unsupported, 0 = no verdict recorded
// for this mask identity (never warmed up).  MUST NOT touch the device.
int sdpa_mask_verdict_cached(const at::Tensor& attn_mask) {
  auto mask = resolve_mask_cached(attn_mask, /*is_causal=*/false,
                                  attn_mask.size(-2), attn_mask.size(-1),
                                  attn_mask);
  auto fwd = lookup_verdict(mask, kFwdKblockM);
  auto bwd = lookup_verdict(mask, kBwdKblockM);
  if (!fwd.has_value() || !bwd.has_value()) return 0;
  return (*fwd && *bwd) ? 1 : -1;
}

// Post-replay envelope check: the graph-mode decomposition publishes its
// supported verdict into a pinned int on every replay; a zero here means
// the mask content replayed against drifted outside the layered-interval
// envelope (device-side faults cannot surface from graph replays on this
// platform, so this host-readable flag is the loud-failure mechanism).
// The check is folded into every sdpa_fwd/bwd call
// (envelope_check_folded), so users need no manual post-replay call
// (mem-efficient-grade usage contract); the manual op
// sdpa_graph_mask_check() below stays available for immediate detection.
// envelope_check_folded is a pure host read of one pinned int — no device
// sync, therefore capture-safe (host code runs normally during stream
// capture).  No-op when no graph was ever captured here; after raising,
// reset the flag so one violation is reported once (the next replay
// refreshes it anyway).
static void envelope_check_folded() {
  if (!g_graph_flag) return;
  auto* flag = g_graph_flag->data_ptr<int>();  // pinned: host-writable
  if (*flag == 0) {
    *flag = 1;  // plain host write, capture-safe (no op recorded)
    TORCH_CHECK(false,
                "flex_flash_attention::sdpa: the mask replayed through a captured "
                "CUDA graph is outside the decomposition envelope (too many "
                "hole layers/slices); the graph's attention outputs are "
                "invalid. Re-validate the mask eagerly and re-capture.");
  }
}

void sdpa_graph_mask_check() {
  if (!g_graph_flag) return;
  TORCH_CHECK(*g_graph_flag->const_data_ptr<int>() != 0,
              "flex_flash_attention::sdpa: the mask replayed through a captured "
              "CUDA graph is outside the decomposition envelope (too many "
              "hole layers/slices); the graph's attention outputs are "
              "invalid. Re-validate the mask eagerly and re-capture.");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_bwd(
    const at::Tensor& grad_out, const at::Tensor& query,
    const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    std::array<bool, 3> grad_input_mask, const at::Tensor& out,
    const at::Tensor& logsumexp, const at::Tensor& rng_state,
    bool is_causal, double dropout_p, std::optional<double> scale) {
  envelope_check_folded();
  // BHSD -> BSHD views; grads may arrive as stride-0/broadcast views, so
  // normalize layouts the kernel requires.
  auto dout = grad_out.transpose(1, 2);
  if (dout.stride(-1) != 1) {
    dout = dout.contiguous();
  }
  auto q = query.transpose(1, 2);
  auto k = key.transpose(1, 2);
  auto v = value.transpose(1, 2);
  auto out_bshd = out.transpose(1, 2);
  const auto seqlen_q = query.size(-2);
  const auto seqlen_k = key.size(-2);

  // Same specialized fast path as sdpa_fwd (see there); bwd runs its own
  // 768-row Q-tile grid, but the synthetic single-slice descs are valid on
  // any grid and the explicit-mask descs are cached per (mask, grid).
  std::optional<at::Tensor> attn_mask_eff = attn_mask;
  if (attn_mask_eff.has_value() && !attn_mask_eff->defined()) {
    attn_mask_eff = std::nullopt;
  }
  // CUDA graph capture mode — same contract as sdpa_fwd: the 768-row bwd
  // grid decomposition is recorded into the graph and re-runs on every
  // replay.  rng_state arrives as the DEVICE tensor the captured fwd graph
  // published; the bwd kernel reads it on-device at each replay, so
  // fwd/bwd stay dropout-consistent on every replay (no frozen seed).
  const bool capturing = at::cuda::currentStreamCaptureStatus() !=
                         at::cuda::CaptureStatus::None;
  if (capturing && attn_mask_eff.has_value()) {
    auto mask = resolve_mask_cached(attn_mask_eff, /*is_causal=*/false,
                                    seqlen_q, seqlen_k, query);
    auto gd = graph_mode_decompose(mask, kBwdKblockM);
    auto dout = grad_out.transpose(1, 2);
    if (dout.stride(-1) != 1) {
      dout = dout.contiguous();
    }
    auto q = query.transpose(1, 2);
    auto k = key.transpose(1, 2);
    auto v = value.transpose(1, 2);
    auto out_bshd = out.transpose(1, 2);
    const auto bwd_op = query.scalar_type() == at::kFloat
                            ? flex_flash_attention_bwd_f32()
                            : flex_flash_attention_bwd();
    auto [dq, dk, dv] = bwd_op->call(
        q, k, v, out_bshd, logsumexp, dout,
        gd.slices[0], gd.slices[1], gd.slices[2], gd.slices[3],
        gd.slices[4], gd.rows[0], gd.slices[5], gd.slices[6],
        gd.vbatch, gd.rows[1], gd.rows[2],
        std::nullopt, std::nullopt, std::nullopt,  // dq/dk/dv
        scale,
        false,               // deterministic
        std::nullopt,        // softcap
        false,               // reduce_kv
        false,               // use_loop_k
        true,                // inner_min_to_max (measured bwd win)
        false,               // persistent_scheduler (kept OFF for bwd)
        dropout_p,
        rng_state.defined() ? std::optional<at::Tensor>(rng_state)
                            : std::optional<at::Tensor>(std::nullopt),
        std::nullopt,        // mask_bits
        std::nullopt, std::nullopt, std::nullopt);
    at::Tensor grad_q, grad_k, grad_v;
    if (grad_input_mask[0]) grad_q = dq.transpose(1, 2);
    if (grad_input_mask[1]) grad_k = dk.transpose(1, 2);
    if (grad_input_mask[2]) grad_v = dv.transpose(1, 2);
    return {grad_q, grad_k, grad_v};
  }
  DecompEntry ent;
  if (!attn_mask_eff.has_value()) {
    DecompKey key{{nullptr, 0, nullptr, seqlen_q, seqlen_k, is_causal},
                  kBwdKblockM};
    {
      std::lock_guard<std::mutex> lk(g_decomp_mtx);
      auto it = g_decomp_cache.find(key);
      if (it != g_decomp_cache.end()) {
        ent = it->second;
      }
    }
    if (ent.n == 0) {
      ent = synth_desc(is_causal, seqlen_q, seqlen_k, query);
      std::lock_guard<std::mutex> lk(g_decomp_mtx);
      if (g_decomp_cache.size() >= kDecompCacheMax) g_decomp_cache.clear();
      g_decomp_cache.emplace(key, ent);
    }
  } else {
    auto mask = resolve_mask_cached(attn_mask_eff, /*is_causal=*/false,
                                    seqlen_q, seqlen_k, query);
    ent = decomp_cached(mask, kBwdKblockM);
  }
  TORCH_CHECK(ent.supported,
              "flex_flash_attention::sdpa_bwd: mask is outside the decomposition "
              "envelope (too many hole layers/slices); it should have been "
              "rejected by can_use_flex_flash_attention");

  // fp32 rides the isolated bwd_f32 kernel set; fp16/bf16 keep bwd.
  const auto bwd_op = query.scalar_type() == at::kFloat
                          ? flex_flash_attention_bwd_f32()
                          : flex_flash_attention_bwd();
  auto [dq, dk, dv] = bwd_op->call(
      q, k, v, out_bshd, logsumexp, dout,
      ent.q_starts, ent.q_ends, ent.k_starts, ent.k_ends, ent.mask_types,
      ent.row_to_slice, ent.diagonal_offsets, ent.band_widths,
      ent.vbatch, ent.row_to_vb_start, ent.row_to_vb_end,
      std::nullopt, std::nullopt, std::nullopt,  // dq/dk/dv
      scale,
      false,               // deterministic
      std::nullopt,        // softcap
      false,               // reduce_kv
      false,               // use_loop_k
      true,                // inner_min_to_max (measured bwd win)
      // persistent_scheduler: kept OFF for bwd — A/B measured work-stealing
      // as a clear fwd win (causal -8..-18%) but a bwd loss (+24..+68%
      // fwd+bwd: atomic dK/dV contention / L2 reuse degrades under
      // dynamic tile order), so only sdpa_fwd uses use_dyn_sched().
      false,
      dropout_p,           // dropout_p
      // rng_state arrives undefined on the dropout-free path; the kernel
      // only requires it when dropout_p > 0 (then fwd always returned it).
      rng_state.defined() ? std::optional<at::Tensor>(rng_state)
                          : std::optional<at::Tensor>(std::nullopt),
      std::nullopt,        // mask_bits
      std::nullopt, std::nullopt, std::nullopt);

  // autograd convention: skip grads come back undefined.
  at::Tensor grad_q, grad_k, grad_v;
  if (grad_input_mask[0]) grad_q = dq.transpose(1, 2);
  if (grad_input_mask[1]) grad_k = dk.transpose(1, 2);
  if (grad_input_mask[2]) grad_v = dv.transpose(1, 2);
  return {grad_q, grad_k, grad_v};
}

} // namespace flex_flash_attention
