// Public C++ contract of the FA3 flex flash attention library for the pytorch
// SDPA integration (SDPBackend::flex_flash_attention).
//
// The pytorch fork compiles against this header and links
// libflex_flash_attention.so (built by flex_flash_attention/build_lib.py, driven from
// cmake/flex_flash_attention.cmake).  Tensors follow SDPA conventions:
// BHSD, fp16/bf16 (fp32 routes to the isolated fwd_f32/bwd_f32 kernel set),
// mask None / 2-D (sq, sk) / 4-D (1, 1, sq, sk) boolean.
//
// The returned triples match the aten op contract:
//   fwd -> (output, logsumexp, rng_state)
//   bwd -> (grad_query, grad_key, grad_value)   // skipped grads undefined
#pragma once

// Forward-declare at::Tensor instead of including ATen headers: the torch
// side compiles with -DAT_PER_OPERATOR_HEADERS, which forbids umbrella
// includes like <ATen/ATen.h>; every caller already holds a Tensor type.
namespace at {
class Tensor;
}

#include <array>
#include <optional>
#include <tuple>

namespace flex_flash_attention {

// Runtime availability probe used by can_use_flex_flash_attention.
// True once the library is linked into the process (its kernel ops are
// registered by static initializers at load time).  The fp32 kernel set
// (fwd_f32/bwd_f32) is mandatory: a build without it reports unavailable.
bool sdpa_available();

// Content-level envelope probe used by can_use_flex_flash_attention
// before backend selection.  `attn_mask` must be boolean, 2-D (sq, sk) or
// 4-D (1, 1, sq, sk), on CUDA (can_use guarantees all of that).  Returns
// true iff the mask decomposes within the layered-interval envelope for
// BOTH the fwd (128-row) and bwd (768-row) Q-tile grids, so training on
// an accepted mask can never hit an unsupported backward.  Decomposition
// results are cached by mask identity, so sdpa_fwd/bwd reuse them.
bool sdpa_mask_decomposable(const at::Tensor& attn_mask);

// CUDA-graph support.  Graph replay executes no host code, so:
//   * the capture-time gate cannot run the device probe — it consults the
//     host-side verdict recorded by an earlier EAGER probe of the same
//     mask identity (the warmup run preceding capture);
//   * mask content changes between replays are tracked by re-running the
//     decomposition inside the graph (graph mode in sdpa_fwd/bwd);
//   * a replay whose mask drifted outside the envelope publishes a zero
//     into a pinned flag instead of faulting (PPU swallows device-side
//     errors inside replays) — call sdpa_graph_mask_check() after each
//     graph.replay() to surface it loudly.
// sdpa_mask_verdict_cached returns 1 = decomposable on both Q-tile grids,
// -1 = unsupported, 0 = no eager verdict recorded for this mask identity.
// It is pure host bookkeeping and capture-safe.
int sdpa_mask_verdict_cached(const at::Tensor& attn_mask);

// Throws if the most recent graph replay through this backend decomposed
// a mask outside the envelope.  No-op when no graph was ever captured
// here.  This check is ALSO folded into every sdpa_fwd/bwd call, so
// calling this manually is OPTIONAL — do it right after graph.replay()
// only when you need immediate detection instead of at the next call.
void sdpa_graph_mask_check();

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_fwd(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, bool is_causal,
    double dropout_p, std::optional<double> scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_bwd(
    const at::Tensor& grad_out, const at::Tensor& query,
    const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    std::array<bool, 3> grad_input_mask, const at::Tensor& out,
    const at::Tensor& logsumexp, const at::Tensor& rng_state,
    bool is_causal, double dropout_p, std::optional<double> scale);

} // namespace flex_flash_attention
