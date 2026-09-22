/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// C++ API and pybind11 registration for flex flash attention forward.
//
// Entry point: mha_flex_flash_fwd() — called from Python via torch.ops.flex_flash_attention.fwd
// Populates FlexFlashAttentionFwdParams and dispatches to run_mha_flex_flash_fwd.
//
// This file is compiled by the host C++ compiler (not hgcc), so it must NOT
// include any GPU-specific headers (cute, cutlass, launch templates).
// The template dispatch is in flex_flash_attention_dispatch.h (compiled by hgcc).

// Python.h and the PyInit stub are only needed for the wheel build; the
// library build (libflex_flash_attention.so for the pytorch SDPA integration)
// defines FLEX_FLASH_ATTENTION_LIBRARY_BUILD to skip them.
#ifndef FLEX_FLASH_ATTENTION_LIBRARY_BUILD
#include <Python.h>
#endif
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <torch/nn/functional/padding.h>
#include <ATen/EmptyTensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include "../hopper/flash.h"
#include "../hopper/cuda_check.h"
#include "attn_slice.h"

#ifndef FLEX_FLASH_ATTENTION_LIBRARY_BUILD
extern "C" {
PyObject* PyInit__flex_flash_C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_flex_flash_C",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}
}
#endif

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Forward declaration — implemented in flex_flash_attention_dispatch.cu
void run_mha_flex_flash_fwd(FlexFlashAttentionFwdParams &params, hggcStream_t stream);
// Forward declaration — implemented in instantiations/flex_flash_attention_bwd_dispatch.cu
void run_mha_flex_flash_bwd(FlexFlashAttentionBwdParams &params, hggcStream_t stream);
// fp32 kernel set — implemented in instantiations/flex_flash_attention_f32_dispatch.cu
void run_mha_flex_flash_fwd_f32(FlexFlashAttentionFwdParams &params, hggcStream_t stream);
void run_mha_flex_flash_bwd_f32(FlexFlashAttentionBwdParams &params, hggcStream_t stream);

static int round_up_headdim_arb(int x) {
    if (x <= 64) return 64;
    if (x <= 96) return 96;
    if (x <= 128) return 128;
    if (x <= 192) return 192;
    return 256;
}

// EXPERIMENTAL narrow-headdim buckets for the ISOLATED fp32 kernel set
// (fwd only): D<=16 maps to the dedicated hdim16 instantiation and D<=32
// to the hdim32 instantiation instead of padding into the 64 bucket (up to
// 8x / 3.76x compute waste).  hdim8 was tried and produces wrong results —
// silent minimum-granularity violation below 16.  Must stay in sync with
// the <=16/<=32 switch arms in flex_flash_attention_f32_dispatch.cu and the fwd
// hdim16/hdim32 TUs in sources.py.  The 16-bit set keeps
// round_up_headdim_arb() untouched.
static int round_up_headdim_arb_f32(int x) {
    if (x <= 16) return 16;
    if (x <= 32) return 32;  // EXPERIMENTAL fwd hdim32 bucket
    return round_up_headdim_arb(x);
}

// EXPERIMENTAL bwd counterpart: the narrowest bwd width is 32 (the generic
// CVT-free bwd path's MMA atom geometry rejects 16).  Must stay in sync
// with the <=32 arm of HEADDIM_SWITCH_ARB_BWD_F32 in
// flex_flash_attention_f32_dispatch.cu and the bwd hdim32 TU in sources.py.
static int round_up_headdim_arb_f32_bwd(int x) {
    if (x <= 32) return 32;
    return round_up_headdim_arb(x);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_fwd_impl(
    at::Tensor q,       // (b, s_q, h, d)
    at::Tensor k,       // (b, s_k, h_k, d)
    at::Tensor v,       // (b, s_k, h_k, dv)
    at::Tensor q_starts,       // [num_slices] int32
    at::Tensor q_ends,         // [num_slices] int32
    at::Tensor k_starts,       // [num_slices] int32
    at::Tensor k_ends,         // [num_slices] int32
    at::Tensor mask_types,     // [num_slices] int32
    at::Tensor row_to_slice,   // [seqlen_q] int32
    at::Tensor diagonal_offsets, // [num_slices] int32
    at::Tensor band_widths,    // [num_slices] int32
    std::optional<at::Tensor> vbatch_to_slice_,      // [num_vbatches] int32 (RangeMerge)
    std::optional<at::Tensor> row_to_vbatch_start_,  // [seqlen_q] int32 (RangeMerge)
    std::optional<at::Tensor> row_to_vbatch_end_,    // [seqlen_q] int32 (RangeMerge)
    std::optional<at::Tensor> out_,
    std::optional<double> softmax_scale_,
    std::optional<double> softcap_,
    bool inner_min_to_max,
    bool persistent_scheduler,
    std::optional<bool> use_kblockm128_,
    bool kblockn64,
    double p_dropout_,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor> mask_bits_,  // (s_q, stride) uint8 — SLICE_BITMASK fallback
    std::optional<at::Tensor> bh_to_group_,          // [b*h] int32 (P3 hetero layout)
    std::optional<at::Tensor> group_slice_offsets_,  // [num_groups+1] int32 prefix sum
    std::optional<at::Tensor> group_vb_offsets_,     // [num_groups+1] int32 prefix sum
    int64_t trivial_mask_,                           // 0=generic, 1=FULL, 2=CAUSAL (host-proven)
    int64_t trivial_diagonal_,                       // diagonal offset of the trivial CAUSAL slice
    bool fp32_path                                   // fwd_f32 entry: isolated fp32/TF32 kernel set
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_bf16 = q.dtype() == at::ScalarType::BFloat16;
    auto q_type = q.dtype();
    if (!fp32_path) {
        TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
                    "Only fp16 and bf16 are supported");
    } else {
        TORCH_CHECK(q_type == at::ScalarType::Float,
                    "fwd_f32 requires float32 q/k/v inputs");
    }

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "k must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "v must have contiguous last dimension");

    int batch_size = q.size(0);
    int seqlen_q = q.size(1);
    int num_heads = q.size(2);
    int head_size = q.size(3);
    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);
    int head_size_v = v.size(3);

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_v);
    TORCH_CHECK(std::max(head_size, head_size_v) <= 256,
                "head_dim and head_dim_v must be <= 256, got ", head_size, " and ", head_size_v);
    TORCH_CHECK(num_heads % num_heads_k == 0,
                "num_heads must be divisible by num_heads_k for GQA, got ", num_heads, " and ", num_heads_k);

    // Validate slice tensors
    int num_slices = q_starts.size(0);

    // P3 per-(batch, head) heterogeneous layouts (optional).  When present,
    // the per-slice arrays and the Q-row tables below are the FLAT
    // concatenation of num_groups distinct layouts (see AttnSliceParams);
    // row tables have num_groups * seqlen_q entries, and the group prefix
    // sums map each group to its sub-range.  Absent → single shared layout.
    bool const has_bh_layout = bh_to_group_.has_value();
    TORCH_CHECK(has_bh_layout == group_slice_offsets_.has_value()
             && has_bh_layout == group_vb_offsets_.has_value(),
                "bh_to_group/group_slice_offsets/group_vb_offsets must be provided together");
    int num_groups = 1;
    if (has_bh_layout) {
        at::Tensor bh_to_group = bh_to_group_.value();
        at::Tensor gsc = group_slice_offsets_.value();
        at::Tensor gvb = group_vb_offsets_.value();
        TORCH_CHECK(bh_to_group.dtype() == torch::kInt32, "bh_to_group must be int32");
        CHECK_DEVICE(bh_to_group); CHECK_DEVICE(gsc); CHECK_DEVICE(gvb);
        CHECK_CONTIGUOUS(bh_to_group); CHECK_CONTIGUOUS(gsc); CHECK_CONTIGUOUS(gvb);
        CHECK_SHAPE(bh_to_group, batch_size * num_heads);
        num_groups = gsc.size(0) - 1;
        TORCH_CHECK(num_groups >= 1, "group_slice_offsets must have at least 2 entries");
        CHECK_SHAPE(gvb, num_groups + 1);
    }
    int const row_table_len = has_bh_layout ? num_groups * seqlen_q : seqlen_q;

    TORCH_CHECK(q_starts.dtype() == torch::kInt32, "q_starts must be int32");
    TORCH_CHECK(row_to_slice.dtype() == torch::kInt32, "row_to_slice must be int32");
    CHECK_DEVICE(q_starts); CHECK_DEVICE(row_to_slice);
    CHECK_SHAPE(q_starts, num_slices);
    CHECK_SHAPE(q_ends, num_slices);
    CHECK_SHAPE(k_starts, num_slices);
    CHECK_SHAPE(k_ends, num_slices);
    CHECK_SHAPE(mask_types, num_slices);
    CHECK_SHAPE(row_to_slice, row_table_len);
    CHECK_SHAPE(diagonal_offsets, num_slices);
    CHECK_SHAPE(band_widths, num_slices);

    // Validate RangeMerge layout (Q ranges may overlap) if given.
    bool const has_merge_layout = vbatch_to_slice_.has_value();
    TORCH_CHECK(has_merge_layout == row_to_vbatch_start_.has_value()
             && has_merge_layout == row_to_vbatch_end_.has_value(),
                "vbatch_to_slice/row_to_vbatch_start/row_to_vbatch_end must be provided together");
    int num_vbatches = 0;
    if (has_merge_layout) {
        at::Tensor vbatch_to_slice = vbatch_to_slice_.value();
        at::Tensor row_to_vbatch_start = row_to_vbatch_start_.value();
        at::Tensor row_to_vbatch_end = row_to_vbatch_end_.value();
        TORCH_CHECK(vbatch_to_slice.dtype() == torch::kInt32, "vbatch_to_slice must be int32");
        CHECK_DEVICE(vbatch_to_slice); CHECK_DEVICE(row_to_vbatch_start); CHECK_DEVICE(row_to_vbatch_end);
        CHECK_CONTIGUOUS(vbatch_to_slice); CHECK_CONTIGUOUS(row_to_vbatch_start); CHECK_CONTIGUOUS(row_to_vbatch_end);
        num_vbatches = vbatch_to_slice.size(0);
        CHECK_SHAPE(row_to_vbatch_start, row_table_len);
        CHECK_SHAPE(row_to_vbatch_end, row_table_len);
    }

    // P3: the group prefix sums must match the flat table lengths (small
    // one-shot D2H of two int arrays; only paid by heterogeneous calls).
    if (has_bh_layout) {
        auto gsc = group_slice_offsets_.value().to(torch::kCPU);
        auto gvb = group_vb_offsets_.value().to(torch::kCPU);
        TORCH_CHECK(gsc[num_groups].item<int>() == num_slices,
                    "group_slice_offsets tail must equal total flat slice count");
        if (has_merge_layout) {
            TORCH_CHECK(gvb[num_groups].item<int>() == num_vbatches,
                        "group_vb_offsets tail must equal total flat vbatch count");
        }
    }

    // Validate the packed bitmask (SLICE_BITMASK fallback slices).  The row
    // stride pads seqlen_k up to a multiple of 128 bits so the kernel's
    // last K tile (kBlockN <= 128) never reads past the buffer; padding bits
    // must be zero (they double as the seqlen_k tail mask).
    unsigned char const* mask_bits_ptr = nullptr;
    int mask_row_stride = 0;
    if (mask_bits_.has_value()) {
        at::Tensor mask_bits = mask_bits_.value();
        int const stride_expected = ((seqlen_k + 127) / 128) * 16;
        TORCH_CHECK(mask_bits.dtype() == torch::kUInt8, "mask_bits must be uint8");
        CHECK_DEVICE(mask_bits);
        CHECK_CONTIGUOUS(mask_bits);
        CHECK_SHAPE(mask_bits, row_table_len, stride_expected);
        mask_bits_ptr = mask_bits.data_ptr<unsigned char>();
        mask_row_stride = stride_expected;
    }

    float softmax_scale = softmax_scale_.has_value() ? softmax_scale_.value() : 1.0f / sqrtf(float(head_size));
    float softcap = softcap_.has_value() ? float(softcap_.value()) : 0.0f;
    TORCH_CHECK(softcap >= 0.f, "softcap must be non-negative");

    // Round up over BOTH head dims: the kernel is instantiated with
    // kHeadDimV == kHeadDim, so Q/K and V must end up the same padded width.
    // Rounding on head_size alone made head_size_v > head_size produce a
    // negative pad and then narrow() past the end of the output.
    int const head_size_rounded = (fp32_path ? round_up_headdim_arb_f32
                                             : round_up_headdim_arb)(
        std::max(head_size, head_size_v));

    // Zero-pad Q/K/V to the rounded headdim when unaligned (e.g. 80/112):
    // the kernel addresses head_size_rounded columns per row, so a narrower
    // tensor would read OOB.  Zero padding is mathematically exact — padded
    // Q/K columns contribute nothing to scores and padded V columns add
    // zero to O.  softmax_scale above is derived from the ORIGINAL head_size,
    // so the padding cannot change the scores.
    bool const need_pad = head_size_rounded != head_size
                       || head_size_rounded != head_size_v;
    at::Tensor q_use = q, k_use = k, v_use = v;
    if (need_pad) {
        q_use = at::constant_pad_nd(q, {0, head_size_rounded - head_size});
        k_use = at::constant_pad_nd(k, {0, head_size_rounded - head_size});
        v_use = at::constant_pad_nd(v, {0, head_size_rounded - head_size_v});
    }

    // Output tensor
    at::Tensor out;
    auto opts = q.options();
    // Kernel-facing output: rounded width when headdim is unaligned.
    at::Tensor out_kernel;
    if (out_.has_value()) {
        out = out_.value();
        CHECK_DEVICE(out);
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
        out_kernel = need_pad
            ? torch::empty({batch_size, seqlen_q, num_heads, head_size_rounded}, opts)
            : out;
    } else {
        out_kernel = torch::empty({batch_size, seqlen_q, num_heads,
                                   need_pad ? head_size_rounded : head_size_v}, opts);
        out = need_pad ? out_kernel.narrow(-1, 0, head_size_v) : out_kernel;
    }

    // Softmax LSE
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    // Populate FlexFlashAttentionFwdParams
    FlexFlashAttentionFwdParams params;
    memset(&params, 0, sizeof(params));

    // Basic parameters
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.d = head_size_rounded;
    params.dv = head_size_rounded;
    params.d_rounded = head_size_rounded;
    params.dv_rounded = head_size_rounded;
    params.is_bf16 = is_bf16;
    params.scale_softmax = softmax_scale;
    params.softcap = softcap;

    // Feature flags (runtime dispatch in the launch template).
    params.inner_min_to_max = inner_min_to_max;
    params.persistent_scheduler = persistent_scheduler;
    // kBlockM=128 hint from the host (nullopt → launch-template heuristic).
    params.kblockm128_has_hint = use_kblockm128_.has_value();
    params.kblockm128_value = use_kblockm128_.value_or(false);
    params.kblockn64 = kblockn64;

    // Q pointers
    params.q_ptr = q_use.data_ptr();
    params.q_row_stride = q_use.stride(1);
    params.q_head_stride = q_use.stride(2);
    params.q_batch_stride = q_use.stride(0);

    // K pointers
    params.k_ptr = k_use.data_ptr();
    params.k_row_stride = k_use.stride(1);
    params.k_head_stride = k_use.stride(2);
    params.k_batch_stride = k_use.stride(0);

    // V pointers
    params.v_ptr = v_use.data_ptr();
    params.v_row_stride = v_use.stride(1);
    params.v_head_stride = v_use.stride(2);
    params.v_batch_stride = v_use.stride(0);
    params.v_dim_stride = v_use.stride(3);

    // Output pointers
    params.o_ptr = out_kernel.data_ptr();
    params.o_row_stride = out_kernel.stride(1);
    params.o_head_stride = out_kernel.stride(2);
    params.o_batch_stride = out_kernel.stride(0);

    // Softmax LSE
    params.softmax_lse_ptr = softmax_lse.data_ptr<float>();

    // Architecture
    params.arch = dprops->major == 8 ? (dprops->minor == 9 ? 89 : 80) : dprops->major * 10 + dprops->minor;
    params.num_sm = dprops->multiProcessorCount;

    // No causal/local/split for flex flash attention
    params.is_causal = false;
    params.is_local = false;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.num_splits = 1;
    params.num_splits_dynamic_ptr = nullptr;

    // Flex flash attention slice parameters
    params.is_flex_flash_attention = true;
    params.slices.q_starts = q_starts.data_ptr<int>();
    params.slices.q_ends = q_ends.data_ptr<int>();
    params.slices.k_starts = k_starts.data_ptr<int>();
    params.slices.k_ends = k_ends.data_ptr<int>();
    params.slices.mask_types = mask_types.data_ptr<int>();
    params.slices.row_to_slice = row_to_slice.data_ptr<int>();
    params.slices.diagonal_offsets = diagonal_offsets.data_ptr<int>();
    params.slices.band_widths = band_widths.data_ptr<int>();
    params.slices.num_slices = num_slices;

    // RangeMerge layout (nullptr → legacy single-slice dispatch in kernel).
    params.slices.vbatch_to_slice = has_merge_layout
        ? vbatch_to_slice_.value().data_ptr<int>() : nullptr;
    params.slices.row_to_vbatch_start = has_merge_layout
        ? row_to_vbatch_start_.value().data_ptr<int>() : nullptr;
    params.slices.row_to_vbatch_end = has_merge_layout
        ? row_to_vbatch_end_.value().data_ptr<int>() : nullptr;
    params.slices.num_vbatches = num_vbatches;

    // Bitmask fallback (nullptr for the four algebraic mask types).
    params.slices.mask_bits = mask_bits_ptr;
    params.slices.mask_row_stride = mask_row_stride;

    // P3 per-(b,h) heterogeneous layout (nullptr → single shared layout;
    // the kernel's rebase_slices() then degenerates to the identity).
    params.slices.bh_to_group = has_bh_layout
        ? bh_to_group_.value().data_ptr<int>() : nullptr;
    params.slices.group_slice_offsets = has_bh_layout
        ? group_slice_offsets_.value().data_ptr<int>() : nullptr;
    params.slices.group_vb_offsets = has_bh_layout
        ? group_vb_offsets_.value().data_ptr<int>() : nullptr;
    params.slices.num_groups = num_groups;

    // Trivial single-slice specialization (host-proven one FULL / CAUSAL
    // slice over the whole rectangle; the kernel then skips every slice
    // descriptor global load).  Only honored for a plain single-slice
    // layout — heterogeneous (P3) groups and bitmask fallbacks ride the
    // generic machinery.
    bool const trivial_ok =
        !has_bh_layout && mask_bits_ptr == nullptr &&
        (trivial_mask_ == 1 || trivial_mask_ == 2);
    params.slices.trivial_mask = trivial_ok ? int(trivial_mask_) : 0;
    params.slices.trivial_diagonal = int(trivial_diagonal_);

    // Dynamic persistent scheduler: zero-initialized tile counter.
    at::Tensor tile_count_semaphore;
    if (persistent_scheduler) {
        tile_count_semaphore = torch::zeros({1}, opts.dtype(at::kInt));
        params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
    } else {
        params.tile_count_semaphore = nullptr;
    }

    // Dropout (FA2 conventions): p_dropout stores the KEEP probability and
    // p_dropout_in_uint8_t = floor(p_keep * 255) (255 == keep all, the
    // dropout-free path).  The Philox (seed, offset) pair is drawn from the
    // CUDA generator and written host-side into rng_state, which is returned
    // so the backward can replay the exact same mask.  The kernel keys the
    // stream on global (row, col) coordinates plus a per-(batch, head)
    // offset, so slice/zone iteration order cannot change the mask.
    float const p_dropout = static_cast<float>(p_dropout_);
    TORCH_CHECK(p_dropout >= 0.f && p_dropout < 1.f,
                "dropout_p must be in [0, 1), got ", p_dropout);
    // CUDA graph capture: the host cannot draw a Philox pair during capture
    // (philox_engine_inputs refuses), and by-value (seed, offset) baked into
    // the launch would replay the SAME dropout mask forever.  Graph mode
    // therefore returns a DEVICE rng_state that the fwd kernel populates,
    // and the kernel resolves its stream from the generator's extragraph
    // tensors at every replay (mem_eff convention; PyTorch refreshes them
    // before each graph.replay()).
    const bool capturing =
        c10::cuda::currentStreamCaptureStatusMayInitCtx() !=
        c10::cuda::CaptureStatus::None;
    at::Tensor rng_state = capturing
        ? torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64).device(at::kCUDA))
        // CPU tensor: the host writes it here and the launch template reads it
        // back by value (a device tensor would be dereferenced on the host).
        : torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
    params.rng_state = capturing
        ? nullptr
        : reinterpret_cast<uint64_t*>(rng_state.data_ptr());
    if (p_dropout > 0.f) {
        params.p_dropout = 1.f - p_dropout;
        params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
        params.rp_dropout = 1.f / params.p_dropout;
        // Custom RNG: the counter advances by b * h * 32 per draw (FA2).
        int64_t counter_offset = params.b * params.h * 32;
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        if (capturing) {
            // Graph-safe RNG state: pointers into the generator-owned
            // extragraph tensors (alive across replays) plus this graph's
            // intragraph counter.  The kernel also publishes the (seed,
            // offset) pair it actually used into rng_state so the captured
            // bwd graph replays the identical stream.
            auto philox_state = gen->philox_cuda_state(counter_offset);
            params.philox_seed_ptr =
                reinterpret_cast<const unsigned long long*>(philox_state.seed_.ptr);
            params.philox_offset_ptr =
                reinterpret_cast<const unsigned long long*>(philox_state.offset_.ptr);
            params.intragraph_offset = philox_state.offset_intragraph_;
            params.rng_state_dev =
                reinterpret_cast<unsigned long long*>(rng_state.data_ptr());
        } else {
            // philox_engine_inputs advances the generator state by
            // counter_offset and returns the (seed, offset) pair to use now.
            auto seeds = gen->philox_engine_inputs(counter_offset);
            params.rng_state[0] = seeds.first;
            params.rng_state[1] = seeds.second;
        }
    } else {
        params.p_dropout = 1.f;
        params.p_dropout_in_uint8_t = 255;
        params.rp_dropout = 1.f;
        if (!capturing) {
            params.rng_state[0] = 0;
            params.rng_state[1] = 0;
        }
    }

    hggcStream_t stream = (hggcStream_t)at::cuda::getCurrentCUDAStream().stream();
    if (fp32_path) {
        run_mha_flex_flash_fwd_f32(params, stream);
    } else {
        run_mha_flex_flash_fwd(params, stream);
    }

    if (need_pad && out_.has_value()) {
        out.copy_(out_kernel.narrow(-1, 0, head_size_v));
    }
    return std::make_tuple(out, softmax_lse, rng_state);
}

// fp16/bf16 fwd entry — unchanged contract.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_fwd(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor q_starts, at::Tensor q_ends, at::Tensor k_starts, at::Tensor k_ends,
    at::Tensor mask_types, at::Tensor row_to_slice, at::Tensor diagonal_offsets,
    at::Tensor band_widths,
    std::optional<at::Tensor> vbatch_to_slice_,
    std::optional<at::Tensor> row_to_vbatch_start_,
    std::optional<at::Tensor> row_to_vbatch_end_,
    std::optional<at::Tensor> out_,
    std::optional<double> softmax_scale_,
    std::optional<double> softcap_,
    bool inner_min_to_max,
    bool persistent_scheduler,
    std::optional<bool> use_kblockm128_,
    bool kblockn64,
    double p_dropout_,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor> mask_bits_,
    std::optional<at::Tensor> bh_to_group_,
    std::optional<at::Tensor> group_slice_offsets_,
    std::optional<at::Tensor> group_vb_offsets_,
    int64_t trivial_mask_,
    int64_t trivial_diagonal_
) {
    return mha_flex_flash_fwd_impl(
        q, k, v, q_starts, q_ends, k_starts, k_ends, mask_types, row_to_slice,
        diagonal_offsets, band_widths, vbatch_to_slice_, row_to_vbatch_start_,
        row_to_vbatch_end_, out_, softmax_scale_, softcap_, inner_min_to_max,
        persistent_scheduler, use_kblockm128_, kblockn64, p_dropout_, gen_,
        mask_bits_, bh_to_group_, group_slice_offsets_, group_vb_offsets_,
        trivial_mask_, trivial_diagonal_,
        false);
}

// ISOLATED fp32 fwd entry — requires float32 inputs, runs the TF32
// tensor-core kernel set (flex_flash_attention_*_f32 instantiations).
std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_fwd_f32(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor q_starts, at::Tensor q_ends, at::Tensor k_starts, at::Tensor k_ends,
    at::Tensor mask_types, at::Tensor row_to_slice, at::Tensor diagonal_offsets,
    at::Tensor band_widths,
    std::optional<at::Tensor> vbatch_to_slice_,
    std::optional<at::Tensor> row_to_vbatch_start_,
    std::optional<at::Tensor> row_to_vbatch_end_,
    std::optional<at::Tensor> out_,
    std::optional<double> softmax_scale_,
    std::optional<double> softcap_,
    bool inner_min_to_max,
    bool persistent_scheduler,
    std::optional<bool> use_kblockm128_,
    bool kblockn64,
    double p_dropout_,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor> mask_bits_,
    std::optional<at::Tensor> bh_to_group_,
    std::optional<at::Tensor> group_slice_offsets_,
    std::optional<at::Tensor> group_vb_offsets_,
    int64_t trivial_mask_,
    int64_t trivial_diagonal_
) {
    return mha_flex_flash_fwd_impl(
        q, k, v, q_starts, q_ends, k_starts, k_ends, mask_types, row_to_slice,
        diagonal_offsets, band_widths, vbatch_to_slice_, row_to_vbatch_start_,
        row_to_vbatch_end_, out_, softmax_scale_, softcap_, inner_min_to_max,
        persistent_scheduler, use_kblockm128_, kblockn64, p_dropout_, gen_,
        mask_bits_, bh_to_group_, group_slice_offsets_, group_vb_offsets_,
        trivial_mask_, trivial_diagonal_,
        true);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_bwd_impl(
    at::Tensor q,       // (b, s_q, h, d)
    at::Tensor k,       // (b, s_k, h_k, d)
    at::Tensor v,       // (b, s_k, h_k, dv)
    at::Tensor out,     // (b, s_q, h, dv)   — from fwd
    at::Tensor softmax_lse,  // (b, h, s_q)  — from fwd (natural log)
    at::Tensor dout,    // (b, s_q, h, dv)
    at::Tensor q_starts,       // [num_slices] int32
    at::Tensor q_ends,         // [num_slices] int32
    at::Tensor k_starts,       // [num_slices] int32
    at::Tensor k_ends,         // [num_slices] int32
    at::Tensor mask_types,     // [num_slices] int32
    at::Tensor row_to_slice,   // [seqlen_q] int32
    at::Tensor diagonal_offsets, // [num_slices] int32
    at::Tensor band_widths,    // [num_slices] int32
    std::optional<at::Tensor> vbatch_to_slice_,      // [num_vbatches] int32 (RangeMerge)
    std::optional<at::Tensor> row_to_vbatch_start_,  // [seqlen_q] int32 (RangeMerge)
    std::optional<at::Tensor> row_to_vbatch_end_,    // [seqlen_q] int32 (RangeMerge)
    std::optional<at::Tensor> dq_,
    std::optional<at::Tensor> dk_,
    std::optional<at::Tensor> dv_,
    std::optional<double> softmax_scale_,
    bool deterministic,
    std::optional<double> softcap_,
    bool reduce_kv,
    bool use_loop_k,
    bool inner_min_to_max,
    bool persistent_scheduler,
    double p_dropout_,
    std::optional<at::Tensor> rng_state_,
    std::optional<at::Tensor> mask_bits_,  // (s_q, stride) uint8 — SLICE_BITMASK fallback
    std::optional<at::Tensor> bh_to_group_,          // [b*h] int32 (P3 hetero layout)
    std::optional<at::Tensor> group_slice_offsets_,  // [num_groups+1] int32 prefix sum
    std::optional<at::Tensor> group_vb_offsets_,     // [num_groups+1] int32 prefix sum
    bool fp32_path                                   // bwd_f32 entry: isolated fp32/TF32 kernel set
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_bf16 = q.dtype() == at::ScalarType::BFloat16;
    auto q_type = q.dtype();
    if (!fp32_path) {
        TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
                    "Only fp16 and bf16 are supported");
    } else {
        TORCH_CHECK(q_type == at::ScalarType::Float,
                    "bwd_f32 requires float32 inputs");
    }
    TORCH_CHECK(!reduce_kv || use_loop_k, "reduce_kv requires use_loop_k (LoopK mainloop)");

    // P3 Stage A scope: the deterministic turnstile and the LoopK mainloop
    // have not been validated with heterogeneous layouts yet — rejected at
    // the API once the layout tensors are validated below.

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v); CHECK_DEVICE(out); CHECK_DEVICE(dout);
    TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "k must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "v must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout must have contiguous last dimension");

    int batch_size = q.size(0);
    int seqlen_q = q.size(1);
    int num_heads = q.size(2);
    int head_size = q.size(3);
    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);
    int head_size_v = v.size(3);

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_v);
    TORCH_CHECK(std::max(head_size, head_size_v) <= 256,
                "head_dim and head_dim_v must be <= 256, got ", head_size, " and ", head_size_v);
    TORCH_CHECK(num_heads % num_heads_k == 0,
                "num_heads must be divisible by num_heads_k for GQA, got ", num_heads, " and ", num_heads_k);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_v);
    CHECK_SHAPE(softmax_lse, batch_size, num_heads, seqlen_q);
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must be float32");
    CHECK_DEVICE(softmax_lse);
    CHECK_CONTIGUOUS(softmax_lse);  // preprocess reads it with contiguous strides

    // Validate slice tensors (same layout as fwd)
    int num_slices = q_starts.size(0);

    // P3 per-(batch, head) heterogeneous layouts (optional).  When present,
    // the per-slice arrays and the Q-row tables below are the FLAT
    // concatenation of num_groups distinct layouts (see AttnSliceParams);
    // row tables have num_groups * seqlen_q entries, and the group prefix
    // sums map each group to its sub-range.  Absent → single shared layout.
    bool const has_bh_layout = bh_to_group_.has_value();
    TORCH_CHECK(has_bh_layout == group_slice_offsets_.has_value()
             && has_bh_layout == group_vb_offsets_.has_value(),
                "bh_to_group/group_slice_offsets/group_vb_offsets must be provided together");
    int num_groups = 1;
    if (has_bh_layout) {
        at::Tensor bh_to_group = bh_to_group_.value();
        at::Tensor gsc = group_slice_offsets_.value();
        at::Tensor gvb = group_vb_offsets_.value();
        TORCH_CHECK(bh_to_group.dtype() == torch::kInt32, "bh_to_group must be int32");
        CHECK_DEVICE(bh_to_group); CHECK_DEVICE(gsc); CHECK_DEVICE(gvb);
        CHECK_CONTIGUOUS(bh_to_group); CHECK_CONTIGUOUS(gsc); CHECK_CONTIGUOUS(gvb);
        CHECK_SHAPE(bh_to_group, batch_size * num_heads);
        num_groups = gsc.size(0) - 1;
        TORCH_CHECK(num_groups >= 1, "group_slice_offsets must have at least 2 entries");
        CHECK_SHAPE(gvb, num_groups + 1);
    }
    int const row_table_len = has_bh_layout ? num_groups * seqlen_q : seqlen_q;
    TORCH_CHECK(!has_bh_layout || !deterministic,
                "deterministic bwd is not yet supported with per-(b,h) heterogeneous masks");
    TORCH_CHECK(!has_bh_layout || !use_loop_k,
                "use_loop_k/reduce_kv bwd is not yet supported with per-(b,h) heterogeneous masks");

    TORCH_CHECK(q_starts.dtype() == torch::kInt32, "q_starts must be int32");
    TORCH_CHECK(row_to_slice.dtype() == torch::kInt32, "row_to_slice must be int32");
    CHECK_DEVICE(q_starts); CHECK_DEVICE(row_to_slice);
    CHECK_SHAPE(q_starts, num_slices);
    CHECK_SHAPE(q_ends, num_slices);
    CHECK_SHAPE(k_starts, num_slices);
    CHECK_SHAPE(k_ends, num_slices);
    CHECK_SHAPE(mask_types, num_slices);
    CHECK_SHAPE(row_to_slice, row_table_len);
    CHECK_SHAPE(diagonal_offsets, num_slices);
    CHECK_SHAPE(band_widths, num_slices);

    // Validate RangeMerge layout (Q ranges may overlap) if given.
    bool const has_merge_layout = vbatch_to_slice_.has_value();
    TORCH_CHECK(has_merge_layout == row_to_vbatch_start_.has_value()
             && has_merge_layout == row_to_vbatch_end_.has_value(),
                "vbatch_to_slice/row_to_vbatch_start/row_to_vbatch_end must be provided together");
    int num_vbatches = 0;
    if (has_merge_layout) {
        at::Tensor vbatch_to_slice = vbatch_to_slice_.value();
        at::Tensor row_to_vbatch_start = row_to_vbatch_start_.value();
        at::Tensor row_to_vbatch_end = row_to_vbatch_end_.value();
        TORCH_CHECK(vbatch_to_slice.dtype() == torch::kInt32, "vbatch_to_slice must be int32");
        CHECK_DEVICE(vbatch_to_slice); CHECK_DEVICE(row_to_vbatch_start); CHECK_DEVICE(row_to_vbatch_end);
        CHECK_CONTIGUOUS(vbatch_to_slice); CHECK_CONTIGUOUS(row_to_vbatch_start); CHECK_CONTIGUOUS(row_to_vbatch_end);
        num_vbatches = vbatch_to_slice.size(0);
        CHECK_SHAPE(row_to_vbatch_start, row_table_len);
        CHECK_SHAPE(row_to_vbatch_end, row_table_len);
    }

    // P3: the group prefix sums must match the flat table lengths (small
    // one-shot D2H of two int arrays; only paid by heterogeneous calls).
    if (has_bh_layout) {
        auto gsc = group_slice_offsets_.value().to(torch::kCPU);
        auto gvb = group_vb_offsets_.value().to(torch::kCPU);
        TORCH_CHECK(gsc[num_groups].item<int>() == num_slices,
                    "group_slice_offsets tail must equal total flat slice count");
        if (has_merge_layout) {
            TORCH_CHECK(gvb[num_groups].item<int>() == num_vbatches,
                        "group_vb_offsets tail must equal total flat vbatch count");
        }
    }

    // Validate the packed bitmask (SLICE_BITMASK fallback slices); same
    // layout contract as the forward entry point.
    unsigned char const* mask_bits_ptr = nullptr;
    int mask_row_stride = 0;
    if (mask_bits_.has_value()) {
        at::Tensor mask_bits = mask_bits_.value();
        int const stride_expected = ((seqlen_k + 127) / 128) * 16;
        TORCH_CHECK(mask_bits.dtype() == torch::kUInt8, "mask_bits must be uint8");
        CHECK_DEVICE(mask_bits);
        CHECK_CONTIGUOUS(mask_bits);
        CHECK_SHAPE(mask_bits, row_table_len, stride_expected);
        mask_bits_ptr = mask_bits.data_ptr<unsigned char>();
        mask_row_stride = stride_expected;
    }

    float softmax_scale = softmax_scale_.has_value() ? softmax_scale_.value() : 1.0f / sqrtf(float(head_size));
    float softcap = softcap_.has_value() ? float(softcap_.value()) : 0.0f;
    TORCH_CHECK(softcap >= 0.f, "softcap must be non-negative");

    // Dropout replay: same conventions as fwd (p_dropout stores KEEP prob,
    // uint8 255 == dropout-free).  With p > 0 the caller MUST pass back the
    // rng_state tensor returned by fwd so the kernel replays the identical
    // coordinate-keyed Philox mask.
    float const p_dropout = static_cast<float>(p_dropout_);
    TORCH_CHECK(p_dropout >= 0.f && p_dropout < 1.f,
                "dropout_p must be in [0, 1), got ", p_dropout);
    if (p_dropout > 0.f) {
        TORCH_CHECK(rng_state_.has_value(),
                    "rng_state (returned by fwd) is required when dropout_p > 0");
        at::Tensor rng_state = rng_state_.value();
        TORCH_CHECK(rng_state.dtype() == torch::kInt64, "rng_state must be int64");
        // CUDA tensors arrive from captured fwd graphs (the kernel-published
        // pair); eager fwd returns CPU tensors read host-side.
        CHECK_SHAPE(rng_state, 2);
    }

    // Same reasoning as fwd: round up over BOTH head dims so Q/K and V reach
    // the single width the kernel is instantiated for.
    // NOTE: bwd uses its own narrow bucket (min 32, not 16): the generic
    // CVT-free bwd path's MMA geometry rejects width 16, while 32 keeps the
    // per-warp MMA N slice a multiple of 16 with the legacy atom layout.
    int const head_size_rounded = (fp32_path ? round_up_headdim_arb_f32_bwd
                                                 : round_up_headdim_arb)(
            std::max(head_size, head_size_v));

    // Zero-pad inputs to the rounded headdim when unaligned (same rationale
    // as fwd); out/dO are padded too — preprocess reads dv_rounded columns
    // per row (delta = rowsum(dO * O)) and the mainloop consumes Q/K/V at
    // the rounded width.
    bool const need_pad = head_size_rounded != head_size
                       || head_size_rounded != head_size_v;
    at::Tensor q_use = q, k_use = k, v_use = v, out_use = out, dout_use = dout;
    if (need_pad) {
        q_use = at::constant_pad_nd(q, {0, head_size_rounded - head_size});
        k_use = at::constant_pad_nd(k, {0, head_size_rounded - head_size});
        v_use = at::constant_pad_nd(v, {0, head_size_rounded - head_size_v});
        out_use = at::constant_pad_nd(out, {0, head_size_rounded - head_size_v});
        dout_use = at::constant_pad_nd(dout, {0, head_size_rounded - head_size_v});
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // seqlen_q_rounded: preprocess/postprocess/dQaccum address Q in kBlockM
    // tiles and write up to ceil_div(seqlen_q, kBlockM) * kBlockM rows, so
    // the rounded length must cover that range for EVERY bwd tile config
    // (kBlockM ∈ {48,64,96,128}).  384 = lcm(48,64,96,128) is the smallest
    // such unit; rounding to 128 alone let kBlockM=48 tiles run past the
    // per-head accum region whenever S % 128 == 0 && S % 48 != 0 (e.g.
    // S=4096: tiles reach row 4128 > 4096 rounded rows -> OOB atomicAdds
    // landed in the NEXT head's dQaccum rows 0..31, corrupting dq/dk/dv).
    int const seqlen_q_rounded = round_multiple(seqlen_q, 384);
    // seqlen_k_rounded: the GQA epilogue atomicAdds dK/dV over the FULL
    // kBlockN tile with no seqlen predication, so the accum buffers MUST
    // cover ceil_div(seqlen_k, kBlockN) * kBlockN rows for every bwd config
    // (kBlockN ∈ {64,128}).  Additionally the dK/dV convert kernel loads each
    // FULL kBlockN=128 x d tile of the accum (no bounds check) and its CVT
    // read-back permutation sources smem positions spanning the entire 128
    // tile rows, so output rows near the tail read accum rows up to
    // ceil_div(seqlen_k,128)*128 - 1.  192 = lcm(64,128) covers the epilogue
    // writes but NOT the convert reads (e.g. seqlen_k=528: sk_r=576 but the
    // last convert tile spans rows 512..639 -> OOB reads of stale memory
    // surfaced as -inf/garbage dv rows 512..527 cols 64..95).  384 =
    // lcm(192,128) covers both.
    int const seqlen_k_rounded = round_multiple(seqlen_k, 384);

    at::Tensor dq, dk, dv;        // returned to caller (original width)
    at::Tensor dq_k, dk_k, dv_k;  // kernel-facing (rounded width if padded)
    auto qopts = q.options();
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_type, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
        dq_k = need_pad
            ? torch::empty({batch_size, seqlen_q, num_heads, head_size_rounded}, qopts)
            : dq;
    } else {
        dq_k = torch::empty({batch_size, seqlen_q, num_heads,
                             need_pad ? head_size_rounded : head_size}, qopts);
        dq = need_pad ? dq_k.narrow(-1, 0, head_size) : dq_k;
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_type, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
        dk_k = need_pad
            ? torch::empty({batch_size, seqlen_k, num_heads_k, head_size_rounded}, qopts)
            : dk;
    } else {
        dk_k = torch::empty({batch_size, seqlen_k, num_heads_k,
                             need_pad ? head_size_rounded : head_size}, qopts);
        dk = need_pad ? dk_k.narrow(-1, 0, head_size) : dk_k;
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_type, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size_v);
        dv_k = need_pad
            ? torch::empty({batch_size, seqlen_k, num_heads_k, head_size_rounded}, qopts)
            : dv;
    } else {
        dv_k = torch::empty({batch_size, seqlen_k, num_heads_k,
                             need_pad ? head_size_rounded : head_size_v}, qopts);
        dv = need_pad ? dv_k.narrow(-1, 0, head_size_v) : dv_k;
    }

    // Otherwise the kernel will be launched from device 0
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    // Scratch buffers for the 3-kernel bwd pipeline.
    at::Tensor softmax_d = torch::zeros({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor softmax_lse_log2 = torch::zeros({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum = torch::zeros({batch_size, num_heads, seqlen_q_rounded * head_size_rounded}, opts.dtype(at::kFloat));
    // dK/dV always go through the fp32 accum + atomicAdd path (even when
    // h == h_k) because Q-overlapping slices (RangeMerge) can also race, so
    // the accums must be zero-initialized in ALL cases.
    at::Tensor dk_accum = torch::zeros({batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}, opts.dtype(at::kFloat));
    at::Tensor dv_accum = torch::zeros({batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}, opts.dtype(at::kFloat));

    // Populate FlexFlashAttentionBwdParams
    FlexFlashAttentionBwdParams params;
    memset(&params, 0, sizeof(params));

    // Basic parameters
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.d = head_size_rounded;
    params.dv = head_size_rounded;
    params.d_rounded = head_size_rounded;
    params.dv_rounded = head_size_rounded;
    params.is_bf16 = is_bf16;
    params.scale_softmax = softmax_scale;
    params.softcap = softcap;

    // Feature flags (runtime dispatch in the launch template).
    params.inner_min_to_max = inner_min_to_max;
    params.persistent_scheduler = persistent_scheduler;
    params.use_loop_k = use_loop_k;
    params.reduce_kv = reduce_kv;

    // Q / K / V pointers
    params.q_ptr = q_use.data_ptr();
    params.q_row_stride = q_use.stride(1);
    params.q_head_stride = q_use.stride(2);
    params.q_batch_stride = q_use.stride(0);

    params.k_ptr = k_use.data_ptr();
    params.k_row_stride = k_use.stride(1);
    params.k_head_stride = k_use.stride(2);
    params.k_batch_stride = k_use.stride(0);

    params.v_ptr = v_use.data_ptr();
    params.v_row_stride = v_use.stride(1);
    params.v_head_stride = v_use.stride(2);
    params.v_batch_stride = v_use.stride(0);
    params.v_dim_stride = v_use.stride(3);

    // O (fwd output, needed for delta = rowsum(dO*O)) and LSE
    params.o_ptr = out_use.data_ptr();
    params.o_row_stride = out_use.stride(1);
    params.o_head_stride = out_use.stride(2);
    params.o_batch_stride = out_use.stride(0);
    params.softmax_lse_ptr = softmax_lse.data_ptr<float>();

    // dO / dQKV
    params.do_ptr = dout_use.data_ptr();
    params.do_row_stride = dout_use.stride(1);
    params.do_head_stride = dout_use.stride(2);
    params.do_batch_stride = dout_use.stride(0);

    params.dq_ptr = dq_k.data_ptr();
    params.dq_row_stride = dq_k.stride(1);
    params.dq_head_stride = dq_k.stride(2);
    params.dq_batch_stride = dq_k.stride(0);

    params.dk_ptr = dk_k.data_ptr();
    params.dk_row_stride = dk_k.stride(1);
    params.dk_head_stride = dk_k.stride(2);
    params.dk_batch_stride = dk_k.stride(0);

    params.dv_ptr = dv_k.data_ptr();
    params.dv_row_stride = dv_k.stride(1);
    params.dv_head_stride = dv_k.stride(2);
    params.dv_batch_stride = dv_k.stride(0);

    // fp32 accumulators & scratch
    params.dq_accum_ptr = dq_accum.data_ptr();
    params.dk_accum_ptr = dk_accum.data_ptr();
    params.dv_accum_ptr = dv_accum.data_ptr();
    params.dsoftmax_sum = softmax_d.data_ptr();
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();

    // Architecture
    params.arch = dprops->major == 8 ? (dprops->minor == 9 ? 89 : 80) : dprops->major * 10 + dprops->minor;
    params.num_sm = dprops->multiProcessorCount;

    // No causal/local/split for flex flash attention
    params.is_causal = false;
    params.is_local = false;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.num_splits = 1;
    params.num_splits_dynamic_ptr = nullptr;
    params.deterministic = deterministic;

    // Dropout replay state (see validation above).  Eager: rng_state points
    // at the caller-owned CPU tensor holding the (seed, offset) pair from
    // fwd, read by value in the launch template.  Captured fwd graph: the
    // pair lives in a CUDA tensor published by the fwd kernel, and bwd must
    // read it on-device at each replay (host dereference would also defeat
    // per-replay RNG refresh) — hand its two elements to the kernel as
    // philox pointers.
    if (p_dropout > 0.f) {
        params.p_dropout = 1.f - p_dropout;
        params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
        params.rp_dropout = 1.f / params.p_dropout;
        if (rng_state_.value().is_cuda()) {
            auto* pair = reinterpret_cast<unsigned long long*>(rng_state_.value().data_ptr());
            params.rng_state = nullptr;
            params.philox_seed_ptr = pair;        // [0] seed
            params.philox_offset_ptr = pair + 1;  // [1] offset base
            params.intragraph_offset = 0;
        } else {
            params.rng_state = reinterpret_cast<uint64_t*>(rng_state_.value().data_ptr());
        }
    } else {
        params.p_dropout = 1.f;
        params.p_dropout_in_uint8_t = 255;
        params.rp_dropout = 1.f;
        params.rng_state = nullptr;
    }

    // Deterministic mode: zero-initialized semaphore turnstiles.
    // Over-allocated with the smallest tile granularity (64 rows per block) —
    // indices are per (block, batch, head) and block counts only shrink with
    // larger tiles, so this is always large enough.
    at::Tensor dq_semaphore, dk_semaphore, dv_semaphore;
    if (deterministic) {
        int const num_m_blocks_max = (seqlen_q + 63) / 64;
        int const num_n_blocks_max = (seqlen_k + 63) / 64;
        dq_semaphore = torch::zeros({num_m_blocks_max * batch_size * num_heads}, opts.dtype(at::kInt));
        dk_semaphore = torch::zeros({num_n_blocks_max * batch_size * num_heads_k}, opts.dtype(at::kInt));
        dv_semaphore = torch::zeros({num_n_blocks_max * batch_size * num_heads_k}, opts.dtype(at::kInt));
        params.dq_semaphore = dq_semaphore.data_ptr<int>();
        params.dk_semaphore = dk_semaphore.data_ptr<int>();
        params.dv_semaphore = dv_semaphore.data_ptr<int>();
    } else {
        params.dq_semaphore = nullptr;
        params.dk_semaphore = nullptr;
        params.dv_semaphore = nullptr;
    }

    // Dynamic persistent scheduler: zero-initialized tile counter.
    at::Tensor tile_count_semaphore;
    if (persistent_scheduler) {
        tile_count_semaphore = torch::zeros({1}, opts.dtype(at::kInt));
        params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
    } else {
        params.tile_count_semaphore = nullptr;
    }

    // Flex flash attention slice parameters
    params.is_flex_flash_attention = true;
    params.slices.q_starts = q_starts.data_ptr<int>();
    params.slices.q_ends = q_ends.data_ptr<int>();
    params.slices.k_starts = k_starts.data_ptr<int>();
    params.slices.k_ends = k_ends.data_ptr<int>();
    params.slices.mask_types = mask_types.data_ptr<int>();
    params.slices.row_to_slice = row_to_slice.data_ptr<int>();
    params.slices.diagonal_offsets = diagonal_offsets.data_ptr<int>();
    params.slices.band_widths = band_widths.data_ptr<int>();
    params.slices.num_slices = num_slices;

    // RangeMerge layout (nullptr → legacy single-slice dispatch in kernel).
    params.slices.vbatch_to_slice = has_merge_layout
        ? vbatch_to_slice_.value().data_ptr<int>() : nullptr;
    params.slices.row_to_vbatch_start = has_merge_layout
        ? row_to_vbatch_start_.value().data_ptr<int>() : nullptr;
    params.slices.row_to_vbatch_end = has_merge_layout
        ? row_to_vbatch_end_.value().data_ptr<int>() : nullptr;
    params.slices.num_vbatches = num_vbatches;

    // Bitmask fallback (nullptr for the four algebraic mask types).
    params.slices.mask_bits = mask_bits_ptr;
    params.slices.mask_row_stride = mask_row_stride;

    // P3 per-(b,h) heterogeneous layout (nullptr → single shared layout).
    params.slices.bh_to_group = has_bh_layout
        ? bh_to_group_.value().data_ptr<int>() : nullptr;
    params.slices.group_slice_offsets = has_bh_layout
        ? group_slice_offsets_.value().data_ptr<int>() : nullptr;
    params.slices.group_vb_offsets = has_bh_layout
        ? group_vb_offsets_.value().data_ptr<int>() : nullptr;
    params.slices.num_groups = num_groups;

    hggcStream_t stream = (hggcStream_t)at::cuda::getCurrentCUDAStream().stream();
    if (fp32_path) {
        run_mha_flex_flash_bwd_f32(params, stream);
    } else {
        run_mha_flex_flash_bwd(params, stream);
    }

    if (need_pad) {
        if (dq_.has_value()) { dq.copy_(dq_k.narrow(-1, 0, head_size)); }
        if (dk_.has_value()) { dk.copy_(dk_k.narrow(-1, 0, head_size)); }
        if (dv_.has_value()) { dv.copy_(dv_k.narrow(-1, 0, head_size_v)); }
    }
    return std::make_tuple(dq, dk, dv);
}

// fp16/bf16 bwd entry — unchanged contract.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_bwd(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
    at::Tensor softmax_lse, at::Tensor dout,
    at::Tensor q_starts, at::Tensor q_ends, at::Tensor k_starts, at::Tensor k_ends,
    at::Tensor mask_types, at::Tensor row_to_slice, at::Tensor diagonal_offsets,
    at::Tensor band_widths,
    std::optional<at::Tensor> vbatch_to_slice_,
    std::optional<at::Tensor> row_to_vbatch_start_,
    std::optional<at::Tensor> row_to_vbatch_end_,
    std::optional<at::Tensor> dq_,
    std::optional<at::Tensor> dk_,
    std::optional<at::Tensor> dv_,
    std::optional<double> softmax_scale_,
    bool deterministic,
    std::optional<double> softcap_,
    bool reduce_kv,
    bool use_loop_k,
    bool inner_min_to_max,
    bool persistent_scheduler,
    double p_dropout_,
    std::optional<at::Tensor> rng_state_,
    std::optional<at::Tensor> mask_bits_,
    std::optional<at::Tensor> bh_to_group_,
    std::optional<at::Tensor> group_slice_offsets_,
    std::optional<at::Tensor> group_vb_offsets_
) {
    return mha_flex_flash_bwd_impl(
        q, k, v, out, softmax_lse, dout, q_starts, q_ends, k_starts, k_ends,
        mask_types, row_to_slice, diagonal_offsets, band_widths,
        vbatch_to_slice_, row_to_vbatch_start_, row_to_vbatch_end_,
        dq_, dk_, dv_, softmax_scale_, deterministic, softcap_, reduce_kv,
        use_loop_k, inner_min_to_max, persistent_scheduler, p_dropout_,
        rng_state_, mask_bits_, bh_to_group_, group_slice_offsets_,
        group_vb_offsets_, false);
}

// ISOLATED fp32 bwd entry — requires float32 inputs, runs the TF32
// tensor-core kernel set (flex_flash_attention_bwd_sm89_hdim*_f32 instantiations).
std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_bwd_f32(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out,
    at::Tensor softmax_lse, at::Tensor dout,
    at::Tensor q_starts, at::Tensor q_ends, at::Tensor k_starts, at::Tensor k_ends,
    at::Tensor mask_types, at::Tensor row_to_slice, at::Tensor diagonal_offsets,
    at::Tensor band_widths,
    std::optional<at::Tensor> vbatch_to_slice_,
    std::optional<at::Tensor> row_to_vbatch_start_,
    std::optional<at::Tensor> row_to_vbatch_end_,
    std::optional<at::Tensor> dq_,
    std::optional<at::Tensor> dk_,
    std::optional<at::Tensor> dv_,
    std::optional<double> softmax_scale_,
    bool deterministic,
    std::optional<double> softcap_,
    bool reduce_kv,
    bool use_loop_k,
    bool inner_min_to_max,
    bool persistent_scheduler,
    double p_dropout_,
    std::optional<at::Tensor> rng_state_,
    std::optional<at::Tensor> mask_bits_,
    std::optional<at::Tensor> bh_to_group_,
    std::optional<at::Tensor> group_slice_offsets_,
    std::optional<at::Tensor> group_vb_offsets_
) {
    return mha_flex_flash_bwd_impl(
        q, k, v, out, softmax_lse, dout, q_starts, q_ends, k_starts, k_ends,
        mask_types, row_to_slice, diagonal_offsets, band_widths,
        vbatch_to_slice_, row_to_vbatch_start_, row_to_vbatch_end_,
        dq_, dk_, dv_, softmax_scale_, deterministic, softcap_, reduce_kv,
        use_loop_k, inner_min_to_max, persistent_scheduler, p_dropout_,
        rng_state_, mask_bits_, bh_to_group_, group_slice_offsets_,
        group_vb_offsets_, true);
}

// ── Fused mask-decomposition ops (kernels in mask_decomp_kernels.cu) ─────
// Hand-written CUDA replacements for the torch.compile'd sweeps in
// mask_decomp.py (row stats + first-interval peel): ~2.7x faster at
// s=25286 and dynamo-free, a prerequisite for a C++-callable
// decomposition op (SDPA backend port).  Semantics live in that TU's
// header comment; they are 1:1 with the Python reference.
void launch_decomp_row_stats(unsigned char const* mask, int seqlen_q,
                             int seqlen_k, int* ks, int* ke,
                             unsigned char* hh, hggcStream_t stream);
void launch_decomp_peel_first_interval(unsigned char* blk, int seqlen_q,
                                       int seqlen_k, int* ks, int* ke,
                                       hggcStream_t stream);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_decomp_row_stats(at::Tensor const& mask) {
    TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool && mask.dim() == 2 &&
                mask.is_contiguous() && mask.is_cuda(),
        "flex_flash_attention::row_stats expects a contiguous CUDA bool "
        "[seqlen_q, seqlen_k] mask");
    int const sq = mask.size(0), sk = mask.size(1);
    auto ks = at::empty({sq}, mask.options().dtype(at::kInt));
    auto ke = at::empty({sq}, mask.options().dtype(at::kInt));
    auto hh = at::empty({sq}, mask.options().dtype(at::kBool));
    hggcStream_t stream = (hggcStream_t)at::cuda::getCurrentCUDAStream().stream();
    launch_decomp_row_stats(
        reinterpret_cast<unsigned char const*>(mask.data_ptr<bool>()), sq, sk,
        ks.data_ptr<int>(), ke.data_ptr<int>(),
        reinterpret_cast<unsigned char*>(hh.data_ptr<bool>()), stream);
    return {ks, ke, hh};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_decomp_peel_first_interval(at::Tensor blk) {
    TORCH_CHECK(blk.scalar_type() == at::ScalarType::Bool && blk.dim() == 2 &&
                blk.is_contiguous() && blk.is_cuda(),
        "flex_flash_attention::peel_first_interval expects a contiguous CUDA bool "
        "[seqlen_q, seqlen_k] residual (mutated in place)");
    int const sq = blk.size(0), sk = blk.size(1);
    auto ks = at::empty({sq}, blk.options().dtype(at::kInt));
    auto ke = at::empty({sq}, blk.options().dtype(at::kInt));
    hggcStream_t stream = (hggcStream_t)at::cuda::getCurrentCUDAStream().stream();
    launch_decomp_peel_first_interval(
        reinterpret_cast<unsigned char*>(blk.data_ptr<bool>()), sq, sk,
        ks.data_ptr<int>(), ke.data_ptr<int>(), stream);
    return {ks, ke, blk};
}

// ── Whole-mask decomposition op (layer-2 pure-C++ pipeline) ──────────────
// decompose_mask(mask, kblock_m) runs the exact device-side equivalent of
// mask_decomp.py decompose_mask_optimized(mask, kblock_m):
//   row_stats → (16x) peel_flags/gate → segment(layer) → classify(layer)
//   → finalize (layer-cap supported check) → guard → merge/align/vbatch
//   layout → desc pack.  No host round trip: layer gating and the
// supported verdict live in device flags, so the op is torch.compile
// graph-capturable.
//
// desc pack (the AttnSliceParams layout):
//   slices  int32[7, CAP]  q_starts/q_ends/k_starts/k_ends/mask_types/
//                          diagonal_offsets/band_widths (aligned frags)
//   rows    int32[3, sq]   row_to_slice / row_to_vbatch_start / end
//   vbatch  int32[CAP]     vbatch_to_slice
//   counters int32[4]      [num_slices, num_vbatches, supported, n_layers]
// CAP = sq * 17 + 64; when the accumulated slice count or the layer count
// exceeds the caps, counters[2] is 0 (strategy A: honest "not supported",
// never silent truncation).
constexpr int kMaxHoleLayers = 16;   // mask_decomp.py _MAX_HOLE_LAYERS

void launch_decomp_peel_flags(unsigned char* blk, int seqlen_q,
                              int seqlen_k, int* ks, int* ke,
                              int* nonempty_p, int* leftover_p,
                              int const* done_p, hggcStream_t stream);
void launch_decomp_gate_done(int* nonempty_p, int* leftover_p,
                             int* done_p, hggcStream_t stream);
void launch_decomp_finalize(int const* gate_cap_p, int* supported_p,
                            int* counters_sup_p, int* n_layers_p,
                            int n_layers, hggcStream_t stream);
void launch_decomp_guard_zero(int const* supported_p, int* n_pre_p,
                              hggcStream_t stream);
size_t decomp_segment_scratch_size(int seqlen_q);
hggcError_t launch_decomp_segment(int const* ks, int const* ke,
                                  int seqlen_q, int seqlen_k, int* b,
                                  int* n_segs_out, int* seg_q_start,
                                  int* seg_q_end, int* seg_ks0,
                                  int* seg_ke0, int* seg_dks, int* seg_dke,
                                  int const* done, void* scratch,
                                  size_t scratch_bytes, hggcStream_t stream);
size_t decomp_classify_scratch_size(int seqlen_q);
hggcError_t launch_decomp_classify(
    int const* seg_q_start, int const* seg_q_end, int const* seg_ks0,
    int const* seg_ke0, int const* seg_dks, int const* seg_dke,
    int const* n_segs_p, int const* ks, int const* ke, int layer,
    int seqlen_q, int* n_slices_p, int* q_starts, int* q_ends,
    int* k_starts, int* k_ends, int* mask_types, int* diag_offsets,
    int* band_widths, int cap, int* supported_p, int const* done,
    void* scratch, size_t scratch_bytes, hggcStream_t stream);
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_decompose_mask_impl(at::Tensor const& mask, int64_t kblock_m);

// Identity-level desc cache: within one training iteration fwd and bwd
// decompose the SAME mask object; keying on (TensorImpl*, version_counter,
// kblock_m) makes a hit provably the same mask content (version bumps on
// every in-place mutation).  Never fingerprint-sampled — a false hit would
// silently corrupt the layout.  The entry holds a STRONG reference to the
// mask: that pins its TensorImpl, so the impl address cannot be recycled
// by a brand-new tensor and ABA-match an old key (a bare impl-pointer key
// was observed to serve a stale desc after the allocator reused the
// freed TensorImpl slot at version 0).
struct DecompCacheKey {
    void const* impl;
    uint32_t version;
    int64_t kblock_m;
    bool operator==(DecompCacheKey const& o) const {
        return impl == o.impl && version == o.version &&
               kblock_m == o.kblock_m;
    }
};
struct DecompCacheKeyHash {
    size_t operator()(DecompCacheKey const& k) const {
        return std::hash<void const*>()(k.impl) ^
               (std::hash<uint32_t>()(k.version) << 1) ^
               (std::hash<int64_t>()(k.kblock_m) << 2);
    }
};
struct DecompCacheEntry {
    at::Tensor mask;  // pins the TensorImpl (see note above)
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> desc;
};
static std::mutex g_decomp_cache_mtx;
// Heap-allocated, never destroyed: entries hold CUDA tensors, and a static
// destructor running after CUDA context teardown crashes while freeing
// them (static-destruction-order fiasco).
static std::unordered_map<DecompCacheKey, DecompCacheEntry,
                          DecompCacheKeyHash>&
    g_decomp_cache = *new std::unordered_map<DecompCacheKey, DecompCacheEntry,
                                             DecompCacheKeyHash>();
constexpr size_t kDecompCacheMax = 4;  // descs are tens of MB at S=25k

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_decompose_mask(at::Tensor const& mask, int64_t kblock_m) {
    TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool && mask.dim() == 2,
                "flex_flash_attention::decompose_mask expects a bool "
                "[seqlen_q, seqlen_k] mask");
    TORCH_CHECK(kblock_m > 0,
                "flex_flash_attention::decompose_mask: kblock_m must be > 0");
    if (mask.is_cuda() && mask.is_contiguous()) {
        // Inference tensors have NO version counter (reading it raises) and
        // can never be mutated in place, so a constant version key is
        // sound; this keeps torch.inference_mode() workloads usable.
        at::TensorImpl* impl = mask.unsafeGetTensorImpl();
        DecompCacheKey const key{impl,
                                 impl->is_inference()
                                     ? uint32_t{0}
                                     : impl->version_counter()
                                           .current_version(),
                                 kblock_m};
        {
            std::lock_guard<std::mutex> lk(g_decomp_cache_mtx);
            auto const it = g_decomp_cache.find(key);
            if (it != g_decomp_cache.end()) return it->second.desc;
        }
        auto desc = mha_flex_flash_decompose_mask_impl(mask, kblock_m);
        std::lock_guard<std::mutex> lk(g_decomp_cache_mtx);
        if (g_decomp_cache.size() >= kDecompCacheMax) g_decomp_cache.clear();
        g_decomp_cache.emplace(key, DecompCacheEntry{mask, desc});
        return desc;
    }
    // Non-contiguous / meta: no cache (contiguousify would break identity).
    return mha_flex_flash_decompose_mask_impl(mask, kblock_m);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_decompose_mask_impl(at::Tensor const& mask_in,
                                  int64_t kblock_m) {
    at::Tensor mask = mask_in.is_contiguous() ? mask_in : mask_in.contiguous();
    int const sq = mask.size(0), sk = mask.size(1);
    auto const iopt = mask.options().dtype(at::kInt);
    // Cascaded device-side scan grids: nb = ceil(sq/256) <= 256*256.
    // Checked before any big allocation so oversized masks are rejected
    // cheaply (cap below scales with sq and would allocate GiBs first).
    TORCH_CHECK(((int64_t)sq + 255) / 256 <= 65536,
                "flex_flash_attention::decompose_mask: seqlen_q > 16777216 is not "
                "supported by the device-side scan grids");
    int64_t const cap = (int64_t)sq * 17 + 64;

    // Zero-initialized on purpose: compile-friendly callers feed the FULL
    // [7,CAP]/[CAP] tensors to fwd/bwd (no data-dependent slicing), and the
    // tail beyond counters[0] must be inert — all-zero fields describe a
    // zero-width FULL slice ([0,0) Q range), which skip_to_first_valid()
    // rejects, so it contributes nothing.
    auto slices = at::zeros({7, cap}, iopt);
    auto rows = at::empty({3, (int64_t)sq}, iopt);
    auto vbatch = at::zeros({cap}, iopt);
    at::Tensor counters;

    if (sq == 0 || sk == 0) {
        // Empty mask: zero slices, trivially supported (Python returns []).
        counters = at::tensor({0, 0, 1, 0}, at::kInt).to(mask.device());
        return {slices, rows, vbatch, counters};
    }
    counters = at::empty({4}, iopt);

    hggcStream_t stream = (hggcStream_t)at::cuda::getCurrentCUDAStream().stream();
    auto const bopt = iopt.dtype(at::kByte);

    // Working state.
    auto residual = mask.clone();
    auto ks = at::empty({sq}, iopt), ke = at::empty({sq}, iopt);
    // flags[0..2] = nonempty / leftover / supported (global).
    // flags[3 + l] = per-layer gate: set when the residual emptied AFTER
    // layer l's classify ran, so layer l itself is never gated (the
    // all-empty-mask case must still emit its zero-K FULL slice, matching
    // the Python single-layer path) while every later stage short-circuits.
    auto flags = at::zeros({3 + kMaxHoleLayers + 1}, iopt);
    flags[2] = 1;  // supported starts true; cap/leftover checks only clear
    flags[3] = 0;  // gate_0 = false (layer 0 always runs)
    int* nonempty_p = flags.data_ptr<int>();
    int* leftover_p = nonempty_p + 1;
    int* supported_p = leftover_p + 1;
    int* gate_p = supported_p + 1;  // [kMaxHoleLayers + 1]

    // Per-layer intermediates (reused across layers; stream-ordered).
    auto b = at::empty({sq}, iopt);
    auto n_segs = at::empty({1}, iopt);
    auto seg = at::empty({6, (int64_t)sq}, iopt);
    auto pre = at::empty({7, cap}, iopt);
    auto n_pre = at::zeros({1}, iopt);
    auto merged = at::empty({7, cap}, iopt);
    auto n_merged = at::empty({1}, iopt);

    // The segment launcher's scratch formula is sized for >= 2 rows (its
    // scan seeds a sentinel pair), so floor the allocation at sq == 1.
    size_t const seg_sb = decomp_segment_scratch_size(std::max(sq, 2));
    size_t const cls_sb = decomp_classify_scratch_size(sq);
    size_t const lay_sb = decomp_layout_scratch_size(sq, (int)kblock_m);
    auto scratch = at::empty(
        {(int64_t)std::max(seg_sb, std::max(cls_sb, lay_sb))}, bopt);

    unsigned char* resid_p = reinterpret_cast<unsigned char*>(
        residual.data_ptr<bool>());
    int* seg_p[6];
    for (int i = 0; i < 6; ++i) seg_p[i] = seg[i].data_ptr<int>();
    int* pre_p[7];
    for (int i = 0; i < 7; ++i) pre_p[i] = pre[i].data_ptr<int>();
    int* mrg_p[7];
    for (int i = 0; i < 7; ++i) mrg_p[i] = merged[i].data_ptr<int>();
    int* slc_p[7];
    for (int i = 0; i < 7; ++i) slc_p[i] = slices[i].data_ptr<int>();
    int* row_p[3];
    for (int i = 0; i < 3; ++i) row_p[i] = rows[i].data_ptr<int>();

    // Layer 0's peel doubles as row_stats: for a hole-free row the first
    // contiguous True interval IS the [first, last+1) bound pair, so the
    // layered pipeline subsumes the single-layer path (Python picks between
    // them via hh.any(); here both collapse into one).

    // 16 statically-unrolled layers; every gate is device-side.  Layer l's
    // segment/classify runs on layer l's OWN peel result — gated only when
    // the residual was already empty BEFORE layer l (gate_p[l]) — so an
    // all-empty mask still emits its layer-0 zero-K FULL slice (Python
    // single-layer path contract).  Layer 0 passes a null gate since
    // gate_p[0] is false by construction.  gate_p[l+1], written right
    // after layer l's peel, gates layer l+1 onwards (peel and classify),
    // and gate_p[16] doubles as the layer-cap verdict for finalize.
    for (int layer = 0; layer < kMaxHoleLayers; ++layer) {
        int* gate_l = gate_p + layer;      // gates this layer's peel
        int* gate_s = gate_p + layer + 1;  // gates the NEXT layer
        int* const gate_cls = layer == 0 ? nullptr : gate_s;
        launch_decomp_peel_flags(resid_p, sq, sk, ks.data_ptr<int>(),
                                 ke.data_ptr<int>(), nonempty_p, leftover_p,
                                 gate_l, stream);
        launch_decomp_gate_done(nonempty_p, leftover_p, gate_s, stream);
        hggcError_t err = launch_decomp_segment(
            ks.data_ptr<int>(), ke.data_ptr<int>(), sq, sk,
            b.data_ptr<int>(), n_segs.data_ptr<int>(), seg_p[0], seg_p[1],
            seg_p[2], seg_p[3], seg_p[4], seg_p[5], gate_cls,
            scratch.data_ptr(), seg_sb, stream);
        TORCH_CHECK(err == hggcSuccess, "decompose_mask segment: ",
                    hggcGetErrorString(err));
        err = launch_decomp_classify(
            seg_p[0], seg_p[1], seg_p[2], seg_p[3], seg_p[4], seg_p[5],
            n_segs.data_ptr<int>(), ks.data_ptr<int>(), ke.data_ptr<int>(),
            layer, sq, n_pre.data_ptr<int>(), pre_p[0], pre_p[1], pre_p[2],
            pre_p[3], pre_p[4], pre_p[5], pre_p[6], (int)cap, supported_p,
            gate_cls, scratch.data_ptr(), cls_sb, stream);
        TORCH_CHECK(err == hggcSuccess, "decompose_mask classify: ",
                    hggcGetErrorString(err));
    }
    // Layer-cap verdict (gate[16]: some peel emptied the residual within
    // the cap), then publish supported into counters[2] and pack the desc.
    launch_decomp_finalize(gate_p + kMaxHoleLayers, supported_p,
                           counters.data_ptr<int>() + 2,
                           counters.data_ptr<int>() + 3, kMaxHoleLayers,
                           stream);
    launch_decomp_guard_zero(supported_p, n_pre.data_ptr<int>(), stream);
    hggcError_t err = launch_decomp_layout(
        pre_p[0], pre_p[1], pre_p[2], pre_p[3], pre_p[4], pre_p[5],
        pre_p[6], n_pre.data_ptr<int>(), sq, (int)kblock_m, (int)cap,
        mrg_p[0], mrg_p[1], mrg_p[2], mrg_p[3], mrg_p[4], mrg_p[5],
        mrg_p[6], n_merged.data_ptr<int>(), slc_p[0], slc_p[1], slc_p[2],
        slc_p[3], slc_p[4], slc_p[5], slc_p[6], counters.data_ptr<int>(),
        vbatch.data_ptr<int>(), row_p[0], row_p[1], row_p[2],
        counters.data_ptr<int>() + 1, supported_p, scratch.data_ptr(),
        lay_sb, stream);
    TORCH_CHECK(err == hggcSuccess, "decompose_mask layout: ",
                hggcGetErrorString(err));
    return {slices, rows, vbatch, counters};
}

// Meta (fake) impl: shape-only, for torch.compile graph tracing.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_decompose_mask_meta(at::Tensor const& mask, int64_t kblock_m) {
    int64_t const sq = mask.size(0);
    int64_t const cap = sq * 17 + 64;
    auto const iopt = mask.options().dtype(at::kInt);
    return {at::empty({7, cap}, iopt), at::empty({3, sq}, iopt),
            at::empty({cap}, iopt), at::empty({4}, iopt)};
}

// CUDA-graph variant of decompose_mask: bypasses the identity cache so the
// WHOLE device pipeline is recorded into the graph and re-executes on every
// replay — replays therefore track in-place mask content changes (the
// replay path runs no host code, so a cached decomposition would freeze
// stale slice geometry).  Callers feed the full-capacity [7,CAP]/[CAP] desc
// tensors straight to fwd/bwd; the zero tail beyond counters[0] is inert
// (zero-width FULL slices rejected by skip_to_first_valid).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_decompose_mask_nocache(at::Tensor const& mask,
                                     int64_t kblock_m) {
    TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool && mask.dim() == 2,
                "flex_flash_attention::decompose_mask_nocache expects a bool "
                "[seqlen_q, seqlen_k] mask");
    TORCH_CHECK(kblock_m > 0,
                "flex_flash_attention::decompose_mask_nocache: kblock_m must be > 0");
    return mha_flex_flash_decompose_mask_impl(mask, kblock_m);
}

// ── Shape inference (Meta) for the attention ops ──────────────────────
// Mirrors mha_flex_flash_fwd_impl's output rules without launching anything:
// out is (b, sq, h, head_size_v) in q's dtype (headdim padding stays an
// internal concern), lse is (b, h, sq) fp32, rng_state is CPU int64[2].
// rng_state is built on the META device (FakeTensorMode requires every
// meta-kernel output to physically live on `meta`); the empty dispatch key
// set makes FakeTensor expose it as a CPU tensor, matching the eager impl.
static std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_fwd_meta_common(at::Tensor const& q, at::Tensor const& v) {
    auto const b = q.size(0), sq = q.size(1), h = q.size(2);
    auto out = at::empty({b, sq, h, v.size(3)}, q.options());
    auto lse = at::empty({b, h, sq}, q.options().dtype(at::kFloat));
    auto rng = at::Tensor(at::detail::empty_meta({2}, at::kLong));
    return {out, lse, rng};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_fwd_meta(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<double>, std::optional<double>, bool, bool,
    std::optional<bool>, bool, double,
    std::optional<at::Generator>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, int64_t, int64_t) {
    return mha_flex_flash_fwd_meta_common(q, v);
}

// bwd returns gradients shaped exactly like q / k / v (headdim padding is
// undone inside the impl, same as fwd).
static std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_bwd_meta_common(at::Tensor const& q, at::Tensor const& k,
                              at::Tensor const& v) {
    return {at::empty(q.sizes(), q.options()),
            at::empty(k.sizes(), k.options()),
            at::empty(v.sizes(), v.options())};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_bwd_meta(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<double>, bool, std::optional<double>, bool, bool, bool,
    bool, double, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>) {
    return mha_flex_flash_bwd_meta_common(q, k, v);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_fwd_f32_meta(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<double>, std::optional<double>, bool, bool,
    std::optional<bool>, bool, double,
    std::optional<at::Generator>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, int64_t, int64_t) {
    return mha_flex_flash_fwd_meta_common(q, v);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mha_flex_flash_bwd_f32_meta(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<double>, bool, std::optional<double>, bool, bool, bool,
    bool, double, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>) {
    return mha_flex_flash_bwd_meta_common(q, k, v);
}

// NOTE on autograd: this torch build ships a STABLE-ABI (abi3) wheel
// whose public headers do not expose torch::autograd::wrap_kernel_functor,
// so a dispatcher-level (C++) Autograd registration is not possible here.
// Differentiability is provided by the Python-side torch.autograd.Function
// in flex_flash_attention/interface.py wrapping these exact ops (same design as
// upstream FA3); torch.compile handles it via Compiled Autograd, and the
// Meta impls above keep the fwd/bwd ops traceable inside the graph.

// Defined in sdpa_glue.cpp (post-replay envelope check for CUDA-graph
// captures): raises when the most recent replay decomposed a mask outside
// the layered-interval envelope.
namespace flex_flash_attention {
void sdpa_graph_mask_check();
} // namespace flex_flash_attention

TORCH_LIBRARY(flex_flash_attention, m) {
    m.def("fwd("
        "Tensor q,"
        "Tensor k,"
        "Tensor v,"
        "Tensor q_starts,"
        "Tensor q_ends,"
        "Tensor k_starts,"
        "Tensor k_ends,"
        "Tensor mask_types,"
        "Tensor row_to_slice,"
        "Tensor diagonal_offsets,"
        "Tensor band_widths,"
        "Tensor? vbatch_to_slice = None,"
        "Tensor? row_to_vbatch_start = None,"
        "Tensor? row_to_vbatch_end = None,"
        "Tensor? out = None,"
        "float? softmax_scale = None,"
        "float? softcap = None,"
        "bool inner_min_to_max=False,"
        "bool persistent_scheduler=False,"
        "bool? use_kblockm128=None,"
        "bool kblockn64=False,"
        "float dropout_p=0.0,"
        "Generator? gen=None,"
        "Tensor? mask_bits=None,"
        "Tensor? bh_to_group=None,"
        "Tensor? group_slice_offsets=None,"
        "Tensor? group_vb_offsets=None,"
        "int trivial_mask=0,"
        "int trivial_diagonal=0) -> (Tensor, Tensor, Tensor)");
    m.def("bwd("
        "Tensor q,"
        "Tensor k,"
        "Tensor v,"
        "Tensor out,"
        "Tensor softmax_lse,"
        "Tensor dout,"
        "Tensor q_starts,"
        "Tensor q_ends,"
        "Tensor k_starts,"
        "Tensor k_ends,"
        "Tensor mask_types,"
        "Tensor row_to_slice,"
        "Tensor diagonal_offsets,"
        "Tensor band_widths,"
        "Tensor? vbatch_to_slice = None,"
        "Tensor? row_to_vbatch_start = None,"
        "Tensor? row_to_vbatch_end = None,"
        "Tensor? dq = None,"
        "Tensor? dk = None,"
        "Tensor? dv = None,"
        "float? softmax_scale = None,"
        "bool deterministic=False,"
        "float? softcap = None,"
        "bool reduce_kv=False,"
        "bool use_loop_k=False,"
        "bool inner_min_to_max=False,"
        "bool persistent_scheduler=False,"
        "float dropout_p=0.0,"
        "Tensor? rng_state=None,"
        "Tensor? mask_bits=None,"
        "Tensor? bh_to_group=None,"
        "Tensor? group_slice_offsets=None,"
        "Tensor? group_vb_offsets=None) -> (Tensor, Tensor, Tensor)");
    // fp32 kernel set: identical schemas to fwd/bwd, independent ops so the
    // fp16/bf16 entries keep their exact dispatch table.
    m.def("fwd_f32("
        "Tensor q,"
        "Tensor k,"
        "Tensor v,"
        "Tensor q_starts,"
        "Tensor q_ends,"
        "Tensor k_starts,"
        "Tensor k_ends,"
        "Tensor mask_types,"
        "Tensor row_to_slice,"
        "Tensor diagonal_offsets,"
        "Tensor band_widths,"
        "Tensor? vbatch_to_slice = None,"
        "Tensor? row_to_vbatch_start = None,"
        "Tensor? row_to_vbatch_end = None,"
        "Tensor? out = None,"
        "float? softmax_scale = None,"
        "float? softcap = None,"
        "bool inner_min_to_max=False,"
        "bool persistent_scheduler=False,"
        "bool? use_kblockm128=None,"
        "bool kblockn64=False,"
        "float dropout_p=0.0,"
        "Generator? gen=None,"
        "Tensor? mask_bits=None,"
        "Tensor? bh_to_group=None,"
        "Tensor? group_slice_offsets=None,"
        "Tensor? group_vb_offsets=None,"
        "int trivial_mask=0,"
        "int trivial_diagonal=0) -> (Tensor, Tensor, Tensor)");
    m.def("bwd_f32("
        "Tensor q,"
        "Tensor k,"
        "Tensor v,"
        "Tensor out,"
        "Tensor softmax_lse,"
        "Tensor dout,"
        "Tensor q_starts,"
        "Tensor q_ends,"
        "Tensor k_starts,"
        "Tensor k_ends,"
        "Tensor mask_types,"
        "Tensor row_to_slice,"
        "Tensor diagonal_offsets,"
        "Tensor band_widths,"
        "Tensor? vbatch_to_slice = None,"
        "Tensor? row_to_vbatch_start = None,"
        "Tensor? row_to_vbatch_end = None,"
        "Tensor? dq = None,"
        "Tensor? dk = None,"
        "Tensor? dv = None,"
        "float? softmax_scale = None,"
        "bool deterministic=False,"
        "float? softcap = None,"
        "bool reduce_kv=False,"
        "bool use_loop_k=False,"
        "bool inner_min_to_max=False,"
        "bool persistent_scheduler=False,"
        "float dropout_p=0.0,"
        "Tensor? rng_state=None,"
        "Tensor? mask_bits=None,"
        "Tensor? bh_to_group=None,"
        "Tensor? group_slice_offsets=None,"
        "Tensor? group_vb_offsets=None) -> (Tensor, Tensor, Tensor)");
    // Fused mask-decomposition ops (mask_decomp_kernels.cu).
    m.def("row_stats(Tensor mask) -> (Tensor, Tensor, Tensor)");
    m.def("peel_first_interval(Tensor(a!) blk) -> (Tensor, Tensor, Tensor(a!))");
    // Whole-mask decomposition: desc pack (slices [7,CAP], rows [3,sq],
    // vbatch [CAP], counters [4] = [num_slices, num_vbatches, supported,
    // n_layers]); kblock_m is the fwd Q-tile (128) used by the layout.
    m.def("decompose_mask(Tensor mask, int kblock_m) -> "
          "(Tensor, Tensor, Tensor, Tensor)");
    // Same contract as decompose_mask but NEVER cached: used by the SDPA
    // glue inside CUDA graph capture so replays re-run the pipeline.
    m.def("decompose_mask_nocache(Tensor mask, int kblock_m) -> "
          "(Tensor, Tensor, Tensor, Tensor)");
    // Post-replay envelope check (sdpa_glue.cpp): raises when the most
    // recent graph replay decomposed a mask outside the envelope.
    m.def("sdpa_graph_mask_check() -> ()");
}

TORCH_LIBRARY_IMPL(flex_flash_attention, CUDA, m) {
    m.impl("fwd", &mha_flex_flash_fwd);
    m.impl("bwd", &mha_flex_flash_bwd);
    m.impl("row_stats", &mha_flex_flash_decomp_row_stats);
    m.impl("peel_first_interval", &mha_flex_flash_decomp_peel_first_interval);
    m.impl("decompose_mask", &mha_flex_flash_decompose_mask);
    m.impl("decompose_mask_nocache", &mha_flex_flash_decompose_mask_nocache);
    m.impl("fwd_f32", &mha_flex_flash_fwd_f32);
    m.impl("bwd_f32", &mha_flex_flash_bwd_f32);
}

// No-tensor-argument op: the dispatcher cannot infer a device key, so it
// must live on CompositeExplicitAutograd (a CUDA-keyed impl is unreachable).
TORCH_LIBRARY_IMPL(flex_flash_attention, CompositeExplicitAutograd, m) {
    m.impl("sdpa_graph_mask_check",
           []() { flex_flash_attention::sdpa_graph_mask_check(); });
}

// Shape-only impl for FakeTensorMode / torch.compile tracing.
TORCH_LIBRARY_IMPL(flex_flash_attention, Meta, m) {
    m.impl("decompose_mask", &mha_flex_flash_decompose_mask_meta);
    m.impl("decompose_mask_nocache", &mha_flex_flash_decompose_mask_meta);
    m.impl("fwd", &mha_flex_flash_fwd_meta);
    m.impl("bwd", &mha_flex_flash_bwd_meta);
    m.impl("fwd_f32", &mha_flex_flash_fwd_f32_meta);
    m.impl("bwd_f32", &mha_flex_flash_bwd_f32_meta);
}
