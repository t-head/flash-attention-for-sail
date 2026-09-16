/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Launch template for the INDEPENDENT flex flash attention forward kernel.
//
// Key difference from the old launch template:
//   - Uses FlashAttnFlexFlashFwdSm80 (new, independent kernel) instead of
//     the wrapper-based FlexFlashAttentionFwdSm80.
//   - Still uses CollectiveMainloopFwdSm80 as a TYPE PROVIDER only — its
//     mma() function is never called.
//   - Slice metadata (AttnSliceParams) is passed directly to the kernel.

#pragma once

#include <cstdlib>   // getenv (FORCE_KBM64 A/B switch)
#include <cstdio>    // FLASH_ATTENTION_ARB_DBG counter dump

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/kernel_launch.h"

#include "../hopper/static_switch.h"
#include "../hopper/flash.h"
#include "../hopper/tile_size.h"
#include "../hopper/tile_scheduler.hpp"
#include "../hopper/mainloop_fwd_sm80.hpp"   // TYPE PROVIDER ONLY
#include "../hopper/epilogue_fwd.hpp"
#include "flex_flash_attention_fwd_kernel_sm80.h"
#include "tile_scheduler_flex_flash.h"
#include "block_meta_flex_flash.h"     // DispatchDirection
#include "attn_slice.h"

using namespace cute;

template <int Arch, int kHeadDim, int kHeadDimV, typename Element, typename ElementOut,
          bool PackGQA, bool V_colmajor, bool Has_softcap = false,
          flash::DispatchDirection kDir = flash::DispatchDirection::MaxToMin,
          bool PersistentScheduler = false, bool kBlockM128 = false, bool kBlockN64 = false,
          bool Is_dropout = false>
void run_flex_flash_attention_fwd(FlexFlashAttentionFwdParams &params, hggcStream_t stream) {
    // Fixed flags for flex flash attention path.
    static constexpr bool Is_causal = false;
    static constexpr bool Is_local = false;
    static constexpr bool Varlen = false;
    static constexpr bool PagedKVNonTMA = false;
    static constexpr bool AppendKV = false;
    static constexpr bool Split = false;

    using ArchTag = std::conditional_t<Arch == 89, cutlass::arch::PPU0015, cutlass::arch::PPU0010>;
    // S-aux is always passed as nullptr on this path; match the element type
    // anyway so fp32 instantiations never reinterpret across widths.
    using ElementS = std::conditional_t<sizeof(Element) == 4, float,
                                        cutlass::bfloat16_t>;

    // Tile sizes from PPU tile_size_fwd_ppu.
    static constexpr std::tuple<int, int, int, int, bool> kBlockMN_kNWarps_Stages_RS = tile_size_fwd_ppu(
        Arch, kHeadDim, kHeadDimV,
        false /*Is_causal*/, false /*Is_local*/, sizeof(Element) /*element_size*/,
        false /*PagedKVNonTMA*/, false /*Varlen*/, false /*Split*/,
        false /*Has_softcap*/, false /*AppendKV*/, PackGQA,
        kBlockM128, false /*kBlockM16*/, false /*kBlockN16*/,
        false /*PagedKVAiu*/);

    static constexpr int kBlockM  = std::get<0>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kBlockN  = kBlockN64 ? 64 : std::get<1>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kNWarps  = std::get<2>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kStages  = std::get<3>(kBlockMN_kNWarps_Stages_RS);
    static constexpr bool Q_in_regs = std::get<4>(kBlockMN_kNWarps_Stages_RS);

    static constexpr bool PagedKVAiu = false;
    static constexpr int kBlockNPagedPerAiuLoad = kBlockN;  // unused (PagedKV=false)

    using TileShape_MNK    = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
    using ClusterShape     = cute::Shape<Int<1>, _1, _1>;

    // ── CollectiveMainloopFwdSm80 used as TYPE PROVIDER only ──
#if defined(USE_PPU) && USE_AIU
    using CollectiveMainloop = flash::CollectiveMainloopFwdSm80<
        kNWarps, kStages, Q_in_regs, TileShape_MNK, kHeadDimV, Element, float, ArchTag,
        Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, PagedKVAiu, kBlockNPagedPerAiuLoad, AppendKV, PackGQA, Split, ElementS>;
#else
    using CollectiveMainloop = flash::CollectiveMainloopFwdSm80<
        kNWarps, kStages, Q_in_regs, TileShape_MNK, kHeadDimV, Element, float, ArchTag,
        Is_causal, Is_local, Has_softcap, Varlen, PagedKVNonTMA, AppendKV, PackGQA, Split, ElementS>;
#endif

    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
        TileShape_MNK_PV, ClusterShape, ElementOut, ArchTag,
        CollectiveMainloop::NumMmaThreads, Varlen, PackGQA, Split, false>;

    // NOTE: the native PPU fwd path uses SingleTileScheduler (full grid),
    // but A/B (diag_sched, s<=4096 d<=128) showed the full grid is SLOWER for
    // this kernel: our per-CTA setup (AIU desc init, slice metadata) is
    // amortized across tiles only in persistent mode, and diagonal masks make
    // the per-wave tail worse (causal s=4096 d=128: 2.73ms persistent vs
    // 6.80ms single-tile).  Keep the persistent grid; the remaining gap to
    // native is per-tile efficiency, not scheduling.
    using Scheduler = std::conditional_t<PersistentScheduler,
        flash::FlexFlashTileSchedulerSM80<kBlockM>,
        flash::StaticPersistentTileScheduler<Split>>;

    // The INDEPENDENT kernel class.
    using AttnKernel = flash::FlashAttnFlexFlashFwdSm80<
        CollectiveMainloop, CollectiveEpilogue, Scheduler, kDir, Has_softcap, Is_dropout>;

    // ── Build mainloop arguments ──
    int seqlen_q = params.seqlen_q;
    int batch_q  = params.b;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},                        // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},  // stride_Q
        static_cast<Element*>(params.k_ptr),
        {params.seqlen_k, params.d, params.h_k, params.b},              // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},  // stride_K
        static_cast<Element*>(params.v_ptr),
        params.dv,                                                       // headdim_v
        {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},  // stride_V
        nullptr, {0, params.d, params.h_k, 1}, {0, _1{}, 0, 0},         // K_new
        nullptr, {0, _1{}, 0, 0},                                        // V_new
        nullptr, {0, _1{}, 0, 0},                                        // Qv
        nullptr, {params.seqlen_k, 0}, {0, _1{}},                        // rotary cos
        nullptr, {0, _1{}},                                              // rotary sin
        false,                                                           // is_rotary_interleaved
        nullptr, {params.b, 0}, {0, _1{}},                               // pagetable
        params.scale_softmax,
        nullptr, nullptr, nullptr,                                       // descale pointers
        {0, 0}, {0, 0}, {0, 0},                                          // descale strides
        0, 0,                                                            // window_size_left/right
        0,                                                               // attention_chunk
        Has_softcap ? params.softcap : 0.f,                              // softcap_val = cap; mainloop derives scale/cap & cap*log2e
        1,                                                               // num_splits
        nullptr,                                                         // kv_batch_idx
        nullptr, nullptr, nullptr,                                       // cu_seqlens
        nullptr, nullptr,                                                // seqused
        nullptr, nullptr                                                 // leftpad_k, seqlens_rotary
        , static_cast<ElementS const*>(nullptr)                          // ptr_S_aux
    };
    // Softcap note: with Has_softcap the mainloop's to_underlying_arguments
    // converts softmax_scale_log2 -> cap*log2(e) and softcap_val -> scale/cap.

    // ── Build epilogue arguments ──
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q, 1},                    // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride, 0},  // stride_O
        static_cast<float*>(params.oaccum_ptr),
        {0, _1{}, 0, 0, 0},                                              // stride_O_partial
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, params.h * seqlen_q, 0},                        // stride_LSE
        nullptr, {_1{}, seqlen_q, params.h * seqlen_q, 0},               // LSE partial
        params.h_k,
        nullptr, nullptr                                                 // cu_seqlens_q, seqused_q
    };

    // ── Build scheduler arguments ──
    int qhead_per_khead = !PackGQA ? 1 : cutlass::ceil_div(params.h, params.h_k);
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));
    num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        false /*varlen_q*/, false /*extreme_varlen_q*/,
        num_blocks_m, !PackGQA ? params.h : params.h_k, params.b, 1 /*num_splits*/,
        params.h / params.h_k,
        params.seqlen_q,
        params.seqlen_k, params.d, params.dv, sizeof(Element),
        params.tile_count_semaphore, nullptr, nullptr,
        nullptr,  // num_splits_dynamic_ptr
    };

    // ── Assemble kernel params ──
    int device;
    CHECK_CUDA(hggcGetDevice(&device));

    // FLASH_ATTENTION_ARB_DBG=1: perf-debug counters.  The launch serializes
    // (sync + readback after every launch) — diagnostics only, never enable
    // in timing runs.
    static unsigned long long* dbg_counters = [] {
        char const* e = getenv("FLASH_ATTENTION_ARB_DBG");
        if (e == nullptr || e[0] != '1') { return (unsigned long long*)nullptr; }
        unsigned long long* p = nullptr;
        cudaMalloc(&p, 8 * sizeof(unsigned long long));
        return p;
    }();
    if (dbg_counters) {
        cudaMemset(dbg_counters, 0, 8 * sizeof(unsigned long long));
        params.slices.dbg_counters = dbg_counters;
    }

    typename AttnKernel::Arguments kernel_args {
        mainloop_args,
        epilogue_args,
        {device, params.num_sm},
        scheduler_args,
        params.slices
    };

    // Dropout: eager rng_state is a 2-element host tensor ([seed, offset])
    // read here and passed by value.  During CUDA graph capture rng_state
    // is nullptr (device tensor, kernel-populated) and the kernel instead
    // resolves the stream from the philox_* device pointers each replay.
    if constexpr (Is_dropout) {
        kernel_args.dropout = {
            params.rng_state ? params.rng_state[0] : 0ull,
            params.rng_state ? params.rng_state[1] : 0ull,
            params.rp_dropout,
            params.p_dropout_in_uint8_t,
            params.h
        };
        kernel_args.dropout.philox_seed_ptr = params.philox_seed_ptr;
        kernel_args.dropout.philox_offset_ptr = params.philox_offset_ptr;
        kernel_args.dropout.intragraph_offset = params.intragraph_offset;
        kernel_args.dropout.rng_state_dev = params.rng_state_dev;
    }

    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments(kernel_args);

    dim3 grid_dims  = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size   = AttnKernel::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        auto kernel = cutlass::device_kernel<AttnKernel>;
        CHECK_CUDA(hggcFuncSetAttribute(
            kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

#ifdef USE_PPU
    int blocks_per_sm;
    auto kernel = cutlass::device_kernel<AttnKernel>;
    hggcError_t status_ = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, block_dims.x * block_dims.y * block_dims.z, smem_size);
    grid_dims = AttnKernel::get_grid_shape(kernel_params, blocks_per_sm);
#endif

    cutlass::kernel_launch<AttnKernel>(
        grid_dims, block_dims, smem_size, stream, kernel_params,
        false /*launch_with_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();

    if (dbg_counters) {
        cudaDeviceSynchronize();
        unsigned long long c[8] = {0};
        cudaMemcpy(c, dbg_counters, sizeof(c), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[ARB_DBG] kbm=%d kbn=%d steps=%llu masked=%llu cycles=%llu "
                "cycles/step=%.0f\n",
                kBlockM128 ? 128 : 64, kBlockN,
                c[0], c[1], c[4],
                c[0] ? double(c[4]) / double(c[0]) : 0.0);
    }
}

// Dispatch function: runtime feature dispatch (softcap is env-gated so the
// default build keeps a single softcap=false variant; inner direction and
// the persistent scheduler are always compiled).  Shared by the 16-bit and
// the fp32 entry points below; templated on T so fp32 gets its own
// instantiations without relaxing the 16-bit width assert.
template<int Arch, typename T, int kHeadDim, int kHeadDimV, bool PackGQA>
void run_mha_flex_flash_fwd_dispatch_impl(FlexFlashAttentionFwdParams &params, hggcStream_t stream) {
    static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t>
                                 || cute::is_same_v<T, cutlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>;
    static constexpr bool V_colmajor = false;
    // p_dropout_in_uint8_t holds p_keep*255 (FA2 convention): 255 == keep all.
    // Template-gated so the dropout-free path (the common case) compiles to
    // exactly the same code as before.
    BOOL_SWITCH(params.p_dropout_in_uint8_t < 255, Is_dropout, [&] {
#ifdef FLASH_ATTENTION_ARB_SOFTCAP
    BOOL_SWITCH(params.softcap > 0.f, Has_softcap, [&] {
#else
    {
        constexpr static bool Has_softcap = false;
#endif
        // Mirror the native use_kblockm_128 heuristic (non-varlen, non-causal
        // branch of flash_api.cpp): the larger Q tile pays off once the
        // sequence is long enough AND the grid still fills the GPU at
        // kBlockM=128 (>= 8 tiles per SM).  Without it the prefill fwd
        // kernels run at ~45% of the native throughput.  Native raises the
        // seqlen threshold to ~704 for causal; our causal-like masks arrive
        // as slices (is_causal=false) and are covered by the parallelism
        // guard plus the kernel's empty-block clipping, so the lower
        // threshold is kept here.
        // FLASH_ATTENTION_ARB_FORCE_KBM64=1 forces kBlockM=64 (A/B debug).
        static bool const force_kbm64 = [] {
            char const *e = getenv("FLASH_ATTENTION_ARB_FORCE_KBM64");
            return e != nullptr && e[0] == '1';
        }();
        // Host hint: the Python layer knows the slice structure and disables
        // kBlockM=128 for masks with large-span diagonal slices (CAUSAL /
        // INVCAUSAL / BICAUSAL crossing many m_blocks), which run
        // pathologically slow at the larger Q tile.  No hint → allow it
        // (shape-only heuristic for the precomputed entry point).
        bool const hint_allows_kbm128 =
            !params.kblockm128_has_hint || params.kblockm128_value;
        // kHeadDim == 128 (not <=): at head_dim 64 kBlockM=128 measured
        // ~4x slower (stair s=1024: 0.736 vs 0.183ms), so the kernel-side
        // fallback heuristic must never pick it when no host hint is given.
        // hdim 256 has its own kBlockM=128 tile arm in tile_size_fwd_ppu
        // (128x64, 8 warps, FA2-style; measured -19..-34% fwd) and uses the
        // same parallelism guard here.  hdim 192 measured WORSE with the
        // 128 tile (dense fwd +17%), so it stays on kBlockM=64.
        bool const use_kblockm_128 = !force_kbm64 && hint_allows_kbm128 &&
            (kHeadDim == 128 || kHeadDim == 256) &&
            params.seqlen_q > 64 &&
            params.b * (PackGQA ? params.h_k : params.h)
                   * ((params.seqlen_q * (PackGQA ? params.h / params.h_k : 1) + 127) / 128)
                   / params.num_sm >= 8;
        BOOL_SWITCH(use_kblockm_128, kBlockM128, [&] {
        DIRECTION_SWITCH(params.inner_min_to_max, kDir, [&] {
            BOOL_SWITCH(params.persistent_scheduler, PersistentScheduler, [&] {
                if constexpr (kBlockM128) {
                    run_flex_flash_attention_fwd<Arch, kHeadDim, kHeadDimV, T, T_out, PackGQA, V_colmajor,
                                            Has_softcap, kDir, PersistentScheduler, kBlockM128, false,
                                            Is_dropout>(params, stream);
                } else {
                    // kBlockN=64 variant (short-K-span dense slice sets, e.g.
                    // block-diagonal document masks); only for the kBlockM=64
                    // tiles to bound instantiation count.
                    BOOL_SWITCH(params.kblockn64, kBlockN64, [&] {
                        run_flex_flash_attention_fwd<Arch, kHeadDim, kHeadDimV, T, T_out, PackGQA, V_colmajor,
                                                Has_softcap, kDir, PersistentScheduler, kBlockM128, kBlockN64,
                                                Is_dropout>(params, stream);
                    });
                }
            });
        });
        });
#ifdef FLASH_ATTENTION_ARB_SOFTCAP
    });
#else
    }
#endif
    });
}

// 16-bit / 8-bit entry point (unchanged contract: rejects other widths at
// compile time).
template<int Arch, typename T, int kHeadDim, int kHeadDimV, bool PackGQA>
void run_mha_flex_flash_fwd_(FlexFlashAttentionFwdParams &params, hggcStream_t stream) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "Only 16bit and 8bit are supported");
    run_mha_flex_flash_fwd_dispatch_impl<Arch, T, kHeadDim, kHeadDimV, PackGQA>(params, stream);
}

// fp32 entry point — the ISOLATED fp32 kernel set (TF32 tensor-core
// compute).  Shares the runtime feature dispatch above but gets its own
// explicit instantiations (flex_flash_attention_fwd_sm89_hdim*_f32.cu) and its
// own C++/Python API entry, leaving the bf16/fp16 set untouched.
template<int Arch, int kHeadDim, int kHeadDimV, bool PackGQA>
void run_mha_flex_flash_fwd_f32_(FlexFlashAttentionFwdParams &params, hggcStream_t stream) {
    run_mha_flex_flash_fwd_dispatch_impl<Arch, float, kHeadDim, kHeadDimV, PackGQA>(params, stream);
}
