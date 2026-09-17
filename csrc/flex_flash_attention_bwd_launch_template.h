/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Launch template for the INDEPENDENT flex flash attention backward kernel.
//
// Pipeline (all kernels on the same stream):
//   1. FlashAttnBwdPreprocess (reused):
//      delta = rowsum(dO * O)  →  dsoftmax_sum
//      LSE(natural log, from fwd) → LSE * log2(e)  →  softmax_lse_log2
//      clears dQaccum (and the dQ semaphore when Deterministic LoopQ)
//   2. FlashAttnFlexFlashBwdSm80 (this feature), two mainloop variants:
//      LoopQ (BwdInnerLoopK=false): grid (num_n_blocks, h, b)
//      LoopK (BwdInnerLoopK=true):  grid (num_m_blocks, h, b)
//   3. FlashAttnBwdPostprocessConvertdQ ×3 (reused):
//      dQaccum * softmax_scale → dQ ;  dKaccum → dK ;  dVaccum → dV
//      With ReduceKV (LoopK only) the dK/dV postprocess reads the fp32
//      workspaces and the dK multiply uses softmax_scale (the main kernel
//      skipped it) — so no separate reduce kernel is needed.
//
// Feature template parameters (host runtime dispatch in run_mha_flex_flash_bwd_):
//   Has_softcap, Deterministic, ReduceKV, kDir (inner-loop direction),
//   BwdInnerLoopK, PersistentScheduler.
//
// Deterministic: dK/dV semaphore tensors must be passed in via params
// (zero-initialized host-side); LoopQ also needs params.dq_semaphore
// (cleared by the preprocess kernel).  With LoopQ+Deterministic the epilogue
// store is skipped inside the kernel (per-slice turnstile flushes instead),
// so dk/dv accum must be zero-initialized host-side.
// ReduceKV: params.dk_accum_ptr/dv_accum_ptr carry the ZERO-initialized fp32
// workspaces (same shape as the accums).
//
// CollectiveMainloopBwdSm80 is used as a TYPE PROVIDER only — its mma()
// function is never called.

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/kernel_launch.h"
#include <cstdio>  // P4 fp32 diagnosis print (host-side, env-gated)
#include <cstdlib>  // diagnosis getenv/atoi

#include "../hopper/static_switch.h"
#include "../hopper/flash.h"
#include "../hopper/tile_scheduler.hpp"
#include "../hopper/flash_bwd_preprocess_kernel.h"
#include "../hopper/flash_bwd_postprocess_kernel.h"
#include "../hopper/mainloop_bwd_sm80.hpp"   // TYPE PROVIDER ONLY
#include "../hopper/epilogue_bwd.hpp"
#include "flex_flash_attention_bwd_kernel_sm80.h"
#include "tile_scheduler_flex_flash.h"
#include "block_meta_flex_flash.h"     // DispatchDirection
#include "attn_slice.h"

using namespace cute;

template <int Arch, int kHeadDim, int kBlockM, int kBlockN, typename Element,
          int Stages_dO, int Stages,
          bool SdP_swapAB, bool dKV_swapAB, bool dQ_swapAB,
          int NumMmaWarpGroups, int AtomLayoutMSdP, int AtomLayoutNdKV, int AtomLayoutMdQ,
          bool V_in_regs,
          bool Has_softcap = false, bool Deterministic = false, bool ReduceKV = false,
          flash::DispatchDirection kDir = flash::DispatchDirection::MaxToMin,
          bool BwdInnerLoopK = false, bool PersistentScheduler = false,
          bool Is_dropout = false>
void run_flex_flash_attention_bwd(FlexFlashAttentionBwdParams &params, hggcStream_t stream) {
    using ElementAccum = float;
    using ArchTag = std::conditional_t<Arch == 89, cutlass::arch::PPU0015, cutlass::arch::PPU0010>;

    static constexpr bool Is_causal = false;
    static constexpr bool Is_local  = false;
    static constexpr bool Varlen = false;

    int const seqlen_q = params.seqlen_q;
    int const seqlen_k = params.seqlen_k;
    int const seqlen_q_rounded = params.seqlen_q_rounded;
    int const seqlen_k_rounded = params.seqlen_k_rounded;

    // ── 1. Preprocess: delta = rowsum(dO*O), LSE → LSE_log2, clear dQaccum ──
    // LoopQ+Deterministic also clears the dQ semaphore here.
    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using PreprocessKernel = flash::FlashAttnBwdPreprocess<
        TileShape_MK, Element, ElementAccum, ArchTag, /*Clear_dQaccum=*/true, Varlen>;
    typename PreprocessKernel::Arguments preprocess_args {
        static_cast<Element const*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, params.b},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride},  // stride_O
        static_cast<Element const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride, params.do_batch_stride},  // stride_dO
        static_cast<float*>(params.dsoftmax_sum),
        {seqlen_q_rounded, params.h, params.b},  // shape_dPsum
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},  // stride_dPsum
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, params.h * seqlen_q},  // stride_LSE (fwd LSE buffer)
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},  // stride_LSE_log2
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, params.b},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, params.d_rounded * seqlen_q_rounded * params.h},  // stride_dQaccum
        params.b,
        (Deterministic && !BwdInnerLoopK) ? params.dq_semaphore : nullptr  // dq_semaphore
    };
    typename PreprocessKernel::Params preprocess_params =
        PreprocessKernel::to_underlying_arguments(preprocess_args);
    int num_m_block = cute::ceil_div(params.seqlen_q, kBlockM);
    dim3 grid_m(num_m_block, params.h, params.b);
    cutlass::kernel_launch<PreprocessKernel>(grid_m, PreprocessKernel::MaxThreadsPerBlock,
                                             PreprocessKernel::SharedStorageSize, stream,
                                             preprocess_params, false /*launch_with_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();

    // ── 2. Main flex flash attention bwd kernel ──
    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using CollectiveMainloop = flash::CollectiveMainloopBwdSm80<
        Stages, Stages_dO, TileShape_MNK, Element, ElementAccum, ArchTag,
        Is_causal, Is_local, Has_softcap, Varlen, Deterministic,
        SdP_swapAB, dKV_swapAB, dQ_swapAB,
        NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>;

    // GQA epilogue (fp32 accum + atomicAdd) is used for ALL cases.
    using CollectiveEpilogue = flash::CollectiveEpilogueBwdGQA<
        TileShape_MNK, ElementAccum, ArchTag, CollectiveMainloop::NumMmaThreads,
        Varlen, Deterministic>;

    // Tile granularity: LoopQ tiles over n_blocks, LoopK over m_blocks.
    static constexpr int kSchedulerBlock = BwdInnerLoopK ? kBlockM : kBlockN;
    using Scheduler = std::conditional_t<PersistentScheduler,
        flash::FlexFlashTileSchedulerSM80<kSchedulerBlock>,
        flash::SingleTileScheduler<Varlen, false /*Split*/, false /*PackGQA*/, kSchedulerBlock>>;

    using AttnKernel = flash::FlashAttnFlexFlashBwdSm80<
        CollectiveMainloop, CollectiveEpilogue, Scheduler,
        Has_softcap, Deterministic, ReduceKV, kDir, BwdInnerLoopK, Is_dropout>;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, params.b},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},  // stride_Q
        static_cast<Element const*>(params.k_ptr),
        {seqlen_k, params.d, params.h_k, params.b},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},  // stride_K
        static_cast<Element const*>(params.v_ptr),
        {seqlen_k, params.dv, params.h_k, params.b},  // shape_V
        {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},  // stride_V
        static_cast<Element const*>(params.do_ptr),
        {seqlen_q, params.dv, params.h, params.b},  // shape_dO
        {params.do_row_stride, _1{}, params.do_head_stride, params.do_batch_stride},  // stride_dO
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, params.b},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, params.d_rounded * seqlen_q_rounded * params.h},  // stride_dQaccum
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {seqlen_q_rounded, params.h, params.b},  // shape_LSE
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},  // stride_LSE_log2
        static_cast<float*>(params.dsoftmax_sum),
        {_1{}, seqlen_q_rounded, params.h * seqlen_q_rounded},  // stride_dPsum
        params.scale_softmax,
        -1, -1, 0,  // window_size_left, window_size_right, attention_chunk
        Has_softcap ? params.softcap : 0.f,  // softcap (mainloop derives cap*log2e & scale/cap)
        params.b,
        (Deterministic && !BwdInnerLoopK) ? params.dq_semaphore : nullptr  // dq_semaphore
    };

    // ReduceKV routes dK/dV into the fp32 workspaces (passed through the
    // accum pointer slots; zero-initialized host-side).
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementAccum*>(params.dk_accum_ptr),
        {seqlen_k_rounded * params.d_rounded, params.h_k, params.b},  // shape_dKaccum
        {_1{}, params.d_rounded * seqlen_k_rounded, params.h_k * params.d_rounded * seqlen_k_rounded},  // stride_dKaccum
        static_cast<ElementAccum*>(params.dv_accum_ptr),
        {seqlen_k_rounded * params.dv_rounded, params.h_k, params.b},  // shape_dVaccum
        {_1{}, params.dv_rounded * seqlen_k_rounded, params.h_k * params.dv_rounded * seqlen_k_rounded},  // stride_dVaccum
        params.h,
        Deterministic ? params.dk_semaphore : nullptr,
        Deterministic ? params.dv_semaphore : nullptr
    };

    int const num_sched_blocks = BwdInnerLoopK
        ? cutlass::ceil_div(params.seqlen_q, get<0>(TileShape_MNK{}))
        : cutlass::ceil_div(params.seqlen_k, get<1>(TileShape_MNK{}));
    typename flash::TileSchedulerArguments scheduler_args {
        false /* varlen_q */, false /* extreme_varlen_q */,
        num_sched_blocks, params.h, params.b, 1 /*num_splits*/,
        params.h / params.h_k,
        BwdInnerLoopK ? params.seqlen_q : params.seqlen_k,
        BwdInnerLoopK ? params.seqlen_k : params.seqlen_q,
        params.d, params.dv, sizeof(Element),
        PersistentScheduler ? params.tile_count_semaphore : nullptr, nullptr, nullptr
    };

    int device;
    CHECK_CUDA(hggcGetDevice(&device));
    // Dropout replay state: eager rng_state is the 2-element CPU tensor
    // ([seed, offset]) returned by fwd, read by value here.  Captured fwd
    // published the pair into a DEVICE tensor instead, so params.rng_state
    // is nullptr and the kernel resolves the stream from the philox_*
    // device pointers (pointing at that tensor) each replay.
    typename AttnKernel::DropoutArgs dropout_args{};
    if constexpr (Is_dropout) {
        dropout_args = {
            params.rng_state ? params.rng_state[0] : 0ull,
            params.rng_state ? params.rng_state[1] : 0ull,
            params.rp_dropout,
            params.p_dropout_in_uint8_t,
            params.h
        };
        dropout_args.philox_seed_ptr = params.philox_seed_ptr;
        dropout_args.philox_offset_ptr = params.philox_offset_ptr;
        dropout_args.intragraph_offset = params.intragraph_offset;
    }
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args, params.slices,
        nullptr /*dkv_flags (reserved)*/,
        dropout_args
    });

    dim3 grid_dims  = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size   = AttnKernel::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(hggcFuncSetAttribute(cutlass::device_kernel<AttnKernel>,
                                        hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream,
                                       kernel_params, false /*launch_with_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();

    // ── 3. Postprocess: fp32 accums → Element outputs ──
    // dQ: dQaccum * softmax_scale → dQ
    // NOTE: the CVT residue for dQ is NOT a dQaccum row permutation (probed:
    // it is an n-axis dS/K pairing error inside the dQ GEMM), so no Undo here.
    using PostprocessKernel = flash::FlashAttnBwdPostprocessConvertdQ<
        TileShape_MK, Element, ElementAccum, ArchTag,
        CollectiveMainloop::NumMmaThreads,
        typename CollectiveMainloop::TiledMmadQ,
        CollectiveMainloop::dQ_swapAB>;
    typename PostprocessKernel::Arguments postprocess_args {
        static_cast<ElementAccum const*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, params.b},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, params.d_rounded * seqlen_q_rounded * params.h},  // stride_dQaccum
        static_cast<Element*>(params.dq_ptr),
        {seqlen_q, params.d, params.h, params.b},  // shape_dQ
        {params.dq_row_stride, _1{}, params.dq_head_stride, params.dq_batch_stride},  // stride_dQ
        params.scale_softmax
    };
    typename PostprocessKernel::Params postprocess_params =
        PostprocessKernel::to_underlying_arguments(postprocess_args);
    int num_m_block_postprocess = cute::ceil_div(params.seqlen_q, get<0>(TileShape_MK{}));
    dim3 grid_m_postprocess(num_m_block_postprocess, params.h, params.b);
    int smem_size_postprocess = PostprocessKernel::SharedStorageSize;
    if (smem_size_postprocess >= 48 * 1024) {
        CHECK_CUDA(hggcFuncSetAttribute(cutlass::device_kernel<PostprocessKernel>,
                                        hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
    }
    cutlass::kernel_launch<PostprocessKernel>(grid_m_postprocess, PostprocessKernel::MaxThreadsPerBlock,
                                              smem_size_postprocess, stream, postprocess_params,
                                              false /*launch_with_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();

    // dK / dV: fp32 accum (or ReduceKV workspace) → Element.
    // Scale: normally 1.f (dK was scaled in the main kernel); with ReduceKV
    // the main kernel skipped it, so dK picks up softmax_scale here — this
    // IS the "reduce" step, no separate kernel required.
    using TileShape_NK = cute::Shape<Int<kBlockN>, Int<kHeadDim>>;
    // With the slice/AIU PdSt path (PPU1v0||PPU1v5 R2S_SLICE_LAYOUT) the dK/dV
    // accumulators come out row-clean (probed: got==ref[tau] was the OLD undo
    // reapplying tau on already-clean rows), so no Undo here.
    using PostprocessKerneldKV = flash::FlashAttnBwdPostprocessConvertdQ<
        TileShape_NK, Element, ElementAccum, ArchTag,
        CollectiveEpilogue::NumEpilogueThreads,
        typename CollectiveMainloop::TiledMmadKV,
        CollectiveMainloop::dKV_swapAB>;
    typename PostprocessKerneldKV::Arguments postprocess_dK_args {
        static_cast<ElementAccum const*>(params.dk_accum_ptr),
        {seqlen_k_rounded * params.d_rounded, params.h_k, params.b},  // shape_dKaccum
        {_1{}, params.d_rounded * seqlen_k_rounded, params.h_k * params.d_rounded * seqlen_k_rounded},  // stride_dKaccum
        static_cast<Element*>(params.dk_ptr),
        {seqlen_k, params.d, params.h_k, params.b},  // shape_dK
        {params.dk_row_stride, _1{}, params.dk_head_stride, params.dk_batch_stride},  // stride_dK
        ReduceKV ? params.scale_softmax : 1.f
    };
    typename PostprocessKerneldKV::Params postprocess_dK_params =
        PostprocessKerneldKV::to_underlying_arguments(postprocess_dK_args);
    typename PostprocessKerneldKV::Arguments postprocess_dV_args {
        static_cast<ElementAccum const*>(params.dv_accum_ptr),
        {seqlen_k_rounded * params.dv_rounded, params.h_k, params.b},  // shape_dVaccum
        {_1{}, params.dv_rounded * seqlen_k_rounded, params.h_k * params.dv_rounded * seqlen_k_rounded},  // stride_dVaccum
        static_cast<Element*>(params.dv_ptr),
        {seqlen_k, params.dv, params.h_k, params.b},  // shape_dV
        {params.dv_row_stride, _1{}, params.dv_head_stride, params.dv_batch_stride},  // stride_dV
        // ReduceKV defers the dV rp scale here (the main kernel skips its
        // in-kernel multiply on this path); 1.f in the dropout-free case.
        (ReduceKV && params.p_dropout_in_uint8_t < 255) ? params.rp_dropout : 1.f
    };
    typename PostprocessKerneldKV::Params postprocess_dV_params =
        PostprocessKerneldKV::to_underlying_arguments(postprocess_dV_args);
    int num_n_block_postprocess = cute::ceil_div(params.seqlen_k, get<0>(TileShape_NK{}));
    dim3 grid_n_postprocess(num_n_block_postprocess, params.h_k, params.b);
    int smem_size_postprocess_kv = PostprocessKerneldKV::SharedStorageSize;
    if (smem_size_postprocess_kv >= 48 * 1024) {
        CHECK_CUDA(hggcFuncSetAttribute(cutlass::device_kernel<PostprocessKerneldKV>,
                                        hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess_kv));
    }
    // Diagnosis: FA_ARB_SKIP_POST=1 skips the dK convert, =2 skips dV.
    int skip_post = 0;
    if (char const* e = std::getenv("FA_ARB_SKIP_POST")) { skip_post = std::atoi(e); }
    if (!(skip_post & 1)) {
    cutlass::kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock,
                                                 smem_size_postprocess_kv, stream, postprocess_dK_params,
                                                 false /*launch_with_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();
    }
    if (!(skip_post & 2)) {
    cutlass::kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock,
                                                 smem_size_postprocess_kv, stream, postprocess_dV_params,
                                                 false /*launch_with_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();
    }
}

// ── Runtime feature dispatch over a fixed tile config ──
// Feature branches are env-gated so that the default build only compiles
// the base variant (all features false); each env switch adds one axis of
// template instantiations to every TU that includes this header:
//   FLASH_ATTENTION_ARB_LOOPK         → BwdInnerLoopK mainloop
//   FLASH_ATTENTION_ARB_REDUCEKV      → ReduceKV (implies LoopK)
//   FLASH_ATTENTION_ARB_SOFTCAP       → Has_softcap variants
//   FLASH_ATTENTION_ARB_DETERMINISTIC → Deterministic variants
// inner_min_to_max / persistent_scheduler are always compiled (cheap axes).
template <int Arch, typename T, int kHeadDim, int kBlockM, int kBlockN,
          int Stages_dO, int Stages,
          bool SdP_swapAB, bool dKV_swapAB, bool dQ_swapAB,
          int NumMmaWarpGroups, int AtomLayoutMSdP, int AtomLayoutNdKV, int AtomLayoutMdQ,
          bool V_in_regs, bool Is_dropout = false>
void run_mha_flex_flash_bwd_tile(FlexFlashAttentionBwdParams &params, hggcStream_t stream) {
#ifdef FLASH_ATTENTION_ARB_LOOPK
    BOOL_SWITCH(params.use_loop_k, BwdInnerLoopK, [&] {
#else
    {
        constexpr static bool BwdInnerLoopK = false;
#endif
#ifdef FLASH_ATTENTION_ARB_REDUCEKV
        BOOL_SWITCH(params.reduce_kv, ReduceKV, [&] {
#else
        {
            constexpr static bool ReduceKV = false;
#endif
            // Invalid combos (ReduceKV without LoopK, LoopK with V_in_regs
            // tile configs) compile to no-op branches; api.cpp rejects them
            // with TORCH_CHECK before dispatching.
            if constexpr (!ReduceKV || BwdInnerLoopK) {
                if constexpr (!V_in_regs || !BwdInnerLoopK) {
#ifdef FLASH_ATTENTION_ARB_SOFTCAP
                    BOOL_SWITCH(params.softcap > 0.f, Has_softcap, [&] {
#else
                    {
                        constexpr static bool Has_softcap = false;
#endif
#ifdef FLASH_ATTENTION_ARB_DETERMINISTIC
                        BOOL_SWITCH(params.deterministic, Deterministic, [&] {
#else
                        {
                            constexpr static bool Deterministic = false;
#endif
                            DIRECTION_SWITCH(params.inner_min_to_max, kDir, [&] {
                                BOOL_SWITCH(params.persistent_scheduler, PersistentScheduler, [&] {
                                    run_flex_flash_attention_bwd<
                                        Arch, kHeadDim, kBlockM, kBlockN, T,
                                        Stages_dO, Stages,
                                        SdP_swapAB, dKV_swapAB, dQ_swapAB,
                                        NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
                                        V_in_regs,
                                        Has_softcap, Deterministic, ReduceKV, kDir,
                                        BwdInnerLoopK, PersistentScheduler, Is_dropout>(params, stream);
                                });
                            });
#ifdef FLASH_ATTENTION_ARB_DETERMINISTIC
                        });
#else
                        }
#endif
#ifdef FLASH_ATTENTION_ARB_SOFTCAP
                    });
#else
                    }
#endif
                }
            }
#ifdef FLASH_ATTENTION_ARB_REDUCEKV
        });
#else
        }
#endif
#ifdef FLASH_ATTENTION_ARB_LOOPK
    });
#else
    }
#endif
}

// Tile configs mirror the USE_PPU branches of flash_bwd_launch_template.h
// (run_mha_bwd_hdimXX), which are validated on PPU 1.5.

template <int Arch, typename T, bool Is_dropout = false>
void run_mha_flex_flash_bwd_hdim32(FlexFlashAttentionBwdParams &params, hggcStream_t stream) {
    // Narrow-headdim bucket ported from the fp32 set (same geometry):
    // 32 is the narrowest width the generic CVT-free bwd path can express
    // (per-warp MMA N slice 32/2 = 16, a valid atom multiple; width 16
    // would give 8 and hits the 2:1 accum-copy static_assert).
    run_mha_flex_flash_bwd_tile<Arch, T, 32, 64, 128, 1, 2, false, false, false, 2, 2, 4, 4, false, Is_dropout>(params, stream);
}

template <int Arch, typename T, bool Is_dropout = false>
void run_mha_flex_flash_bwd_hdim64(FlexFlashAttentionBwdParams &params, hggcStream_t stream) {
    run_mha_flex_flash_bwd_tile<Arch, T, 64, 128, 64, 1, 2, false, false, false, 2, 4, 4, 4, false, Is_dropout>(params, stream);
}

template <int Arch, typename T, bool Is_dropout = false>
void run_mha_flex_flash_bwd_hdim96(FlexFlashAttentionBwdParams &params, hggcStream_t stream) {
    run_mha_flex_flash_bwd_tile<Arch, T, 96, 64, 128, 2, 2, false, false, false, 2, 2, 4, 4, false, Is_dropout>(params, stream);
}

template <int Arch, typename T, bool Is_dropout = false>
void run_mha_flex_flash_bwd_hdim128(FlexFlashAttentionBwdParams &params, hggcStream_t stream) {
    // AtomLayoutMdQ=1 (all 8 MMA warps along headdim) is required by the CVT
    // partner-warp dQaccum redirect (MR 29059073 validated config uses
    // AtomLayoutMdQ=1); AtomLayoutMdQ=2 breaks the warp^4 partner mapping.
    // kBlockM=48 + AtomLayoutM=1/N=2 mirrors MR 29059073's sm89 hdim128 tile:
    // smem drops to ~125KB so 2 CTA/CU fit (smem cap ~256KB; the 64x128
    // layout needs ~148KB and fell to 1 CTA/CU). Requires sm_89 gencode
    // (FLASH_ATTENTION_DISABLE_SM80=TRUE): the v1.5 CVT PdS atoms tile by 8
    // rows, while the v1.0 slice layout demands kBlockM % 32 == 0.
    run_mha_flex_flash_bwd_tile<Arch, T, 128, 48, 128, 1, 2, false, false, false, 2, 1, 2, 1, false, Is_dropout>(params, stream);
}

template <int Arch, typename T, bool Is_dropout = false>
void run_mha_flex_flash_bwd_hdim192(FlexFlashAttentionBwdParams &params, hggcStream_t stream) {
    // NOTE: the USE_PPU FA2 tile here is 64x128, but the ARB CVT epilogue
    // (epilogue_bwd.hpp) static_asserts 64 accumulator elements/thread, which
    // kBlockN=128 breaks; keep 64x64 until the epilogue supports wider tiles.
    run_mha_flex_flash_bwd_tile<Arch, T, 192, 64, 64, 1, 2, false, true, false, 2, 4, 2, 2, false, Is_dropout>(params, stream);
}

template <int Arch, typename T, bool Is_dropout = false>
void run_mha_flex_flash_bwd_hdim256(FlexFlashAttentionBwdParams &params, hggcStream_t stream) {
    // NOTE: same CVT epilogue constraint as hdim192 above — FA2's 64x128 PPU
    // tile is not usable here without an epilogue rework.
    run_mha_flex_flash_bwd_tile<Arch, T, 256, 64, 64, 1, 1, false, false, false, 2, 4, 2, 2, false, Is_dropout>(params, stream);
}

// Dispatch helper instantiated per (Arch, T, kHeadDim) so that each hdim
// compiles in its own translation unit (parallel builds, see
// instantiations/flex_flash_attention_bwd_sm89_hdimXX.cu).
template <int Arch, typename T, int kHeadDim>
void run_mha_flex_flash_bwd_(FlexFlashAttentionBwdParams &params, hggcStream_t stream) {
    // p_dropout_in_uint8_t holds p_keep*255 (FA2 convention): 255 == keep all.
    // Outermost switch ⇒ the dropout-free path compiles to the exact same
    // instantiations as before.
    BOOL_SWITCH(params.p_dropout_in_uint8_t < 255, Is_dropout, [&] {
        if constexpr (kHeadDim <= 32) {
            run_mha_flex_flash_bwd_hdim32<Arch, T, Is_dropout>(params, stream);
        } else if constexpr (kHeadDim <= 64) {
            run_mha_flex_flash_bwd_hdim64<Arch, T, Is_dropout>(params, stream);
        } else if constexpr (kHeadDim <= 96) {
            run_mha_flex_flash_bwd_hdim96<Arch, T, Is_dropout>(params, stream);
        } else if constexpr (kHeadDim <= 128) {
            run_mha_flex_flash_bwd_hdim128<Arch, T, Is_dropout>(params, stream);
        } else if constexpr (kHeadDim <= 192) {
            run_mha_flex_flash_bwd_hdim192<Arch, T, Is_dropout>(params, stream);
        } else {
            run_mha_flex_flash_bwd_hdim256<Arch, T, Is_dropout>(params, stream);
        }
    });
}
