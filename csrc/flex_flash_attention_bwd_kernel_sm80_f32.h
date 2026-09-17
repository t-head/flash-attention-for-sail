/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Independent flex flash attention BWD kernel for PPU 1.5 (SM89).
//
// Architecture (mirrors flex_flash_attention_fwd_kernel_sm80.h):
//   - NOT a wrapper around FA3's CollectiveMainloopBwdSm80::mma() — this
//     kernel has its own slice-aware mainloop.
//   - Uses CollectiveMainloopBwdSm80 ONLY as a TYPE PROVIDER (TiledMmaSdP /
//     TiledMmadKV / TiledMmadQ, SmemLayout*, GmemTiledCopy*, smem copy
//     atoms) to pick up PPU SM89-optimized MMA atoms and layouts.
//   - Reuses CollectiveEpilogueBwdGQA (dK/dV → fp32 accum via atomicAdd).
//   - Reuses flash::FlashAttnBwdPreprocess (delta = rowsum(dO*O), LSE→log2,
//     clear dQaccum) and flash::FlashAttnBwdPostprocessConvertdQ
//     (fp32 accum → bf16/fp16 with softmax_scale), launched from the
//     launch template.
//
// Two mainloop variants (MagiAttention-style dual loop direction):
//   LoopQ (BwdInnerLoopK=false): CTA = (n_block, h, b).  Outer loop over all
//     slices whose K range intersects this n_block; inner Q-block loop.
//     dK/dV register-accumulated across slices, one atomicAdd at the end;
//     dQ atomicAdded per (slice, m_block) step into fp32 dQaccum.
//   LoopK (BwdInnerLoopK=true):  CTA = (m_block, h, b).  Outer loop over all
//     slices whose Q range intersects this m_block; inner K-block loop with
//     staged K/V loading.  dQ accumulated in REGISTERS across slices (the
//     RangeMerge merge is therefore bitwise deterministic) and atomicAdded
//     once at the end; dK/dV atomicAdded per (slice, n_block) step.
//
// Feature flags (all compile-time, host does runtime dispatch):
//   Has_softcap_: tanh softcapping, FA3 sm80 semantics — apply_softcap right
//     after the S gemm, dtanh BEFORE masking (1-(-inf)^2 = NaN otherwise),
//     dS *= dtanh; dK keeps the ORIGINAL softmax_scale multiplier at the end.
//     The mainloop params carry the premultiplied scale (cap*log2e, scale/cap).
//   Deterministic_: fixed contribution order for every atomicAdd target via
//     semaphore turnstiles (ArbDetBarrier wait_eq/arrive_inc).
//     Turns are gap-free ranks over the exact contributor set (the tile ∩
//     mask-region predicate), so no turn value is ever skipped.
//     NOTE: LoopQ+dQ ordering requires the per-CTA turn sequence to be
//     monotone; the m loop is forced ascending in that mode.  Overlapping
//     slices with OPPOSITE diagonal directions can in principle break the
//     monotonicity — not an issue for the standard single-direction configs.
//   ReduceKV_: LoopK only.  dK/dV atomicAdds are routed into fp32 workspaces
//     (zero-initialized host-side; the per-element first-write-flag trick is
//     deliberately NOT used — it would race a plain store against concurrent
//     atomicAdds), and the softmax_scale multiplication moves to the
//     separate reduce kernel.
//
// Tile-level empty-block pruning (block_range_flex_flash.h): per (slice, tile)
// the inner-loop bounds are tightened with the mask's diagonal constraint.
// MaskFlexFlash::apply() still runs on every computed tile and keeps all of
// its responsibilities (diagonal constraint, K/Q range clipping, seqlen tail).

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/arch/barrier.h>

#include "../hopper/seqlen.h"
#include "../hopper/utils.h"
#include "../hopper/softmax.h"
#include "../hopper/tile_scheduler.hpp"
#include "../hopper/mainloop_bwd_sm80_f32.hpp"   // TYPE PROVIDER ONLY — we don't call mma()

#include "attn_slice.h"
#include "mask_flex_flash.h"
#include "block_meta_flex_flash.h"     // DispatchDirection + block cursors
#include "block_range_flex_flash.h"    // tile-level pruning + determinism helpers
#include "dropout_flex_flash.h"

namespace flash {

using namespace cute;

// ─── CTA-wide semaphore barrier for the deterministic turnstiles ───────────
// cutlass::GenericBarrier<SyncwarpSync>::wait_eq only gates WARP 0: the other
// warps race straight past the wait, so this CTA's arrive_inc can fire before
// waiting CTAs have observed their turn — counter values get skipped and the
// strict `== val` spin deadlocks (observed as intermittent hangs that depend
// on CTA scheduling / memory layout).  This barrier instead:
//   (a) polls with atomicAdd(ptr, 0) — the read goes through the same atomic
//       path as the increment, avoiding plain-load/atomic visibility gaps;
//   (b) synchronizes the WHOLE CTA on both sides of the critical section, so
//       the guarded atomicAdds happen strictly between wait and arrive.
struct ArbDetBarrier {
    CUTLASS_DEVICE
    static void wait_eq(int* lock_ptr, int thread_idx, int flag_idx, int val) {
        int* ptr = lock_ptr + flag_idx;
        if (thread_idx == 0) {
            #pragma unroll 1
            while (atomicAdd(ptr, 0) != val) {}
        }
        __syncthreads();
    }
    CUTLASS_DEVICE
    static void arrive_inc(int* lock_ptr, int thread_idx, int flag_idx) {
        __syncthreads();
        if (thread_idx == 0) {
            __threadfence();
            atomicAdd(lock_ptr + flag_idx, 1);
        }
    }
};

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          bool Has_softcap_ = false, bool Deterministic_ = false, bool ReduceKV_ = false,
          DispatchDirection kDir_ = DispatchDirection::MaxToMin,
          bool BwdInnerLoopK_ = false, bool Is_dropout_ = false>
class FlashAttnFlexFlashBwdSm80 {

public:

    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Deterministic = Deterministic_;
    static constexpr bool ReduceKV = ReduceKV_;
    static constexpr DispatchDirection kDir = kDir_;
    static constexpr bool BwdInnerLoopK = BwdInnerLoopK_;
    static constexpr bool Is_dropout = Is_dropout_;
    static_assert(!ReduceKV || BwdInnerLoopK, "ReduceKV requires the LoopK mainloop");
    // Deterministic LoopQ forces ascending m iteration (deadlock-free turns).
    static constexpr DispatchDirection kDirEff =
        (Deterministic && !BwdInnerLoopK) ? DispatchDirection::MinToMax : kDir;

    // ─── TYPE PROVIDER ALIASES ───────────────────────────────────────────────
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    using TileScheduler = TileScheduler_;

    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TiledMmaSdP   = typename CollectiveMainloop::TiledMmaSdP;
    using TiledMmadKV   = typename CollectiveMainloop::TiledMmadKV;
    using TiledMmadQ    = typename CollectiveMainloop::TiledMmadQ;
    using ArchTag       = typename CollectiveMainloop::ArchTag;
    using Element       = typename CollectiveMainloop::Element;
    using ElementAccum  = float;

    static constexpr bool SdP_swapAB = CollectiveMainloop::SdP_swapAB;
    static constexpr bool dKV_swapAB = CollectiveMainloop::dKV_swapAB;
    static constexpr bool dQ_swapAB  = CollectiveMainloop::dQ_swapAB;
    static constexpr bool Mma_dKV_is_RS = CollectiveMainloop::Mma_dKV_is_RS;
    // The mainloop exposes V_in_regs only as Share_QV_Smem.
    static constexpr bool V_in_regs  = CollectiveMainloop::Share_QV_Smem;
    // Note: `V_in_regs && BwdInnerLoopK` cannot be static_assert'ed here —
    // V_in_regs comes from the still-uninstantiated mainloop, so its
    // initializer is not a constant expression at this point.  The launch
    // template discards that combination via `if constexpr`, and the C++
    // API rejects it at runtime (see flex_flash_attention_api.cpp).
    static constexpr int kStages     = CollectiveMainloop::kStages;
    static constexpr int kStages_dO  = CollectiveMainloop::kStages_dO;
    static constexpr bool Q_dO_same_stages = CollectiveMainloop::Q_dO_same_stages;
    static constexpr bool ShuffleLSE   = CollectiveMainloop::ShuffleLSE;
    static constexpr bool ShuffledPsum = CollectiveMainloop::ShuffledPsum;

    static constexpr int kBlockM   = get<0>(TileShape_MNK{});
    static constexpr int kBlockN   = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim  = get<2>(TileShape_MNK{});
    static constexpr int kBlockKGmem = CollectiveMainloop::kBlockKGmem;

    static_assert(ArchTag::kMinComputeCapability >= 80);

    // Flex flash attention path: no compile-time causal/local, no varlen.
    static constexpr bool Is_causal = false;
    static constexpr bool Is_local  = false;
    static constexpr bool Varlen    = false;

    using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;

    // Epilogue derived types.
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams    = typename CollectiveEpilogue::Params;

    using TileSchedulerArguments = typename flash::TileSchedulerArguments;
    using TileSchedulerParams    = typename TileScheduler::Params;

    static constexpr uint32_t NumThreads         = CUTE_STATIC_V(size(TiledMmaSdP{}));
    static constexpr uint32_t MaxThreadsPerBlock = NumThreads;
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
    static constexpr int NumProducerThreads = CollectiveMainloop::NumMmaThreads;

    // ─── SHARED MEMORY STORAGE ────────────────────────────────────────────────
    struct SharedStorage {
        struct CUTE_ALIGNAS(128) TensorStorage {
            union {
                typename CollectiveMainloop::TensorStorage mainloop;
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        } tensors;

        alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // ─── HOST-SIDE ARGUMENTS & DEVICE-SIDE PARAMS ────────────────────────────
    using DropoutArgs = flash::DropoutArbArgs;

    struct Arguments {
        typename CollectiveMainloop::Arguments mainloop{};
        EpilogueArguments epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
        AttnSliceParams slices{};
        // ReduceKV only: per-element first-write flags for the dK/dV
        // workspaces (one shared array for dK and dV; size = workspace elems).
        int* dkv_flags = nullptr;
        DropoutArgs dropout{};
    };

    struct Params {
        typename CollectiveMainloop::Params mainloop{};
        EpilogueParams epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
        AttnSliceParams slices{};
        int* dkv_flags = nullptr;
        DropoutArgs dropout{};
    };

    static Params to_underlying_arguments(Arguments const& args) {
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) {
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
                args.hw_info.device_id);
        }
        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            hw_info,
            TileScheduler::to_underlying_arguments(args.scheduler),
            args.slices,
            args.dkv_flags,
            args.dropout
        };
    }

    static dim3 get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
    }

    static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

    // ─── KERNEL ENTRY POINT ──────────────────────────────────────────────────
    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf) {

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        CollectiveEpilogue epilogue;
        TileScheduler scheduler(
            reinterpret_cast<typename TileScheduler::SharedStorage*>(
                &shared_storage.smem_scheduler));

        TiledMmadKV tiled_mma_dKV;
        TiledMmadQ  tiled_mma_dQ;
        scheduler.init_consumer();

        int const warp_idx   = cutlass::canonical_warp_idx_sync();
        int const thread_idx = threadIdx.x;

        for (auto work_tile_info = warp_idx == 0
                 ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler)
                 : scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
             work_tile_info.is_valid(params.scheduler);
             work_tile_info = warp_idx == 0
                 ? scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info)
                 : scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info))
        {
            auto block_coord_ = work_tile_info.get_block_coord(params.scheduler);
            auto [block_idx, bidh, bidb, _ /*split_idx*/] = block_coord_;

            if constexpr (!BwdInnerLoopK) {
                // ── LoopQ: CTA = (n_block, h, b) ──
                int const n_block = block_idx;
                cute::tuple<int32_t, int32_t, int32_t> block_coord = {n_block, bidh, bidb};

                // dK and dV output accumulator (register resident, accumulated
                // across all covering slices, stored once at the end).
                Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV,
                    select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
                Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV,
                    select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));

                bool tile_valid = run_mainloop(params, shared_storage, tdKrdK, tdVrdV,
                                               thread_idx, n_block, bidh, bidb);

                scheduler.prefetch_next_work(params.scheduler, work_tile_info);

                if constexpr (Deterministic) {
                    // dK/dV were flushed per-slice through the semaphores;
                    // the accum buffers are zero-initialized host-side.
                } else {
                    if (tile_valid) {
                        epilogue.store(params.epilogue, tdKrdK, tdVrdV, shared_storage,
                                       tiled_mma_dKV, thread_idx, block_coord);
                    } else {
                        epilogue.store_zero(params.epilogue, thread_idx, block_coord);
                    }
                }
            } else {
                // ── LoopK: CTA = (m_block, h, b) ──
                int const m_block = block_idx;

                // dQ accumulator: register-resident across ALL slices, one
                // atomicAdd at the end (bitwise deterministic by construction).
                Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ,
                    select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));

                run_mainloop_k(params, shared_storage, tdQrdQ,
                               thread_idx, m_block, bidh, bidb);

                scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                // Empty tiles: dQaccum is zero-cleared by the preprocess
                // kernel and this is the only CTA writing this (m_block),
                // so there is nothing to zero-fill.
            }
        }
    }

    // ─── SLICE-AWARE BWD MAINLOOP (LoopQ) ────────────────────────────────────
    template <typename FrgTensordKV>
    CUTLASS_DEVICE bool run_mainloop(
        Params const& params,
        SharedStorage& shared_storage,
        FrgTensordKV& tdKrdK,
        FrgTensordKV& tdVrdV,
        int thread_idx,
        int n_block, int bidh, int bidb)
    {
        static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

        using MainloopParams = typename CollectiveMainloop::Params;
        MainloopParams const& ml = params.mainloop;

        SeqlenInfo_t seqlen_info{
            bidb, get<0>(ml.shape_Q), size<0>(ml.shape_K),
            ml.cu_seqlens_q, ml.cu_seqlens_k, ml.seqused_q, ml.seqused_k
        };
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = size<0>(ml.shape_K);

        // Dropout stream resolve (CUDA-graph-safe; see dropout_flex_flash.h):
        // graph mode reads the (seed, offset) pair published by the captured
        // fwd graph from device memory, replaying the identical Philox
        // stream; eager keeps the by-value scalars baked at launch.
        unsigned long long rng_seed_eff = 0, rng_offset_base_eff = 0;
        if constexpr (Is_dropout) {
            arb_resolve_rng_stream(params.dropout, rng_seed_eff,
                                   rng_offset_base_eff);
        }

        // P3 per-(b,h) heterogeneous masks: rebase the slice layout onto
        // this (batch, head) pair's layout group.  Shared layout
        // (bh_to_group == nullptr) → zero-cost identity copy.
        // shape_Q = (seqlen_q, d, h, b): num_heads from get<2> (always
        // populated, unlike dropout.num_heads which is Is_dropout-only).
        AttnSliceParams const sl = rebase_slices(
            params.slices,
            bidb * cute::get<2>(ml.shape_Q) + bidh, seqlen_q);

        // ── Shared memory tensors ──
        Tensor sQ   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
                                  typename CollectiveMainloop::SmemLayoutQ{});
        Tensor sdO  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
                                  typename CollectiveMainloop::SmemLayoutdO{});
        Tensor sK   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
                                  typename CollectiveMainloop::SmemLayoutK{});
        Tensor sV   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
                                  typename CollectiveMainloop::SmemLayoutV{});
        Tensor sQt  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
                                  typename CollectiveMainloop::SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
                                  typename CollectiveMainloop::SmemLayoutdOt{});
        Tensor sKt  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
                                  typename CollectiveMainloop::SmemLayoutKt{});
        Tensor sP   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
                                  typename CollectiveMainloop::SmemLayoutPdS{});
        Tensor sPt  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
                                  typename CollectiveMainloop::SmemLayoutPdSt{});
        Tensor sdS  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
                                  typename CollectiveMainloop::SmemLayoutPdS{});
        Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
                                  typename CollectiveMainloop::SmemLayoutPdSt{});
        // R2S copy view: AIU_LOAD writes 8x64 tiles in row-major (LayoutRight) order.
        // The compute path still uses sQ/sdO/sK/sV for partition_S; the PPU copy
        // atoms compute smem addresses internally so both views share the same
        // physical buffer. Only referenced on the Use_CVT_SWZL_LD path.
        // ISOLATED fp32 kernel set: the LayoutRight views exist for every
        // element type, so they are declared unconditionally (float never
        // references them since Use_CVT_SWZL_LD is false there).
        Tensor sQ_AIU_COPY  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
                                          typename CollectiveMainloop::SmemLayoutQ_AIU_COPY{});
        Tensor sdO_AIU_COPY = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
                                          typename CollectiveMainloop::SmemLayoutdO_AIU_COPY{});
        Tensor sK_AIU_COPY  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
                                          typename CollectiveMainloop::SmemLayoutK_AIU_COPY{});
        Tensor sV_AIU_COPY  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
                                          typename CollectiveMainloop::SmemLayoutV_AIU_COPY{});
#if PPU1v0_R2S_SLICE_LAYOUT || PPU1v5_R2S_SLICE_LAYOUT
        // Slice-format views of the P/dS smem buffers (written via r2s with the
        // slice layout, read back via the AIU TSM load atoms).
        Tensor sdS_slice  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
                                        typename CollectiveMainloop::SmemLayoutPdS_R2Slice{});
        Tensor sdSt_slice = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
                                        typename CollectiveMainloop::SmemLayoutPdSt_R2Slice{});
        Tensor sP_slice   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
                                        typename CollectiveMainloop::SmemLayoutPdS_R2Slice{});
        Tensor sPt_slice  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
                                        typename CollectiveMainloop::SmemLayoutPdSt_R2Slice{});
#endif
        Tensor sLSE    = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()),
                                     typename CollectiveMainloop::SmemLayoutLSE{});
        Tensor sdPsum  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()),
                                     typename CollectiveMainloop::SmemLayoutLSE{});
        Tensor sLSEMma   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()),
                                       typename CollectiveMainloop::SmemLayoutLSEMma{});
        Tensor sdPsumMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()),
                                       typename CollectiveMainloop::SmemLayoutLSEMma{});

        // ── Gmem tensors ──
        int const bidh_kv = ml.qhead_per_khead_divmod.divide(bidh);
        Tensor mQ   = make_tensor(make_gmem_ptr(ml.ptr_Q),   ml.shape_Q,   ml.stride_Q)(_, _, bidh, bidb);
        Tensor mdO  = make_tensor(make_gmem_ptr(ml.ptr_dO),  ml.shape_dO,  ml.stride_dO)(_, _, bidh, bidb);
        Tensor mK   = make_tensor(make_gmem_ptr(ml.ptr_K),   ml.shape_K,   ml.stride_K)(_, _, bidh_kv, bidb);
        Tensor mV   = make_tensor(make_gmem_ptr(ml.ptr_V),   ml.shape_V,   ml.stride_V)(_, _, bidh_kv, bidb);
        Tensor mLSE   = make_tensor(make_gmem_ptr(ml.ptr_LSE_log2), ml.shape_LSE, ml.stride_LSE_log2)(_, bidh, bidb);
        Tensor mdPsum = make_tensor(make_gmem_ptr(ml.ptr_dPsum),    ml.shape_LSE, ml.stride_dPsum)(_, bidh, bidb);
        Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(ml.ptr_dQaccum)),
                                      ml.shape_dQaccum, ml.stride_dQaccum)(_, bidh, bidb);

        // ISOLATED fp32 kernel set: mix tensors are only valid for the AIU
        // copy atoms; float's stock CP_ASYNC path tiles the raw gmem views
        // (mirrors flex_flash_attention_fwd_kernel_sm80.h).
        Tensor gQ = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return local_tile(make_mix_tensor_like(mQ),  select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
            } else {
                return local_tile(mQ,  select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
            }
        }();
        Tensor gdO = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return local_tile(make_mix_tensor_like(mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
            } else {
                return local_tile(mdO, select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
            }
        }();
        Tensor gK = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return local_tile(make_mix_tensor_like(mK),  select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
            } else {
                return local_tile(mK,  select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
            }
        }();
        Tensor gV = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return local_tile(make_mix_tensor_like(mV),  select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
            } else {
                return local_tile(mV,  select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
            }
        }();
        Tensor gLSE   = local_tile(mLSE,   select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdPsum = local_tile(mdPsum, select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdQaccum = local_tile(mdQaccum, Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)

        // ── Gmem tiled copies ──
        typename CollectiveMainloop::GmemTiledCopyQ  gmem_tiled_copy_Q;
        typename CollectiveMainloop::GmemTiledCopyKV gmem_tiled_copy_K;
        typename CollectiveMainloop::GmemTiledCopyKV gmem_tiled_copy_V;
        typename CollectiveMainloop::GmemTiledCopyQ  gmem_tiled_copy_dO;

#if defined(USE_PPU) && USE_AIU
        // PPU AIU descriptor initialization (required before any AIU copy).
        // ISOLATED fp32 kernel set: only 16-bit instantiations own an AIU
        // descriptor; float's stock CP_ASYNC tiled copy has no desc_.
        if constexpr (CollectiveMainloop::Use_aiu) {
        if constexpr (ArchTag::kMinComputeCapability >= 89) {
            if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                // CVT path: 8x64 tile desc for direct AIU copy loops.
                // gmem block offset is handled by the direct AIU copy loop.
                static constexpr int kBlockNPerAiuLoad = 8;
                gmem_tiled_copy_Q.desc_.init(nullptr, kBlockNPerAiuLoad, kBlockKGmem, get<0>(ml.stride_Q));
                gmem_tiled_copy_K.desc_.init(nullptr, kBlockNPerAiuLoad, CollectiveMainloop::kBlockNGmem, get<0>(ml.stride_K));
                gmem_tiled_copy_V.desc_.init(nullptr, kBlockNPerAiuLoad, CollectiveMainloop::kBlockNGmem, get<0>(ml.stride_V));
                gmem_tiled_copy_dO.desc_.init(nullptr, kBlockNPerAiuLoad, kBlockKGmem, get<0>(ml.stride_dO));
            } else {
                gmem_tiled_copy_Q.desc_.init(nullptr, seqlen_info.seqlen_q, get<1>(ml.shape_Q), get<0>(ml.stride_Q));
                gmem_tiled_copy_K.desc_.init(nullptr, seqlen_info.seqlen_k, get<1>(ml.shape_Q), get<0>(ml.stride_K));
                gmem_tiled_copy_V.desc_.init(nullptr, seqlen_info.seqlen_k, get<1>(ml.shape_V), get<0>(ml.stride_V));
                gmem_tiled_copy_dO.desc_.init(nullptr, seqlen_info.seqlen_q, get<1>(ml.shape_V), get<0>(ml.stride_dO));
            }
        } else {
            int aiu_offset_q = get<1>(ml.shape_Q) == kHeadDim ? 0 : (get<0>(ml.stride_Q) - get<1>(ml.shape_Q));
            int aiu_offset_k = get<1>(ml.shape_Q) == kHeadDim ? 0 : (get<0>(ml.stride_K) - get<1>(ml.shape_Q));
            int aiu_offset_v = get<1>(ml.shape_V) == kHeadDim ? 0 : (get<0>(ml.stride_V) - get<1>(ml.shape_V));
            int aiu_offset_do = get<1>(ml.shape_V) == kHeadDim ? 0 : (get<0>(ml.stride_dO) - get<1>(ml.shape_V));
            using AiuDesc = decltype(gmem_tiled_copy_Q.desc_);
            gmem_tiled_copy_Q.desc_  = AiuDesc{nullptr, seqlen_info.seqlen_q, get<0>(ml.stride_Q),  kBlockM, kBlockKGmem, aiu_offset_q};
            gmem_tiled_copy_K.desc_  = AiuDesc{nullptr, seqlen_info.seqlen_k, get<0>(ml.stride_K),  kBlockN, kBlockKGmem, aiu_offset_k};
            gmem_tiled_copy_V.desc_  = AiuDesc{nullptr, seqlen_info.seqlen_k, get<0>(ml.stride_V),  kBlockN, kBlockKGmem, aiu_offset_v};
            gmem_tiled_copy_dO.desc_ = AiuDesc{nullptr, seqlen_info.seqlen_q, get<0>(ml.stride_dO), kBlockM, kBlockKGmem, aiu_offset_do};
        }
        }
        const int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
        const int tid_thread_slice = CollectiveMainloop::Use_aiu ? warp_idx * 32 : thread_idx;
#if PPU1v0_R2S_SLICE_LAYOUT
        auto r2S_thread_idx_PdS = thread_idx;
        if (warp_idx % 2) { // keep slice0 layout
            r2S_thread_idx_PdS = __shfl_sync(0xffffffff, r2S_thread_idx_PdS, (r2S_thread_idx_PdS & 31) ^ CollectiveMainloop::PPUChannelSliceCount); // simulate slice1 layout
        }
#elif PPU1v5_R2S_SLICE_LAYOUT
        auto r2S_thread_idx_PdS = thread_idx;
#endif
#else
        const int tid_thread_slice = thread_idx;
#endif

        auto gmem_thr_copy_Q  = gmem_tiled_copy_Q.get_thread_slice(thread_idx);
        auto gmem_thr_copy_K  = gmem_tiled_copy_K.get_thread_slice(thread_idx);
        auto gmem_thr_copy_V  = gmem_tiled_copy_V.get_thread_slice(thread_idx);
        auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_Q  = gmem_tiled_copy_Q.get_thread_slice(_0{});
        auto gmem_thr0_copy_KV = gmem_tiled_copy_K.get_thread_slice(_0{});
        typename CollectiveMainloop::GmemTiledCopyLSE gmem_tiled_copy_lse;
        auto gmem_thr_copy_lse = gmem_tiled_copy_lse.get_thread_slice(thread_idx);
        typename CollectiveMainloop::R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
        auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);

        Tensor tQgQ   = gmem_thr_copy_Q.partition_S(gQ);
        Tensor tQsQ   = gmem_thr_copy_Q.partition_D(sQ);
        Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
        Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
        Tensor tLSEgLSE   = gmem_thr_copy_lse.partition_S(gLSE);
        Tensor tLSEsLSE   = gmem_thr_copy_lse.partition_D(sLSE);
        Tensor tLSEgdPsum = gmem_thr_copy_lse.partition_S(gdPsum);
        Tensor tLSEsdPsum = gmem_thr_copy_lse.partition_D(sdPsum);
        Tensor tdQgdQaccum = r2s_thr_copy_dQaccum.partition_D(gdQaccum);

        TiledMmaSdP tiled_mma_SdP;
        TiledMmadKV tiled_mma_dKV;
        TiledMmadQ  tiled_mma_dQ;

        auto thr_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto thr_mma_dKV = tiled_mma_dKV.get_thread_slice(thread_idx);
        auto thr_mma_dQ  = tiled_mma_dQ.get_thread_slice(thread_idx);

        Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(thr_mma_SdP, sV);

        // Copy Atom retiling
        auto smem_copy_atom_SdP_B = cute::conditional_return<CollectiveMainloop::MmaSdPEvenN>(
            typename CollectiveMainloop::SmemCopyAtom{}, typename CollectiveMainloop::SmemCopyAtomHalf{});
#if defined(USE_PPU) && USE_AIU
        auto smem_tiled_copy_QdO = cute::conditional_return<!SdP_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomQ{}, tiled_mma_SdP),
            make_tiled_copy_B(typename CollectiveMainloop::SmemCopyAtomQ{}, tiled_mma_SdP));
#else
        auto smem_tiled_copy_QdO = cute::conditional_return<!SdP_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtom{}, tiled_mma_SdP),
            make_tiled_copy_B(smem_copy_atom_SdP_B, tiled_mma_SdP));
#endif
        auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tid_thread_slice);
        Tensor tSsQ = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_QdO.partition_S(make_mix_tensor_like(sQ)); }
            else { return smem_thr_copy_QdO.partition_S(sQ); }
        }();
        Tensor tdPsdO = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_QdO.partition_S(make_mix_tensor_like(sdO)); }
            else { return smem_thr_copy_QdO.partition_S(sdO); }
        }();

#if defined(USE_PPU) && USE_AIU
        auto smem_tiled_copy_KV = cute::conditional_return<!SdP_swapAB>(
            make_tiled_copy_B(typename CollectiveMainloop::SmemCopyAtomK{}, tiled_mma_SdP),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomK{}, tiled_mma_SdP));
#else
        auto smem_tiled_copy_KV = cute::conditional_return<!SdP_swapAB>(
            make_tiled_copy_B(smem_copy_atom_SdP_B, tiled_mma_SdP),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtom{}, tiled_mma_SdP));
#endif
        auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tid_thread_slice);
        Tensor tSsK = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_KV.partition_S(make_mix_tensor_like(sK)); }
            else { return smem_thr_copy_KV.partition_S(sK); }
        }();
        Tensor tdPsV = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_KV.partition_S(make_mix_tensor_like(sV)); }
            else { return smem_thr_copy_KV.partition_S(sV); }
        }();

        auto r2s_tiled_copy_PdS = make_tiled_copy_C(typename CollectiveMainloop::R2SCopyAtomPdS{}, tiled_mma_SdP);
#if PPU1v0_R2S_SLICE_LAYOUT || PPU1v5_R2S_SLICE_LAYOUT
        auto r2s_thr_copy_PdS = r2s_tiled_copy_PdS.get_thread_slice(r2S_thread_idx_PdS);
        // ISOLATED fp32 kernel set: the slice-format P/dS views are the AIU
        // TSM read-back geometry; float writes/reads the standard swizzled
        // PdS layout instead (both sides must agree on the layout).
        Tensor tdSsdS = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS_slice, sdSt_slice));
            } else {
                return r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS, sdSt));
            }
        }();
        Tensor tPsP = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP_slice, sPt_slice));
            } else {
                return r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP, sPt));
            }
        }();
#else
        auto r2s_thr_copy_PdS = r2s_tiled_copy_PdS.get_thread_slice(thread_idx);
        Tensor tdSsdS = r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS, sdSt));
        Tensor tPsP   = r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP, sPt));
#endif

#if PPU1v0_R2S_SLICE_LAYOUT || PPU1v5_R2S_SLICE_LAYOUT
        auto smem_copy_atom_dKV_B = cute::conditional_return<CollectiveMainloop::MmadKVEvenN>(
            typename CollectiveMainloop::SmemCopyAtomPdSt_AIU{}, typename CollectiveMainloop::SmemCopyAtomTransposedHalf{});
        auto smem_tiled_copy_PdSt = cute::conditional_return<!dKV_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomPdSt_AIU{}, tiled_mma_dKV),
            make_tiled_copy_B(smem_copy_atom_dKV_B, tiled_mma_dKV));
        auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tid_thread_slice);
        Tensor tdVsPt = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_PdSt.partition_S(make_mix_tensor_like(sPt)); }
            else { return smem_thr_copy_PdSt.partition_S(sPt); }
        }();
        Tensor tdKsdSt = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_PdSt.partition_S(make_mix_tensor_like(sdSt)); }
            else { return smem_thr_copy_PdSt.partition_S(sdSt); }
        }();
#else
        auto smem_copy_atom_dKV_B = cute::conditional_return<CollectiveMainloop::MmadKVEvenN>(
            typename CollectiveMainloop::SmemCopyAtomTransposed{}, typename CollectiveMainloop::SmemCopyAtomTransposedHalf{});
        auto smem_tiled_copy_PdSt = cute::conditional_return<!dKV_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomTransposed{}, tiled_mma_dKV),
            make_tiled_copy_B(smem_copy_atom_dKV_B, tiled_mma_dKV));
        auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(thread_idx);
        Tensor tdVsPt  = smem_thr_copy_PdSt.partition_S(sPt);
        Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);
#endif

#if defined(USE_PPU) && USE_AIU
        auto smem_tiled_copy_QdOt = cute::conditional_return<!dKV_swapAB>(
            make_tiled_copy_B(typename CollectiveMainloop::SmemCopyAtomQt{}, tiled_mma_dKV),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomQt{}, tiled_mma_dKV));
#else
        auto smem_tiled_copy_QdOt = cute::conditional_return<!dKV_swapAB>(
            make_tiled_copy_B(smem_copy_atom_dKV_B, tiled_mma_dKV),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomTransposed{}, tiled_mma_dKV));
#endif
        auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tid_thread_slice);
        Tensor tdVsdOt = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_QdOt.partition_S(make_mix_tensor_like(sdOt)); }
            else { return smem_thr_copy_QdOt.partition_S(sdOt); }
        }();
        Tensor tdKsQt = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_QdOt.partition_S(make_mix_tensor_like(sQt)); }
            else { return smem_thr_copy_QdOt.partition_S(sQt); }
        }();

#if PPU1v0_R2S_SLICE_LAYOUT || PPU1v5_R2S_SLICE_LAYOUT
        auto smem_tiled_copy_dS = cute::conditional_return<!dQ_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomdS_AIU{}, tiled_mma_dQ),
            make_tiled_copy_B(cute::conditional_return<CollectiveMainloop::MmadQEvenN>(
                typename CollectiveMainloop::SmemCopyAtom{}, typename CollectiveMainloop::SmemCopyAtomHalf{}), tiled_mma_dQ));
        auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tid_thread_slice);
        Tensor tdQsdS = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_dS.partition_S(make_mix_tensor_like(sdS)); }
            else { return smem_thr_copy_dS.partition_S(sdS); }
        }();
#else
        // Non-slice path: PdS/dS live in the standard swizzled smem layout and
        // are read back with plain LDSM atoms (no mix tensor).
        auto smem_tiled_copy_dS = cute::conditional_return<!dQ_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtom{}, tiled_mma_dQ),
            make_tiled_copy_B(cute::conditional_return<CollectiveMainloop::MmadQEvenN>(
                typename CollectiveMainloop::SmemCopyAtom{}, typename CollectiveMainloop::SmemCopyAtomHalf{}), tiled_mma_dQ));
        auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(thread_idx);
        Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);
#endif

#if defined(USE_PPU) && USE_AIU
        auto smem_tiled_copy_Kt = cute::conditional_return<!dQ_swapAB>(
            make_tiled_copy_B(typename CollectiveMainloop::SmemCopyAtomKVt{}, tiled_mma_dQ),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomKVt{}, tiled_mma_dQ));
#else
        auto smem_tiled_copy_Kt = cute::conditional_return<!dQ_swapAB>(
            make_tiled_copy_B(cute::conditional_return<CollectiveMainloop::MmadQEvenN>(
                typename CollectiveMainloop::SmemCopyAtomTransposed{}, typename CollectiveMainloop::SmemCopyAtomTransposedHalf{}), tiled_mma_dQ),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomTransposed{}, tiled_mma_dQ));
#endif
        auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tid_thread_slice);
        Tensor tdQsKt = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_Kt.partition_S(make_mix_tensor_like(sKt)); }
            else { return smem_thr_copy_Kt.partition_S(sKt); }
        }();

        // LSE / dPsum smem→reg partitioning.
        Tensor tSsLSEMma = [&]() -> auto {
#ifdef USE_PPU
            if constexpr (ArchTag::kMinComputeCapability >= 89) {
                return group_modes<0, 2>(logical_divide(thr_mma_SdP.partition_C(sLSEMma), Shape<_2>{})(make_coord(_, make_coord(_, _0{})), _, _, _));
            } else {
                return logical_divide(thr_mma_SdP.partition_C(sLSEMma), Shape<_4>{});
            }
#else
            return logical_divide(thr_mma_SdP.partition_C(sLSEMma), Shape<_2>{});
#endif
        }();
        Tensor tSsLSE = group_modes<0, 2>(cute::conditional_return<!SdP_swapAB>(
            tSsLSEMma(make_coord(_0{}, _), _, _0{}, _),   // (2, MMA_M, PIPE)
            tSsLSEMma(make_coord(_, _0{}), _0{}, _, _))); // (2, MMA_N, PIPE)
        Tensor tSsdPsumMma = [&]() -> auto {
#ifdef USE_PPU
            if constexpr (ArchTag::kMinComputeCapability >= 89) {
                return group_modes<0, 2>(logical_divide(thr_mma_SdP.partition_C(sdPsumMma), Shape<_2>{})(make_coord(_, make_coord(_, _0{})), _, _, _));
            } else {
                return logical_divide(thr_mma_SdP.partition_C(sdPsumMma), Shape<_4>{});
            }
#else
            return logical_divide(thr_mma_SdP.partition_C(sdPsumMma), Shape<_2>{});
#endif
        }();
        Tensor tSsdPsum = group_modes<0, 2>(cute::conditional_return<!SdP_swapAB>(
            tSsdPsumMma(make_coord(_0{}, _), _, _0{}, _),
            tSsdPsumMma(make_coord(_, _0{}), _0{}, _, _)));
        static constexpr int kStatsPerThread = cute::ceil_div(decltype(size(tSsLSE))::value, 8);

        // ── Predicates ──
        Tensor cQ = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
        Tensor tQcQ  = gmem_thr_copy_Q.partition_S(cQ);
        Tensor t0QcQ = gmem_thr0_copy_Q.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(ml.shape_Q); }
        Tensor cLSE = cute::make_identity_tensor(select<0>(TileShape_MNK{}));
        Tensor tLSEcLSE = gmem_thr_copy_lse.partition_S(cLSE);
        Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdOsdO)));
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(ml.shape_dO); }

        static constexpr bool EvenM = kBlockM % CUTE_STATIC_V(shape<0>(typename CollectiveMainloop::GmemLayoutAtom{})) == 0;

        // ── Load K/V once per work tile (they are fixed for this n_block) ──
        {
            Tensor tKgK = gmem_thr_copy_K.partition_S(gK);
            Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
            Tensor tVgV = gmem_thr_copy_V.partition_S(gV);
            Tensor tVsV = gmem_thr_copy_V.partition_D(sV);
            Tensor cKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
            Tensor tKVcKV  = gmem_thr_copy_K.partition_S(cKV);
            Tensor t0KVcKV = gmem_thr0_copy_KV.partition_S(cKV);
            Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));
            Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tVsV)));
            #pragma unroll
            for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(ml.shape_K); }
            #pragma unroll
            for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(ml.shape_V); }
            static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(typename CollectiveMainloop::GmemLayoutAtom{})) == 0;
            int const seqlenk_row_limit = seqlen_k - n_block * kBlockN - get<0>(tKVcKV(_0{}, _0{}, _0{}));
#if defined(USE_PPU) && USE_AIU
            if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for V.
                // Hierarchical tiling: 8x64 -> 8x128 (horizontal x2) -> blockN x128 (vertical)
                //                      -> blockN x headdim (horizontal)
                // offset(n,k) = (k/2)*kHalfSize + n*kTile8x128 + (k%2)*kAtomSize
                const Element* block_v_ptr = mV.data().get()
                    + (seqlen_info.offset_k + n_block * kBlockN) * get<0>(ml.stride_V);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumNTiles = kBlockN / kAiuTileH;
                static constexpr int kTotalTiles = kNumNTiles * kNumKTilesTotal;
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile8x128 = kAiuTileH * 2 * kAiuTileW;             // 8x128 = 1024
                static constexpr int kHalfSize = kBlockN * 2 * kAiuTileW;                // blockN x 128
                static constexpr int kKTileLog = (kNumKTilesTotal == 4) ? 2
                                               : (kNumKTilesTotal == 2) ? 1 : 0;         // log2(kNumKTilesTotal)
                int const aiu_gmem_row_step = kAiuTileH * get<0>(ml.stride_V);           // 8-row gmem span
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += CollectiveMainloop::NumMmaWarps) {
                    // kNumKTilesTotal ∈ {1,2,4}: tile decode via shift/and.
                    int const n = tile_linear >> kKTileLog;
                    int const k = tile_linear & (kNumKTilesTotal - 1);
                    const Element* tile_gmem = block_v_ptr
                        + n * aiu_gmem_row_step
                        + k * kBlockKGmem;
                    auto tile_smem = sV_AIU_COPY.data()
                        + (k >> 1)*kHalfSize + n*kTile8x128 + (k & 1)*kAtomSize;
                    // dim_h must bound the seqlen tail: AIU zero-fills rows past
                    // dim_h, keeping padded tail rows exact-0 (garbage rows would
                    // carry inf/nan into dS and poison dK/dV).
                    int const valid_rows = cute::min(kAiuTileH, cute::max(0,
                        seqlen_info.seqlen_k - (seqlen_info.offset_k + n_block * kBlockN + n * kAiuTileH)));
                    if (valid_rows > 0) {
                        gmem_tiled_copy_V.desc_.dim_h = valid_rows;
                        CollectiveMainloop::Gmem_copy_struct_KV::copy(
                            cute::raw_pointer_cast(tile_smem),
                            tile_gmem,
                            gmem_tiled_copy_V.desc_,
                            0, 0);
                    } else {
                        uint32_t* zp = reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(tile_smem));
                        #pragma unroll
                        for (int i = threadIdx.x & 31; i < int(kAtomSize * sizeof(Element) / 4); i += 32) { zp[i] = 0u; }
                    }
                }
            } else {
                // Stock fp32 path: predicate rows past seqlen_k and zero-fill
                // them in smem (cp.async zfill), mirroring the AIU dim_h tail
                // zero-fill — stale gmem bits there can be NaN/inf and poison
                // dS/dK/dV through the 0*inf neutralization gap.
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_V, tVgV, tVsV, t0KVcKV, tVpV, seqlenk_row_limit);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tVsV); ++m) {
                if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
                    bool const predicate_n = get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tVsV); ++k) {
                        cute::copy(gmem_tiled_copy_V.with(tVpV(k) && predicate_n), tVgV(_, m, k), tVsV(_, m, k));
                    }
                }
            }
#endif
            if constexpr (V_in_regs) { flash::cp_async_fence(); }
#if defined(USE_PPU) && USE_AIU
            if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for K.
                // Hierarchical tiling: 8x64 -> 8x128 (horizontal x2) -> blockN x128 (vertical)
                //                      -> blockN x headdim (horizontal)
                // offset(n,k) = (k/2)*kHalfSize + n*kTile8x128 + (k%2)*kAtomSize
                const Element* block_k_ptr = mK.data().get()
                    + (seqlen_info.offset_k + n_block * kBlockN) * get<0>(ml.stride_K);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumNTiles = kBlockN / kAiuTileH;
                static constexpr int kTotalTiles = kNumNTiles * kNumKTilesTotal;
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile8x128 = kAiuTileH * 2 * kAiuTileW;             // 8x128 = 1024
                static constexpr int kHalfSize = kBlockN * 2 * kAiuTileW;                // blockN x 128
                static constexpr int kKTileLog = (kNumKTilesTotal == 4) ? 2
                                               : (kNumKTilesTotal == 2) ? 1 : 0;         // log2(kNumKTilesTotal)
                int const aiu_gmem_row_step = kAiuTileH * get<0>(ml.stride_K);           // 8-row gmem span
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += CollectiveMainloop::NumMmaWarps) {
                    // kNumKTilesTotal ∈ {1,2,4}: tile decode via shift/and.
                    int const n = tile_linear >> kKTileLog;
                    int const k = tile_linear & (kNumKTilesTotal - 1);
                    const Element* tile_gmem = block_k_ptr
                        + n * aiu_gmem_row_step
                        + k * kBlockKGmem;
                    auto tile_smem = sK_AIU_COPY.data()
                        + (k >> 1)*kHalfSize + n*kTile8x128 + (k & 1)*kAtomSize;
                    // dim_h must bound the seqlen tail (see the V copy above).
                    int const valid_rows = cute::min(kAiuTileH, cute::max(0,
                        seqlen_info.seqlen_k - (seqlen_info.offset_k + n_block * kBlockN + n * kAiuTileH)));
                    if (valid_rows > 0) {
                        gmem_tiled_copy_K.desc_.dim_h = valid_rows;
                        CollectiveMainloop::Gmem_copy_struct_KV::copy(
                            cute::raw_pointer_cast(tile_smem),
                            tile_gmem,
                            gmem_tiled_copy_K.desc_,
                            0, 0);
                    } else {
                        uint32_t* zp = reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(tile_smem));
                        #pragma unroll
                        for (int i = threadIdx.x & 31; i < int(kAtomSize * sizeof(Element) / 4); i += 32) { zp[i] = 0u; }
                    }
                }
            } else {
                // Stock fp32 path: predicate rows past seqlen_k and zero-fill
                // them in smem (see the V copy above).
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_K, tKgK, tKsK, t0KVcKV, tKpK, seqlenk_row_limit);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tKsK); ++m) {
                if (EvenN || m < size<1>(tKsK) - 1 || get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
                    bool const predicate_n = get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tKsK); ++k) {
                        cute::copy(gmem_tiled_copy_K.with(tKpK(k) && predicate_n), tKgK(_, m, k), tKsK(_, m, k));
                    }
                }
            }
#endif
            flash::cp_async_fence();
        }

        if constexpr (V_in_regs) {
            flash::cp_async_wait<1>();
            __syncthreads();
            Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
            Tensor tdPsV_copy_view = [&]() -> auto {
                if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_KV.partition_S(make_mix_tensor_like(sV)); }
                else { return smem_thr_copy_KV.partition_S(sV); }
            }();
            cute::copy(smem_tiled_copy_KV, tdPsV_copy_view, tdPrV_copy_view);
            __syncthreads();
        }

        // ── Q / dO loading lambdas (per m_block, staged) ──
        auto load_Q_LSE = [&] (int const m_block, int const smem_pipe_write) {
            Tensor tQsQ_cur = tQsQ(_, _, _, smem_pipe_write);
            Tensor tQgQ_cur = tQgQ(_, _, _, m_block);
            int const seqlenq_row_limit = seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}));
#if defined(USE_PPU) && USE_AIU
            if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for Q.
                // Hierarchical tiling: 8x64 -> 16x64 (vertical x2) -> 16x128 (horizontal x2)
                //                      -> blockM x128 (vertical) -> blockM x headdim (horizontal)
                // offset(m,k) = (k/2)*kHalfSize + (m/2)*kTile16x128 + (k%2)*kTile16x64 + (m%2)*kAtomSize
                static_assert(kBlockKGmem == 64, "AIU hierarchical layout assumes 8x64 atom");
                const Element* block_q_ptr = mQ.data().get()
                    + (seqlen_info.offset_q + m_block * kBlockM) * get<0>(ml.stride_Q);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumMTiles = kBlockM / kAiuTileH;
                static constexpr int kTotalTiles = kNumMTiles * kNumKTilesTotal;
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile16x64 = 2 * kAtomSize;                         // 16x64 = 1024
                static constexpr int kTile16x128 = 2 * kTile16x64;                       // 16x128 = 2048
                static constexpr int kHalfSize = kBlockM * 2 * kAiuTileW;                // blockM x 128
                static constexpr int kLog2NumKTiles = __builtin_ctz(kNumKTilesTotal);
                static constexpr int kLog2BlockKGmem = __builtin_ctz(kBlockKGmem);
                static constexpr int kLog2AtomSize = __builtin_ctz(kAtomSize);
                static constexpr int kLog2Tile16x64 = __builtin_ctz(kTile16x64);
                static constexpr int kLog2Tile16x128 = __builtin_ctz(kTile16x128);
                // NumMmaWarps is always a multiple of kNumKTilesTotal, so k is loop-invariant
                // and m increments by kMStride (always even) => m&1 is also loop-invariant.
                static constexpr int kMStride = CollectiveMainloop::NumMmaWarps >> kLog2NumKTiles;
                static_assert(kMStride % 2 == 0, "kMStride must be even for m&1 loop-invariance");
                static constexpr int kSmemMHalfDelta = (kMStride >> 1) << kLog2Tile16x128;

                auto const gmem_m_stride_Q = kAiuTileH * get<0>(ml.stride_Q);

                // Precompute loop-invariant offsets; scope k/m_init to free registers
                auto smem_loop_base = sQ_AIU_COPY.data() + smem_pipe_write * kBlockM * kHeadDim;
                int smem_m_half_offset;
                const Element* tile_gmem;
                {
                    int const k = warp_idx & (kNumKTilesTotal - 1);
                    int const m_init = warp_idx >> kLog2NumKTiles;
                    smem_loop_base = smem_loop_base + ((k >> 1) * kHalfSize + ((k & 1) << kLog2Tile16x64) + ((m_init & 1) << kLog2AtomSize));
                    smem_m_half_offset = (m_init >> 1) << kLog2Tile16x128;
                    tile_gmem = block_q_ptr
                        + (k << kLog2BlockKGmem)
                        + static_cast<int64_t>(m_init) * gmem_m_stride_Q;
                }

                // Loop body: only 1 smem add + copy + 2 constant increments
                int m_rows_cur = warp_idx >> kLog2NumKTiles;
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += CollectiveMainloop::NumMmaWarps) {
                    auto tile_smem = smem_loop_base + smem_m_half_offset;
                    // dim_h must bound the seqlen tail: AIU zero-fills rows past
                    // dim_h, keeping padded tail rows exact-0 (garbage rows would
                    // carry inf/nan into dS and poison dK/dV).
                    int const valid_rows = cute::min(kAiuTileH, cute::max(0,
                        seqlen_info.seqlen_q - (seqlen_info.offset_q + m_block * kBlockM + m_rows_cur * kAiuTileH)));
                    if (valid_rows > 0) {
                        gmem_tiled_copy_Q.desc_.dim_h = valid_rows;
                        CollectiveMainloop::Gmem_copy_struct_Q::copy(
                            cute::raw_pointer_cast(tile_smem),
                            tile_gmem,
                            gmem_tiled_copy_Q.desc_,
                            0, 0);
                    } else {
                        uint32_t* zp = reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(tile_smem));
                        #pragma unroll
                        for (int i = threadIdx.x & 31; i < int(kAtomSize * sizeof(Element) / 4); i += 32) { zp[i] = 0u; }
                    }
                    tile_gmem += static_cast<int64_t>(kMStride) * gmem_m_stride_Q;
                    smem_m_half_offset += kSmemMHalfDelta;
                    m_rows_cur += kMStride;
                }
            } else {
                // Stock fp32 path: predicate rows past seqlen_q and zero-fill
                // them in smem (cp.async zfill), mirroring the AIU dim_h tail
                // zero-fill — garbage Q/dO tail rows poison dP/dS (0*inf=NaN).
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_Q, tQgQ_cur, tQsQ_cur, t0QcQ, tQpQ, seqlenq_row_limit);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tQsQ); ++m) {
                if (EvenM || m < size<1>(tQsQ) - 1 || get<0>(tQcQ(_0{}, m, _0{})) < kBlockM) {
                    bool const predicate_m = get<0>(t0QcQ(_0{}, m, _0{})) < seqlenq_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tQsQ); ++k) {
                        cute::copy(gmem_tiled_copy_Q.with(tQpQ(k) && predicate_m), tQgQ_cur(_, m, k), tQsQ_cur(_, m, k));
                    }
                }
            }
#endif
            Tensor tLSEgLSE_cur = tLSEgLSE(_, _, m_block);
            Tensor tLSEsLSE_cur = tLSEsLSE(_, _, smem_pipe_write);
            #pragma unroll
            for (int m = 0; m < size<1>(tLSEsLSE); ++m) {
                if (get<0>(tLSEcLSE(_0{}, m)) < kBlockM) {
                    cute::copy(gmem_tiled_copy_lse, tLSEgLSE_cur(_, m), tLSEsLSE_cur(_, m));
                }
            }
        };

        auto load_dO_dPsum = [&] (int const m_block, int const smem_pipe_write) {
            Tensor tdOsdO_cur = tdOsdO(_, _, _, smem_pipe_write);
            Tensor tdOgdO_cur = tdOgdO(_, _, _, m_block);
            int const seqlenq_row_limit = seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}));
#if defined(USE_PPU) && USE_AIU
            if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for dO.
                // Hierarchical tiling: 8x64 -> 16x64 (vertical x2) -> 16x128 (horizontal x2)
                //                      -> blockM x128 (vertical) -> blockM x headdim (horizontal)
                // offset(m,k) = (k/2)*kHalfSize + (m/2)*kTile16x128 + (k%2)*kTile16x64 + (m%2)*kAtomSize
                const Element* block_do_ptr = mdO.data().get()
                    + (seqlen_info.offset_q + m_block * kBlockM) * get<0>(ml.stride_dO);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumMTiles = kBlockM / kAiuTileH;
                static constexpr int kTotalTiles = kNumMTiles * kNumKTilesTotal;
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile16x64 = 2 * kAtomSize;                         // 16x64 = 1024
                static constexpr int kTile16x128 = 2 * kTile16x64;                       // 16x128 = 2048
                static constexpr int kHalfSize = kBlockM * 2 * kAiuTileW;                // blockM x 128
                static constexpr int kLog2NumKTiles = __builtin_ctz(kNumKTilesTotal);
                static constexpr int kLog2BlockKGmem = __builtin_ctz(kBlockKGmem);
                static constexpr int kLog2AtomSize = __builtin_ctz(kAtomSize);
                static constexpr int kLog2Tile16x64 = __builtin_ctz(kTile16x64);
                static constexpr int kLog2Tile16x128 = __builtin_ctz(kTile16x128);
                // NumMmaWarps is always a multiple of kNumKTilesTotal, so k is loop-invariant
                // and m increments by kMStride (always even) => m&1 is also loop-invariant.
                static constexpr int kMStride = CollectiveMainloop::NumMmaWarps >> kLog2NumKTiles;
                static_assert(kMStride % 2 == 0, "kMStride must be even for m&1 loop-invariance");
                static constexpr int kSmemMHalfDelta = (kMStride >> 1) << kLog2Tile16x128;

                auto const gmem_m_stride_dO = kAiuTileH * get<0>(ml.stride_dO);

                // Precompute loop-invariant offsets; scope k/m_init to free registers
                auto smem_loop_base = sdO_AIU_COPY.data() + smem_pipe_write * kBlockM * kHeadDim;
                int smem_m_half_offset;
                const Element* tile_gmem;
                {
                    int const k = warp_idx & (kNumKTilesTotal - 1);
                    int const m_init = warp_idx >> kLog2NumKTiles;
                    smem_loop_base = smem_loop_base + ((k >> 1) * kHalfSize + ((k & 1) << kLog2Tile16x64) + ((m_init & 1) << kLog2AtomSize));
                    smem_m_half_offset = (m_init >> 1) << kLog2Tile16x128;
                    tile_gmem = block_do_ptr
                        + (k << kLog2BlockKGmem)
                        + static_cast<int64_t>(m_init) * gmem_m_stride_dO;
                }

                // Loop body: only 1 smem add + copy + 2 constant increments
                int m_rows_cur = warp_idx >> kLog2NumKTiles;
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += CollectiveMainloop::NumMmaWarps) {
                    auto tile_smem = smem_loop_base + smem_m_half_offset;
                    // dim_h must bound the seqlen tail (see the Q copy above).
                    int const valid_rows = cute::min(kAiuTileH, cute::max(0,
                        seqlen_info.seqlen_q - (seqlen_info.offset_q + m_block * kBlockM + m_rows_cur * kAiuTileH)));
                    if (valid_rows > 0) {
                        gmem_tiled_copy_dO.desc_.dim_h = valid_rows;
                        CollectiveMainloop::Gmem_copy_struct_Q::copy(
                            cute::raw_pointer_cast(tile_smem),
                            tile_gmem,
                            gmem_tiled_copy_dO.desc_,
                            0, 0);
                    } else {
                        uint32_t* zp = reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(tile_smem));
                        #pragma unroll
                        for (int i = threadIdx.x & 31; i < int(kAtomSize * sizeof(Element) / 4); i += 32) { zp[i] = 0u; }
                    }
                    tile_gmem += static_cast<int64_t>(kMStride) * gmem_m_stride_dO;
                    smem_m_half_offset += kSmemMHalfDelta;
                    m_rows_cur += kMStride;
                }
            } else {
                // Stock fp32 path: predicate rows past seqlen_q and zero-fill
                // them in smem (see the Q copy above).
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_dO, tdOgdO_cur, tdOsdO_cur, t0QcQ, tQpQ, seqlenq_row_limit);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tdOsdO); ++m) {
                if (EvenM || m < size<1>(tdOsdO) - 1 || get<0>(tQcQ(_0{}, m, _0{})) < kBlockM) {
                    bool const predicate_m = get<0>(t0QcQ(_0{}, m, _0{})) < seqlenq_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tdOsdO); ++k) {
                        cute::copy(gmem_tiled_copy_dO.with(tdOpdO(k) && predicate_m), tdOgdO_cur(_, m, k), tdOsdO_cur(_, m, k));
                    }
                }
            }
#endif
            Tensor tLSEgdPsum_cur = tLSEgdPsum(_, _, m_block);
            Tensor tLSEsdPsum_cur = tLSEsdPsum(_, _, smem_pipe_write);
            #pragma unroll
            for (int m = 0; m < size<1>(tLSEsdPsum); ++m) {
                if (get<0>(tLSEcLSE(_0{}, m)) < kBlockM) {
                    cute::copy(gmem_tiled_copy_lse, tLSEgdPsum_cur(_, m), tLSEsdPsum_cur(_, m));
                }
            }
        };

        if constexpr (!Deterministic) {
            clear(tdKrdK);
            clear(tdVrdV);
        }

        bool any_work = false;

        // Mask for the S tile (SdP_swapAB ⇒ transposed fragment).
        MaskFlexFlash<kBlockM, kBlockN, TiledMmaSdP, /*Transposed=*/SdP_swapAB> mask;

        // ── Deterministic dK/dV turnstile state (LoopQ) ──
        // Contributors to dKaccum[n_block, bidh_kv, bidb]: every CTA
        // (n_block, bidh, bidb) flushes once per INTERSECTING slice.  Fixed
        // order: (slice_idx ascending, q-head ascending).  Gap-free turns:
        // prior_intersect * qhead_per_khead + (bidh % qhead_per_khead).
        using Barrier = ArbDetBarrier;
        int const num_batch = ml.num_batch;
        // shape_K = (seqlen_k, d, h_k, b) — h_k is get<2>, NOT get<1> (that's d).
        int const num_head_kv = get<2>(ml.shape_K);
        int const qhead_per_khead = cute::ceil_div(get<2>(ml.shape_Q), num_head_kv);
        int const head_idx_in_group = bidh - bidh_kv * qhead_per_khead;
        int* const dk_lock = !Deterministic ? nullptr : params.epilogue.dk_semaphore + (bidb * num_head_kv + bidh_kv);
        int* const dv_lock = !Deterministic ? nullptr : params.epilogue.dv_semaphore + (bidb * num_head_kv + bidh_kv);
        // Per-slice dK/dV staging accumulators (Deterministic flushes per slice).
        // In the default path these are zero-sized placeholders (unused).
        Tensor tdKrdK_slice = [&]() -> auto {
            if constexpr (Deterministic) {
                return partition_fragment_C(tiled_mma_dKV,
                    select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
            } else {
                return make_tensor<float>(Shape<_0>{});
            }
        }();
        Tensor tdVrdV_slice = [&]() -> auto {
            if constexpr (Deterministic) {
                return partition_fragment_C(tiled_mma_dKV,
                    select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
            } else {
                return make_tensor<float>(Shape<_0>{});
            }
        }();
        // bwd_step receives its dK/dV target accumulators by reference: the
        // per-slice staging in Deterministic mode, the CTA-wide accumulators
        // otherwise (selected at each call site with if constexpr).
        int prior_intersect = 0;  // intersecting slices flushed so far

        // ── Outer slice loop: every slice whose K range intersects this
        //    n_block contributes to dK/dV (register accum) and dQ (atomic).
        int const n_lo = n_block * kBlockN;
        int const n_hi = n_lo + kBlockN;
        for (int slice_idx = 0; slice_idx < sl.num_slices; ++slice_idx) {
            int const ks = sl.k_starts[slice_idx];
            int const ke = sl.k_ends[slice_idx];
            if (ke <= n_lo || ks >= n_hi) { continue; }  // K range misses this n_block

            int const q_start = sl.q_starts[slice_idx];
            int const q_end   = min(sl.q_ends[slice_idx], seqlen_q);
            if (q_end <= q_start) { continue; }

            int const mask_type       = sl.mask_types[slice_idx];
            int const diagonal_offset = sl.diagonal_offsets[slice_idx];
            int const band_width      = sl.band_widths[slice_idx];

            // Tile-level diagonal pruning: the m_blocks that may contribute
            // form a contiguous interval (Q range ∩ K range ∩ mask region ∩
            // seqlen).  The K-range clamp inside bwd_m_cover_interval keeps
            // this pruning predicate identical to the one used by the
            // deterministic turn accounting (gap-free turns).
            int m_block_min, m_block_max;
            bwd_m_cover_interval(mask_type, diagonal_offset, band_width,
                                 q_start, q_end, ks, ke, n_lo, n_hi,
                                 kBlockM, seqlen_q, seqlen_k,
                                 m_block_min, m_block_max);
            if (m_block_max <= m_block_min) { continue; }  // fully-masked (slice, n_block)

            // Direction-aware iteration helpers.  Deterministic LoopQ is
            // forced ascending (kDirEff) so the per-CTA dQ turn sequence is
            // monotone (deadlock-free semaphore hand-off).
            auto blk_item = [&] (int k, int lo, int hi) {
                return (kDirEff == DispatchDirection::MaxToMin) ? hi - 1 - k : lo + k;
            };
            auto blk_finish = [&] (int k, int lo, int hi) {
                return (kDirEff == DispatchDirection::MaxToMin) ? k >= hi - lo : lo + k >= hi;
            };
            int const m_cnt = m_block_max - m_block_min;

            // Cross-slice smem hazard guard: a processed slice ends with the dK
            // GEMM (reads sQ/sdSt) and, for kStages > 1, has no trailing
            // __syncthreads() (the "make sure sdS is written" barrier precedes
            // dQ/dK).  The staging below re-issues cp.async into the Q/dO smem,
            // so a thread racing into the next intersecting slice could overwrite
            // Q while a lagging thread still reads it in the previous slice's dK
            // GEMM -- corrupting dK alone (dV reads dO before that barrier; dQ
            // reads K/dS, which staging never rewrites).  Only n_blocks covered
            // by >=2 slices (non-tile-aligned masks) hit this.  slice_idx 0 is
            // already ordered by the K/V load barrier above.
            if (slice_idx > 0) { __syncthreads(); }

            // ── Stage the first m_blocks for this slice (load order ==
            //    consumption order, direction-aware) ──
            for_each(make_int_sequence<kStages>{}, [&] (auto stage) {
                static constexpr bool Is_first_stage = CUTE_STATIC_V(stage) == 0;
                static constexpr bool Is_last_stage  = CUTE_STATIC_V(stage) == kStages - 1;
                if constexpr (!Is_last_stage || kStages == 1) {
                    if (Is_first_stage || int(stage) < m_cnt) {
                        load_Q_LSE(blk_item(stage, m_block_min, m_block_max), stage);
                    }
                }
                cute::cp_async_fence();
                if constexpr (stage < kStages_dO) {
                    if (Is_first_stage || int(stage) < m_cnt) {
                        load_dO_dPsum(blk_item(stage, m_block_min, m_block_max), stage);
                    }
                    cute::cp_async_fence();
                }
            });

            int smem_pipe_read = 0, smem_pipe_read_do = 0, smem_pipe_write = kStages - 1, smem_pipe_write_do = 0;
            int step_cnt = 0;  // steps executed so far in this slice

            auto load_Q_next = [&] {
                int const nxt = step_cnt + (kStages > 1 ? kStages - 1 : 1);
                if (nxt < m_cnt) {
                    load_Q_LSE(blk_item(nxt, m_block_min, m_block_max), kStages > 1 ? smem_pipe_write : 0);
                }
                cute::cp_async_fence();
            };

            auto load_dO_next = [&] {
                int const nxt = step_cnt + kStages_dO;
                if (nxt < m_cnt) {
                    load_dO_dPsum(blk_item(nxt, m_block_min, m_block_max), kStages_dO > 1 ? smem_pipe_write_do : 0);
                }
                cute::cp_async_fence();
            };

            // ── One bwd step for a single (slice, m_block) tile ──
            auto bwd_step = [&](int m_block, auto& dK_acc, auto& dV_acc, auto mask_fn) {
                Tensor tSrS = partition_fragment_C(tiled_mma_SdP,
                    select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
                clear(tSrS);
                flash::cp_async_wait<(kStages > 1) ? 1 : 0>();
                __syncthreads();
                Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(thr_mma_SdP, sQ(_, _, _0{}));
                Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(thr_mma_SdP, sK);
                /** recompute s=Q*K^T */
                flash::gemm_sm80<false /*A_in_regs*/, false /*B_in_regs*/, SdP_swapAB>(
                    tSrS, tSrQ, tSrK, tSsQ(_, _, _, kStages > 1 ? smem_pipe_read : 0), tSsK,
                    tiled_mma_SdP, smem_tiled_copy_QdO, smem_tiled_copy_KV,
                    smem_thr_copy_QdO, smem_thr_copy_KV, nullptr /*hook*/);
                Tensor tLSErLSE = cute::conditional_return<!ShuffleLSE>(
                    make_fragment_like(tSsLSE(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
                if constexpr (!ShuffleLSE) {
                    cute::copy(tSsLSE(_, kStages > 1 ? smem_pipe_read : 0), tLSErLSE);
                } else {
                    #pragma unroll
                    for (int i = 0; i < kStatsPerThread; ++i) {
                        tLSErLSE(i) = tSsLSE((thread_idx % 32) / 4 + i * 8, kStages > 1 ? smem_pipe_read : 0);
                    }
                }

                // ── Softcap: S = tanh(S * scale/cap), then dtanh BEFORE mask
                //    (mask fills entries with -inf; 1-(-inf)^2 = NaN). ──
                if constexpr (Has_softcap) { flash::apply_softcap(tSrS, ml.softcap_val); }
                Tensor scores = make_tensor(tSrS.data(),
                    flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
                auto dtanh = [&] { if constexpr (Has_softcap) return flash::calculate_dtanh(scores); else return nullptr; }();

                // Mask BEFORE P = exp2(S - LSE): applies algebraic diagonal
                // constraint + per-slice K/Q range clipping + seqlenk tail.
                mask_fn(tSrS, m_block);

                /** recompute P = softmax(S) using fwd LSE */
                #pragma unroll
                for (int mi = 0; mi < size<0>(scores); ++mi) {
                    float const lse_scaled = [&] {
                        if constexpr (!ShuffleLSE) return tLSErLSE(mi);
                        else return __shfl_sync(0xffffffff, tLSErLSE(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                    }();
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ++ni) {
                        scores(mi, ni) = exp2f(scores(mi, ni) * ml.softmax_scale_log2 - lse_scaled);
                    }
                }

                // ── Dropout replay (LoopM): zero the recomputed P with the
                // SAME coordinate-keyed Philox stream as fwd — one pass is
                // enough: dV uses the zeroed P, and dS = Pdrop ∘ (rp·dP − δ)
                // vanishes on dropped entries through Pdrop = 0. ──
                if constexpr (Is_dropout) {
                    DropoutFlexFlash<kBlockM, kBlockN, TiledMmaSdP, SdP_swapAB> dropout(
                        rng_seed_eff,
                        rng_offset_base_eff
                            + static_cast<unsigned long long>(bidb) * params.dropout.num_heads + bidh,
                        params.dropout.p_keep_in_uint8_t);
                    dropout.apply(tSrS, m_block, n_block, thread_idx);
                }

                /** dP = dO*V^T */
                Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP,
                    select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
                clear(tdPrdP);
                int smem_pipe_read_do_cur = Q_dO_same_stages ? smem_pipe_read : smem_pipe_read_do;
                flash::cp_async_wait<(kStages_dO > 1) ? 1 : 0>();
                __syncthreads();
                auto hook = cute::conditional_return<(kStages > 1)>(load_Q_next, nullptr);
                Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(thr_mma_SdP, sdO(_, _, _0{}));
                Tensor tdPrV_cur = cute::conditional_return<V_in_regs>(tdPrV,
                    mma_partition_fragment_AB</*A=*/SdP_swapAB>(thr_mma_SdP, sV));
                flash::gemm_sm80<false /*A_in_regs*/, V_in_regs, SdP_swapAB>(
                    tdPrdP, tdPrdO, tdPrV_cur, tdPsdO(_, _, _, kStages_dO > 1 ? smem_pipe_read_do_cur : 0), tdPsV,
                    tiled_mma_SdP, smem_tiled_copy_QdO, smem_tiled_copy_KV,
                    smem_thr_copy_QdO, smem_thr_copy_KV, hook);
                Tensor tLSErdPsum = cute::conditional_return<!ShuffledPsum>(
                    make_fragment_like(tSsdPsum(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
                if constexpr (!ShuffledPsum) {
                    cute::copy(tSsdPsum(_, kStages_dO > 1 ? smem_pipe_read_do_cur : 0), tLSErdPsum);
                } else {
                    #pragma unroll
                    for (int i = 0; i < kStatsPerThread; ++i) {
                        tLSErdPsum(i) = tSsdPsum((thread_idx % 32) / 4 + i * 8, kStages_dO > 1 ? smem_pipe_read_do_cur : 0);
                    }
                }

                /** dS = P * (dP - dPsum) [* dtanh with softcap]
                    Dropout: dP enters as rp·dP so that with the preprocess
                    delta = rowsum(dO·O) (which carries rp through O):
                    dS = Pdrop ∘ (rp·dP_raw − delta) — the exact chain rule
                    of O = rp·(Pdrop @ V).  dQ/dK then need no extra rp. */
                Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
                #pragma unroll
                for (int mi = 0; mi < size<0>(dS); ++mi) {
                    float const dP_sum_cur = [&] {
                        if constexpr (!ShuffledPsum) return tLSErdPsum(mi);
                        else return __shfl_sync(0xffffffff, tLSErdPsum(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                    }();
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(dS); ++ni) {
                        float dp = dS(mi, ni);
                        if constexpr (Is_dropout) { dp *= params.dropout.rp_dropout; }
                        dS(mi, ni) = scores(mi, ni) * (dp - dP_sum_cur);
                        if constexpr (Has_softcap) { dS(mi, ni) *= dtanh(mi, ni); }
                    }
                }

                // Convert scores from fp32 to fp16/bf16
                Tensor rP = make_tensor_like<Element>(tSrS);
                flash::convert_type_out(tSrS, rP);
                if constexpr (!Mma_dKV_is_RS) {
                    Tensor tPaP = r2s_thr_copy_PdS.retile_S(rP);
                    cute::copy(r2s_tiled_copy_PdS, tPaP, tPsP);
                }
                Tensor rdS = make_tensor_like<Element>(tdPrdP);
                flash::convert_type_out(tdPrdP, rdS);
                if constexpr (!Mma_dKV_is_RS) { __syncthreads(); }
                Tensor tdSadS = r2s_thr_copy_PdS.retile_S(rdS);
                cute::copy(r2s_tiled_copy_PdS, tdSadS, tdSsdS);

                /** dV = P^T*dO */
                Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(thr_mma_dKV, sdOt(_, _, _0{}));
                Tensor tdVsdO_cur = tdVsdOt(_, _, _, kStages_dO > 1 ? smem_pipe_read_do_cur : 0);
                if constexpr (Mma_dKV_is_RS) {
#ifdef USE_PPU
                    Tensor tdVrP_acc = [&]() -> auto {
                        if constexpr (ArchTag::kMinComputeCapability >= 89) {
                            return make_tensor_like<Element>(tSrS);
                        } else {
                            return flash::convert_acc<Element>(tSrS);
                        }
                    }();
                    Tensor tdVrP = [&]() -> auto {
                        if constexpr (ArchTag::kMinComputeCapability >= 89) {
                            return make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
                        } else {
                            return make_tensor(tdVrP_acc.data(), make_layout(get<0>(tSrQ.layout()), get<1>(tSrS.layout()), get<2>(tSrS.layout())));
                        }
                    }();
#else
                    Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
#endif
                    flash::gemm_rs_sm80(dV_acc, tdVrP, tdVrdO, tdVsdO_cur,
                                        tiled_mma_dKV, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
                } else {
                    Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(thr_mma_dKV, sPt);
                    flash::gemm_sm80<false, false, /*SwapAB=*/dKV_swapAB>(
                        dV_acc, tdVrP, tdVrdO, tdVsPt, tdVsdO_cur,
                        tiled_mma_dKV, smem_tiled_copy_PdSt, smem_tiled_copy_QdOt,
                        smem_thr_copy_PdSt, smem_thr_copy_QdOt, nullptr);
                }
                __syncthreads();  // make sure sdS is written
                /** dQ = dS*K (atomic into fp32 dQaccum) */
                auto do_mma_dQ = [&] (auto hook_dQ) {
                    Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ,
                        select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                    clear(tdQrdQ);
                    Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(thr_mma_dQ, sdS);
                    Tensor tdQrK  = mma_partition_fragment_AB</*A=*/dQ_swapAB>(thr_mma_dQ, sKt);
                    flash::gemm_sm80<false /*A_in_regs*/, false /*B_in_regs*/, /*SwapAB=*/dQ_swapAB>(
                        tdQrdQ, tdQrdS, tdQrK, tdQsdS, tdQsKt, tiled_mma_dQ,
                        smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt, hook_dQ);
                    if constexpr (Deterministic) {
                        // Turn = gap-free rank of this (slice, n_block) among
                        // all contributors of dQaccum[m_block], ordered
                        // (slice ascending, n_block ascending).  Prior slices:
                        // their covering n_block counts for this m_block.
                        int turn_q = 0;
                        for (int s = 0; s < slice_idx; ++s) {
                            turn_q += fwd_n_cover_count(
                                sl.mask_types[s], sl.diagonal_offsets[s],
                                sl.band_widths[s],
                                sl.q_starts[s], sl.q_ends[s],
                                sl.k_starts[s], sl.k_ends[s],
                                m_block, kBlockM, kBlockN, seqlen_q, seqlen_k);
                        }
                        // Within-slice rank: covering n_blocks below n_block.
                        int cn_lo = max(ks, 0) / kBlockN;
                        int cn_hi = cute::ceil_div(min(ke, seqlen_k), kBlockN);
                        fwd_valid_n_range(mask_type, diagonal_offset, band_width,
                                          m_block * kBlockM, min(m_block * kBlockM + kBlockM, seqlen_q),
                                          kBlockN, cn_lo, cn_hi);
                        turn_q += max(0, min(n_block, cn_hi) - cn_lo);
                        int* const dq_lock = ml.dq_semaphore + bidb * get<2>(ml.shape_Q) + bidh;
                        Barrier::wait_eq(dq_lock, thread_idx, m_block * num_batch * get<2>(ml.shape_Q), turn_q);
                        Tensor tdQrdQ_atomic = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
                        static_assert(CUTE_STATIC_V(size(tdQrdQ_atomic)) == CUTE_STATIC_V(size(tdQgdQaccum(_, _, 0))));
                        if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                            // CVT SHUFFLING_GAIT causes bit3<->bit6 headdim swap: each acc
                            // half belongs to the own or partner warp's 48x8 tile, so write
                            // through the matching thread_slice partition (partner = warp^4).
                            // Single-loop form: the side select (thread_idx bit7/bit8) is
                            // warp-uniform and the (i%8)<4 pattern is a compile-time
                            // constant per unrolled element, so the duplicated side
                            // branches collapse into one predicated pointer select.
                            constexpr int partner_warp_mask = kHeadDim == 256 ? 256 : 128;
                            int partner_tid = thread_idx ^ partner_warp_mask;
                            auto r2s_thr_copy_partner = r2s_tiled_copy_dQaccum.get_thread_slice(partner_tid);
                            Tensor tdQgdQaccum_partner = r2s_thr_copy_partner.partition_D(gdQaccum);
                            Tensor tdQgdQaccum_atomic_own = tdQgdQaccum(_, _, m_block);
                            Tensor tdQgdQaccum_atomic_partner = tdQgdQaccum_partner(_, _, m_block);
                            constexpr int kTotal = CUTE_STATIC_V(size(decltype(tdQrdQ_atomic){}));
                            // low side (bit clear): i%8<4 → own(i),       else → partner(i-4)
                            // high side (bit set):  i%8<4 → partner(i+4), else → own(i)
                            bool const low_side = (thread_idx & partner_warp_mask) == 0;
                            int const partner_adj = low_side ? -4 : 4;
                            #pragma unroll
                            for (int i = 0; i < kTotal; ++i) {
                                bool const to_own = (((i % 8) < 4) == low_side);
                                auto* dst = to_own ? &tdQgdQaccum_atomic_own(i)
                                                   : &tdQgdQaccum_atomic_partner(i + partner_adj);
                                atomicAdd(dst, tdQrdQ_atomic(i));
                            }
                        } else {
                            Tensor tdQgdQaccum_atomic = tdQgdQaccum(_, _, m_block);
                            #pragma unroll
                            for (int i = 0; i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
                        }
                        Barrier::arrive_inc(dq_lock, thread_idx, m_block * num_batch * get<2>(ml.shape_Q));
                    } else {
                        Tensor tdQrdQ_atomic = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
                        static_assert(CUTE_STATIC_V(size(tdQrdQ_atomic)) == CUTE_STATIC_V(size(tdQgdQaccum(_, _, 0))));
                        if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                            // CVT SHUFFLING_GAIT causes bit3<->bit6 headdim swap: each acc
                            // half belongs to the own or partner warp's 48x8 tile, so write
                            // through the matching thread_slice partition (partner = warp^4).
                            // Single-loop form (see the seqlen-boundary twin above).
                            constexpr int partner_warp_mask = kHeadDim == 256 ? 256 : 128;
                            int partner_tid = thread_idx ^ partner_warp_mask;
                            auto r2s_thr_copy_partner = r2s_tiled_copy_dQaccum.get_thread_slice(partner_tid);
                            Tensor tdQgdQaccum_partner = r2s_thr_copy_partner.partition_D(gdQaccum);
                            Tensor tdQgdQaccum_atomic_own = tdQgdQaccum(_, _, m_block);
                            Tensor tdQgdQaccum_atomic_partner = tdQgdQaccum_partner(_, _, m_block);
                            constexpr int kTotal = CUTE_STATIC_V(size(decltype(tdQrdQ_atomic){}));
                            // low side (bit clear): i%8<4 → own(i),       else → partner(i-4)
                            // high side (bit set):  i%8<4 → partner(i+4), else → own(i)
                            bool const low_side = (thread_idx & partner_warp_mask) == 0;
                            int const partner_adj = low_side ? -4 : 4;
                            #pragma unroll
                            for (int i = 0; i < kTotal; ++i) {
                                bool const to_own = (((i % 8) < 4) == low_side);
                                auto* dst = to_own ? &tdQgdQaccum_atomic_own(i)
                                                   : &tdQgdQaccum_atomic_partner(i + partner_adj);
                                atomicAdd(dst, tdQrdQ_atomic(i));
                            }
                        } else {
                            Tensor tdQgdQaccum_atomic = tdQgdQaccum(_, _, m_block);
                            #pragma unroll
                            for (int i = 0; i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
                        }
                    }
                };
                if constexpr (kStages > 1) { do_mma_dQ(load_dO_next); }
                /** dK = dS^T * Q */
                Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(thr_mma_dKV, sQt(_, _, _0{}));
                Tensor tdKsQ_cur = tdKsQt(_, _, _, kStages > 1 ? smem_pipe_read : 0);
                if constexpr (Mma_dKV_is_RS) {
#ifdef USE_PPU
                    Tensor tdKrdS_acc = [&]() -> auto {
                        if constexpr (ArchTag::kMinComputeCapability >= 89) {
                            return make_tensor_like<Element>(tdPrdP);
                        } else {
                            return flash::convert_acc<Element>(tdPrdP);
                        }
                    }();
                    Tensor tdKrdS = [&]() -> auto {
                        if constexpr (ArchTag::kMinComputeCapability >= 89) {
                            return make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
                        } else {
                            return make_tensor(tdKrdS_acc.data(), make_layout(get<0>(tdPrdO.layout()), get<1>(tdPrdP.layout()), get<2>(tdPrdP.layout())));
                        }
                    }();
#else
                    Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
#endif
                    flash::gemm_rs_sm80(dK_acc, tdKrdS, tdKrQ, tdKsQ_cur,
                                        tiled_mma_dKV, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
                } else {
                    Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(thr_mma_dKV, sdSt);
                    flash::gemm_sm80<false, false, /*SwapAB=*/dKV_swapAB>(
                        dK_acc, tdKrdS, tdKrQ, tdKsdSt, tdKsQ_cur,
                        tiled_mma_dKV, smem_tiled_copy_PdSt, smem_tiled_copy_QdOt,
                        smem_thr_copy_PdSt, smem_thr_copy_QdOt,
                        cute::conditional_return<(kStages > 1)>(nullptr, load_dO_next));
                }
                if constexpr (kStages == 1) {
                    __syncthreads();
                    do_mma_dQ(load_Q_next);
                }

                smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
                smem_pipe_read_do = smem_pipe_read_do < kStages_dO - 1 ? smem_pipe_read_do + 1 : 0;
                smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
                smem_pipe_write_do = smem_pipe_write_do < kStages_dO - 1 ? smem_pipe_write_do + 1 : 0;
            };

            // Flex flash attention: MaskFlexFlash handles everything (diagonal +
            // K/Q range clipping + seqlenk tail), so there is no separate
            // no-mask iteration; mask.apply() runs on every computed block
            // (tile pruning above only tightened the loop bounds).
            auto mask_fn = [&](auto& tSrS, int m_block) {
                mask.template apply</*Seqlenk_mask=*/true>(
                    tSrS, m_block, n_block,
                    mask_type, ks, ke,
                    diagonal_offset, band_width,
                    q_start, q_end,
                    thread_idx, seqlen_q, seqlen_k,
                    sl.mask_bits, sl.mask_row_stride);
            };

            if constexpr (Deterministic) { clear(tdKrdK_slice); clear(tdVrdV_slice); }

            CUTLASS_PRAGMA_NO_UNROLL
            for (int step = 0; step < m_cnt; ++step) {
                int const m_block_cur = blk_item(step, m_block_min, m_block_max);
                if constexpr (Deterministic) {
                    bwd_step(m_block_cur, tdKrdK_slice, tdVrdV_slice, mask_fn);
                } else {
                    bwd_step(m_block_cur, tdKrdK, tdVrdV, mask_fn);
                }
                ++step_cnt;
            }
            any_work = true;

            if constexpr (Deterministic) {
                // ── Per-slice dK/dV flush through the turnstile ──
                // Order: (slice ascending, q-head ascending); turns are
                // gap-free because prior_intersect counts exactly the slices
                // that actually flushed for this n_block.
                #pragma unroll
                for (int i = 0; i < size(tdKrdK_slice); ++i) { tdKrdK_slice(i) *= ml.softmax_scale; }
                if constexpr (Is_dropout) {
                    // dV = rp · Pdrop^T dO — the rp factor lands here (dK
                    // already carries it through the rp-scaled dS).
                    #pragma unroll
                    for (int i = 0; i < size(tdVrdV_slice); ++i) { tdVrdV_slice(i) *= params.dropout.rp_dropout; }
                }
                Tensor mdKaccum = make_tensor(make_gmem_ptr(params.epilogue.ptr_dKaccum),
                    params.epilogue.shape_dKaccum, params.epilogue.stride_dKaccum)(_, bidh_kv, bidb);
                Tensor mdVaccum = make_tensor(make_gmem_ptr(params.epilogue.ptr_dVaccum),
                    params.epilogue.shape_dVaccum, params.epilogue.stride_dVaccum)(_, bidh_kv, bidb);
                Tensor gdKaccum = local_tile(mdKaccum, Shape<Int<kBlockN * kHeadDim>>{}, make_coord(n_block));
                Tensor gdVaccum = local_tile(mdVaccum, Shape<Int<kBlockN * kHeadDim>>{}, make_coord(n_block));
                typename CollectiveEpilogue::R2GTiledCopydKVaccum r2g_tiled_copy_dKVaccum;
                auto r2g_thr_copy_dKVaccum = r2g_tiled_copy_dKVaccum.get_thread_slice(thread_idx);
                int const turn_kv = prior_intersect * qhead_per_khead + head_idx_in_group;
                Tensor tdKrdK_atomic = r2g_thr_copy_dKVaccum.retile_S(tdKrdK_slice);
                Tensor tdKgdK_atomic = r2g_thr_copy_dKVaccum.partition_D(gdKaccum);
                Tensor tdVrdV_atomic = r2g_thr_copy_dKVaccum.retile_S(tdVrdV_slice);
                Tensor tdVgdV_atomic = r2g_thr_copy_dKVaccum.partition_D(gdVaccum);
                Barrier::wait_eq(dk_lock, thread_idx, n_block * num_batch * num_head_kv, turn_kv);
                #pragma unroll
                for (int i = 0; i < size(tdKrdK_atomic); ++i) { atomicAdd(&tdKgdK_atomic(i), tdKrdK_atomic(i)); }
                Barrier::arrive_inc(dk_lock, thread_idx, n_block * num_batch * num_head_kv);
                Barrier::wait_eq(dv_lock, thread_idx, n_block * num_batch * num_head_kv, turn_kv);
                #pragma unroll
                for (int i = 0; i < size(tdVrdV_atomic); ++i) { atomicAdd(&tdVgdV_atomic(i), tdVrdV_atomic(i)); }
                Barrier::arrive_inc(dv_lock, thread_idx, n_block * num_batch * num_head_kv);
                ++prior_intersect;
            }
        }  // end slice loop

        if constexpr (!Deterministic) {
            // dK = softmax_scale * dS^T Q  (dV has no scale)
            #pragma unroll
            for (int i = 0; i < size(tdKrdK); ++i) { tdKrdK(i) *= ml.softmax_scale; }
            if constexpr (Is_dropout) {
                #pragma unroll
                for (int i = 0; i < size(tdVrdV); ++i) { tdVrdV(i) *= params.dropout.rp_dropout; }
            }
        }

        return any_work;
    }

    // ─── SLICE-AWARE BWD MAINLOOP (LoopK) ──────────────────────────────────
    //
    // CTA = (m_block, h, b).  Q/dO/LSE/dPsum are loaded ONCE (fixed for this
    // m_block); the outer loop walks every slice whose Q range intersects
    // this m_block (ascending slice_idx), the inner loop walks the slice's
    // covering n_blocks with per-step K/V loading (smem_k/smem_v are
    // single-stage in the CollectiveMainloop storage, so K/V are loaded
    // synchronously per step — correctness-first pipelining).
    //
    //   dQ: accumulated in REGISTERS across all slices and atomicAdded into
    //       dQaccum exactly once at the end → bitwise deterministic by
    //       construction (no dQ semaphore needed).
    //   dK/dV: flushed per (slice, n_block) step.  Three routing modes:
    //       default      → atomicAdd into dK/dV accum (dK pre-scaled here,
    //                      since there is no single end-of-tile accumulator);
    //       Deterministic→ same target but through the dk/dv semaphore
    //                      turnstile (turn = gap-free rank over the exact
    //                      contributor set, ordered (slice, m_block, head));
    //       ReduceKV     → accum pointers carry the fp32 workspaces and the
    //                      softmax_scale multiply moves to the reduce kernel
    //                      (workspaces are zero-initialized host-side; a
    //                      first-write flag trick would race a plain store
    //                      against concurrent atomicAdds, so we deliberately
    //                      do NOT use it).
    template <typename FrgTensordQ>
    CUTLASS_DEVICE void run_mainloop_k(
        Params const& params,
        SharedStorage& shared_storage,
        FrgTensordQ& tdQrdQ,
        int thread_idx,
        int m_block, int bidh, int bidb)
    {
        static_assert(is_rmem<FrgTensordQ>::value, "dQ tensor must be rmem resident.");

        using MainloopParams = typename CollectiveMainloop::Params;
        MainloopParams const& ml = params.mainloop;

        SeqlenInfo_t seqlen_info{
            bidb, get<0>(ml.shape_Q), size<0>(ml.shape_K),
            ml.cu_seqlens_q, ml.cu_seqlens_k, ml.seqused_q, ml.seqused_k
        };
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = size<0>(ml.shape_K);

        // Dropout stream resolve (CUDA-graph-safe; see dropout_flex_flash.h):
        // graph mode reads the (seed, offset) pair published by the captured
        // fwd graph from device memory, replaying the identical Philox
        // stream; eager keeps the by-value scalars baked at launch.
        unsigned long long rng_seed_eff = 0, rng_offset_base_eff = 0;
        if constexpr (Is_dropout) {
            arb_resolve_rng_stream(params.dropout, rng_seed_eff,
                                   rng_offset_base_eff);
        }

        // P3 per-(b,h) heterogeneous masks: rebase the slice layout onto
        // this (batch, head) pair's layout group.  Shared layout
        // (bh_to_group == nullptr) → zero-cost identity copy.
        // shape_Q = (seqlen_q, d, h, b): num_heads from get<2> (always
        // populated, unlike dropout.num_heads which is Is_dropout-only).
        AttnSliceParams const sl = rebase_slices(
            params.slices,
            bidb * cute::get<2>(ml.shape_Q) + bidh, seqlen_q);

        // ── Shared memory tensors (same members as LoopQ; K/V single-stage) ──
        Tensor sQ   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
                                  typename CollectiveMainloop::SmemLayoutQ{});
        Tensor sdO  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
                                  typename CollectiveMainloop::SmemLayoutdO{});
        Tensor sK   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
                                  typename CollectiveMainloop::SmemLayoutK{});
        Tensor sV   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
                                  typename CollectiveMainloop::SmemLayoutV{});
        Tensor sQt  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
                                  typename CollectiveMainloop::SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
                                  typename CollectiveMainloop::SmemLayoutdOt{});
        Tensor sKt  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
                                  typename CollectiveMainloop::SmemLayoutKt{});
        Tensor sP   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
                                  typename CollectiveMainloop::SmemLayoutPdS{});
        Tensor sPt  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
                                  typename CollectiveMainloop::SmemLayoutPdSt{});
        Tensor sdS  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
                                  typename CollectiveMainloop::SmemLayoutPdS{});
        Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
                                  typename CollectiveMainloop::SmemLayoutPdSt{});
        // R2S copy view: AIU_LOAD writes 8x64 tiles in row-major (LayoutRight) order.
        // The compute path still uses sQ/sdO/sK/sV for partition_S; the PPU copy
        // atoms compute smem addresses internally so both views share the same
        // physical buffer. Only referenced on the Use_CVT_SWZL_LD path.
        // ISOLATED fp32 kernel set: the LayoutRight views exist for every
        // element type, so they are declared unconditionally (float never
        // references them since Use_CVT_SWZL_LD is false there).
        Tensor sQ_AIU_COPY  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
                                          typename CollectiveMainloop::SmemLayoutQ_AIU_COPY{});
        Tensor sdO_AIU_COPY = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
                                          typename CollectiveMainloop::SmemLayoutdO_AIU_COPY{});
        Tensor sK_AIU_COPY  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
                                          typename CollectiveMainloop::SmemLayoutK_AIU_COPY{});
        Tensor sV_AIU_COPY  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
                                          typename CollectiveMainloop::SmemLayoutV_AIU_COPY{});
#if PPU1v0_R2S_SLICE_LAYOUT || PPU1v5_R2S_SLICE_LAYOUT
        // Slice-format views of the P/dS smem buffers (written via r2s with the
        // slice layout, read back via the AIU TSM load atoms).
        Tensor sdS_slice  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
                                        typename CollectiveMainloop::SmemLayoutPdS_R2Slice{});
        Tensor sdSt_slice = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
                                        typename CollectiveMainloop::SmemLayoutPdSt_R2Slice{});
        Tensor sP_slice   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
                                        typename CollectiveMainloop::SmemLayoutPdS_R2Slice{});
        Tensor sPt_slice  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
                                        typename CollectiveMainloop::SmemLayoutPdSt_R2Slice{});
#endif
        Tensor sLSE    = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()),
                                     typename CollectiveMainloop::SmemLayoutLSE{});
        Tensor sdPsum  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()),
                                     typename CollectiveMainloop::SmemLayoutLSE{});
        Tensor sLSEMma   = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()),
                                       typename CollectiveMainloop::SmemLayoutLSEMma{});
        Tensor sdPsumMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()),
                                       typename CollectiveMainloop::SmemLayoutLSEMma{});

        // ── Gmem tensors ──
        int const bidh_kv = ml.qhead_per_khead_divmod.divide(bidh);
        Tensor mQ   = make_tensor(make_gmem_ptr(ml.ptr_Q),   ml.shape_Q,   ml.stride_Q)(_, _, bidh, bidb);
        Tensor mdO  = make_tensor(make_gmem_ptr(ml.ptr_dO),  ml.shape_dO,  ml.stride_dO)(_, _, bidh, bidb);
        Tensor mK   = make_tensor(make_gmem_ptr(ml.ptr_K),   ml.shape_K,   ml.stride_K)(_, _, bidh_kv, bidb);
        Tensor mV   = make_tensor(make_gmem_ptr(ml.ptr_V),   ml.shape_V,   ml.stride_V)(_, _, bidh_kv, bidb);
        Tensor mLSE   = make_tensor(make_gmem_ptr(ml.ptr_LSE_log2), ml.shape_LSE, ml.stride_LSE_log2)(_, bidh, bidb);
        Tensor mdPsum = make_tensor(make_gmem_ptr(ml.ptr_dPsum),    ml.shape_LSE, ml.stride_dPsum)(_, bidh, bidb);
        Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(ml.ptr_dQaccum)),
                                      ml.shape_dQaccum, ml.stride_dQaccum)(_, bidh, bidb);

        // LoopK keeps the n_block mode of K/V (it varies per inner step).
        // ISOLATED fp32 kernel set: mix gmem tiles are AIU-only; float rides
        // the stock local_tile path (mirrors LoopQ / fwd kernel).
        Tensor gQ = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return local_tile(make_mix_tensor_like(mQ),  select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
            } else {
                return local_tile(mQ,  select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
            }
        }();
        Tensor gdO = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return local_tile(make_mix_tensor_like(mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
            } else {
                return local_tile(mdO, select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
            }
        }();
        Tensor gK = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return local_tile(make_mix_tensor_like(mK),  select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
            } else {
                return local_tile(mK,  select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
            }
        }();
        Tensor gV = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return local_tile(make_mix_tensor_like(mV),  select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
            } else {
                return local_tile(mV,  select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
            }
        }();
        Tensor gLSE   = local_tile(mLSE,   select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdPsum = local_tile(mdPsum, select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdQaccum = local_tile(mdQaccum, Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)

        // ── Gmem tiled copies ──
        typename CollectiveMainloop::GmemTiledCopyQ  gmem_tiled_copy_Q;
        typename CollectiveMainloop::GmemTiledCopyKV gmem_tiled_copy_K;
        typename CollectiveMainloop::GmemTiledCopyKV gmem_tiled_copy_V;
        typename CollectiveMainloop::GmemTiledCopyQ  gmem_tiled_copy_dO;

#if defined(USE_PPU) && USE_AIU
        // PPU AIU descriptor initialization (required before any AIU copy).
        // ISOLATED fp32 kernel set: only 16-bit instantiations own an AIU
        // descriptor; float's stock CP_ASYNC tiled copy has no desc_.
        if constexpr (CollectiveMainloop::Use_aiu) {
        if constexpr (ArchTag::kMinComputeCapability >= 89) {
            if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                // CVT path: 8x64 tile desc for direct AIU copy loops.
                // gmem block offset is handled by the direct AIU copy loop.
                static constexpr int kBlockNPerAiuLoad = 8;
                gmem_tiled_copy_Q.desc_.init(nullptr, kBlockNPerAiuLoad, kBlockKGmem, get<0>(ml.stride_Q));
                gmem_tiled_copy_K.desc_.init(nullptr, kBlockNPerAiuLoad, CollectiveMainloop::kBlockNGmem, get<0>(ml.stride_K));
                gmem_tiled_copy_V.desc_.init(nullptr, kBlockNPerAiuLoad, CollectiveMainloop::kBlockNGmem, get<0>(ml.stride_V));
                gmem_tiled_copy_dO.desc_.init(nullptr, kBlockNPerAiuLoad, kBlockKGmem, get<0>(ml.stride_dO));
            } else {
                gmem_tiled_copy_Q.desc_.init(nullptr, seqlen_info.seqlen_q, get<1>(ml.shape_Q), get<0>(ml.stride_Q));
                gmem_tiled_copy_K.desc_.init(nullptr, seqlen_info.seqlen_k, get<1>(ml.shape_Q), get<0>(ml.stride_K));
                gmem_tiled_copy_V.desc_.init(nullptr, seqlen_info.seqlen_k, get<1>(ml.shape_V), get<0>(ml.stride_V));
                gmem_tiled_copy_dO.desc_.init(nullptr, seqlen_info.seqlen_q, get<1>(ml.shape_V), get<0>(ml.stride_dO));
            }
        } else {
            int aiu_offset_q = get<1>(ml.shape_Q) == kHeadDim ? 0 : (get<0>(ml.stride_Q) - get<1>(ml.shape_Q));
            int aiu_offset_k = get<1>(ml.shape_Q) == kHeadDim ? 0 : (get<0>(ml.stride_K) - get<1>(ml.shape_Q));
            int aiu_offset_v = get<1>(ml.shape_V) == kHeadDim ? 0 : (get<0>(ml.stride_V) - get<1>(ml.shape_V));
            int aiu_offset_do = get<1>(ml.shape_V) == kHeadDim ? 0 : (get<0>(ml.stride_dO) - get<1>(ml.shape_V));
            using AiuDesc = decltype(gmem_tiled_copy_Q.desc_);
            gmem_tiled_copy_Q.desc_  = AiuDesc{nullptr, seqlen_info.seqlen_q, get<0>(ml.stride_Q),  kBlockM, kBlockKGmem, aiu_offset_q};
            gmem_tiled_copy_K.desc_  = AiuDesc{nullptr, seqlen_info.seqlen_k, get<0>(ml.stride_K),  kBlockN, kBlockKGmem, aiu_offset_k};
            gmem_tiled_copy_V.desc_  = AiuDesc{nullptr, seqlen_info.seqlen_k, get<0>(ml.stride_V),  kBlockN, kBlockKGmem, aiu_offset_v};
            gmem_tiled_copy_dO.desc_ = AiuDesc{nullptr, seqlen_info.seqlen_q, get<0>(ml.stride_dO), kBlockM, kBlockKGmem, aiu_offset_do};
        }
        }
        const int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
        const int tid_thread_slice = CollectiveMainloop::Use_aiu ? warp_idx * 32 : thread_idx;
#if PPU1v0_R2S_SLICE_LAYOUT
        auto r2S_thread_idx_PdS = thread_idx;
        if (warp_idx % 2) { // keep slice0 layout
            r2S_thread_idx_PdS = __shfl_sync(0xffffffff, r2S_thread_idx_PdS, (r2S_thread_idx_PdS & 31) ^ CollectiveMainloop::PPUChannelSliceCount); // simulate slice1 layout
        }
#elif PPU1v5_R2S_SLICE_LAYOUT
        auto r2S_thread_idx_PdS = thread_idx;
#endif
#else
        const int tid_thread_slice = thread_idx;
#endif

        auto gmem_thr_copy_Q  = gmem_tiled_copy_Q.get_thread_slice(thread_idx);
        auto gmem_thr_copy_K  = gmem_tiled_copy_K.get_thread_slice(thread_idx);
        auto gmem_thr_copy_V  = gmem_tiled_copy_V.get_thread_slice(thread_idx);
        auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_Q  = gmem_tiled_copy_Q.get_thread_slice(_0{});
        auto gmem_thr0_copy_KV = gmem_tiled_copy_K.get_thread_slice(_0{});
        typename CollectiveMainloop::GmemTiledCopyLSE gmem_tiled_copy_lse;
        auto gmem_thr_copy_lse = gmem_tiled_copy_lse.get_thread_slice(thread_idx);
        typename CollectiveMainloop::R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
        auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);

        Tensor tQgQ   = gmem_thr_copy_Q.partition_S(gQ);
        Tensor tQsQ   = gmem_thr_copy_Q.partition_D(sQ);
        Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
        Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
        Tensor tLSEgLSE   = gmem_thr_copy_lse.partition_S(gLSE);
        Tensor tLSEsLSE   = gmem_thr_copy_lse.partition_D(sLSE);
        Tensor tLSEgdPsum = gmem_thr_copy_lse.partition_S(gdPsum);
        Tensor tLSEsdPsum = gmem_thr_copy_lse.partition_D(sdPsum);
        Tensor tdQgdQaccum = r2s_thr_copy_dQaccum.partition_D(gdQaccum);

        // K/V partitions (gmem source keeps the n_block mode; smem is single-stage).
        Tensor tKgK = gmem_thr_copy_K.partition_S(gK);
        Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
        Tensor tVgV = gmem_thr_copy_V.partition_S(gV);
        Tensor tVsV = gmem_thr_copy_V.partition_D(sV);

        TiledMmaSdP tiled_mma_SdP;
        TiledMmadKV tiled_mma_dKV;
        TiledMmadQ  tiled_mma_dQ;

        auto thr_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto thr_mma_dKV = tiled_mma_dKV.get_thread_slice(thread_idx);
        auto thr_mma_dQ  = tiled_mma_dQ.get_thread_slice(thread_idx);

        // Copy Atom retiling (identical to LoopQ).
        auto smem_copy_atom_SdP_B = cute::conditional_return<CollectiveMainloop::MmaSdPEvenN>(
            typename CollectiveMainloop::SmemCopyAtom{}, typename CollectiveMainloop::SmemCopyAtomHalf{});
#if defined(USE_PPU) && USE_AIU
        auto smem_tiled_copy_QdO = cute::conditional_return<!SdP_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomQ{}, tiled_mma_SdP),
            make_tiled_copy_B(typename CollectiveMainloop::SmemCopyAtomQ{}, tiled_mma_SdP));
#else
        auto smem_tiled_copy_QdO = cute::conditional_return<!SdP_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtom{}, tiled_mma_SdP),
            make_tiled_copy_B(smem_copy_atom_SdP_B, tiled_mma_SdP));
#endif
        auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tid_thread_slice);
        Tensor tSsQ = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_QdO.partition_S(make_mix_tensor_like(sQ)); }
            else { return smem_thr_copy_QdO.partition_S(sQ); }
        }();
        Tensor tdPsdO = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_QdO.partition_S(make_mix_tensor_like(sdO)); }
            else { return smem_thr_copy_QdO.partition_S(sdO); }
        }();

#if defined(USE_PPU) && USE_AIU
        auto smem_tiled_copy_KV = cute::conditional_return<!SdP_swapAB>(
            make_tiled_copy_B(typename CollectiveMainloop::SmemCopyAtomK{}, tiled_mma_SdP),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomK{}, tiled_mma_SdP));
#else
        auto smem_tiled_copy_KV = cute::conditional_return<!SdP_swapAB>(
            make_tiled_copy_B(smem_copy_atom_SdP_B, tiled_mma_SdP),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtom{}, tiled_mma_SdP));
#endif
        auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tid_thread_slice);
        Tensor tSsK = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_KV.partition_S(make_mix_tensor_like(sK)); }
            else { return smem_thr_copy_KV.partition_S(sK); }
        }();
        Tensor tdPsV = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_KV.partition_S(make_mix_tensor_like(sV)); }
            else { return smem_thr_copy_KV.partition_S(sV); }
        }();

        auto r2s_tiled_copy_PdS = make_tiled_copy_C(typename CollectiveMainloop::R2SCopyAtomPdS{}, tiled_mma_SdP);
#if PPU1v0_R2S_SLICE_LAYOUT || PPU1v5_R2S_SLICE_LAYOUT
        auto r2s_thr_copy_PdS = r2s_tiled_copy_PdS.get_thread_slice(r2S_thread_idx_PdS);
        // ISOLATED fp32 kernel set: the slice-format P/dS views are the AIU
        // TSM read-back geometry; float writes/reads the standard swizzled
        // PdS layout instead (both sides must agree on the layout).
        Tensor tdSsdS = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS_slice, sdSt_slice));
            } else {
                return r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS, sdSt));
            }
        }();
        Tensor tPsP = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) {
                return r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP_slice, sPt_slice));
            } else {
                return r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP, sPt));
            }
        }();
#else
        auto r2s_thr_copy_PdS = r2s_tiled_copy_PdS.get_thread_slice(thread_idx);
        Tensor tdSsdS = r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS, sdSt));
        Tensor tPsP   = r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP, sPt));
#endif

#if PPU1v0_R2S_SLICE_LAYOUT || PPU1v5_R2S_SLICE_LAYOUT
        auto smem_copy_atom_dKV_B = cute::conditional_return<CollectiveMainloop::MmadKVEvenN>(
            typename CollectiveMainloop::SmemCopyAtomPdSt_AIU{}, typename CollectiveMainloop::SmemCopyAtomTransposedHalf{});
        auto smem_tiled_copy_PdSt = cute::conditional_return<!dKV_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomPdSt_AIU{}, tiled_mma_dKV),
            make_tiled_copy_B(smem_copy_atom_dKV_B, tiled_mma_dKV));
        auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tid_thread_slice);
        Tensor tdVsPt = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_PdSt.partition_S(make_mix_tensor_like(sPt)); }
            else { return smem_thr_copy_PdSt.partition_S(sPt); }
        }();
        Tensor tdKsdSt = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_PdSt.partition_S(make_mix_tensor_like(sdSt)); }
            else { return smem_thr_copy_PdSt.partition_S(sdSt); }
        }();
#else
        auto smem_copy_atom_dKV_B = cute::conditional_return<CollectiveMainloop::MmadKVEvenN>(
            typename CollectiveMainloop::SmemCopyAtomTransposed{}, typename CollectiveMainloop::SmemCopyAtomTransposedHalf{});
        auto smem_tiled_copy_PdSt = cute::conditional_return<!dKV_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomTransposed{}, tiled_mma_dKV),
            make_tiled_copy_B(smem_copy_atom_dKV_B, tiled_mma_dKV));
        auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(thread_idx);
        Tensor tdVsPt  = smem_thr_copy_PdSt.partition_S(sPt);
        Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);
#endif

#if defined(USE_PPU) && USE_AIU
        auto smem_tiled_copy_QdOt = cute::conditional_return<!dKV_swapAB>(
            make_tiled_copy_B(typename CollectiveMainloop::SmemCopyAtomQt{}, tiled_mma_dKV),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomQt{}, tiled_mma_dKV));
#else
        auto smem_tiled_copy_QdOt = cute::conditional_return<!dKV_swapAB>(
            make_tiled_copy_B(smem_copy_atom_dKV_B, tiled_mma_dKV),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomTransposed{}, tiled_mma_dKV));
#endif
        auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tid_thread_slice);
        Tensor tdVsdOt = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_QdOt.partition_S(make_mix_tensor_like(sdOt)); }
            else { return smem_thr_copy_QdOt.partition_S(sdOt); }
        }();
        Tensor tdKsQt = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_QdOt.partition_S(make_mix_tensor_like(sQt)); }
            else { return smem_thr_copy_QdOt.partition_S(sQt); }
        }();

#if PPU1v0_R2S_SLICE_LAYOUT || PPU1v5_R2S_SLICE_LAYOUT
        auto smem_tiled_copy_dS = cute::conditional_return<!dQ_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomdS_AIU{}, tiled_mma_dQ),
            make_tiled_copy_B(cute::conditional_return<CollectiveMainloop::MmadQEvenN>(
                typename CollectiveMainloop::SmemCopyAtom{}, typename CollectiveMainloop::SmemCopyAtomHalf{}), tiled_mma_dQ));
        auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tid_thread_slice);
        Tensor tdQsdS = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_dS.partition_S(make_mix_tensor_like(sdS)); }
            else { return smem_thr_copy_dS.partition_S(sdS); }
        }();
#else
        // Non-slice path: PdS/dS live in the standard swizzled smem layout and
        // are read back with plain LDSM atoms (no mix tensor).
        auto smem_tiled_copy_dS = cute::conditional_return<!dQ_swapAB>(
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtom{}, tiled_mma_dQ),
            make_tiled_copy_B(cute::conditional_return<CollectiveMainloop::MmadQEvenN>(
                typename CollectiveMainloop::SmemCopyAtom{}, typename CollectiveMainloop::SmemCopyAtomHalf{}), tiled_mma_dQ));
        auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(thread_idx);
        Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);
#endif

#if defined(USE_PPU) && USE_AIU
        auto smem_tiled_copy_Kt = cute::conditional_return<!dQ_swapAB>(
            make_tiled_copy_B(typename CollectiveMainloop::SmemCopyAtomKVt{}, tiled_mma_dQ),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomKVt{}, tiled_mma_dQ));
#else
        auto smem_tiled_copy_Kt = cute::conditional_return<!dQ_swapAB>(
            make_tiled_copy_B(cute::conditional_return<CollectiveMainloop::MmadQEvenN>(
                typename CollectiveMainloop::SmemCopyAtomTransposed{}, typename CollectiveMainloop::SmemCopyAtomTransposedHalf{}), tiled_mma_dQ),
            make_tiled_copy_A(typename CollectiveMainloop::SmemCopyAtomTransposed{}, tiled_mma_dQ));
#endif
        auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tid_thread_slice);
        Tensor tdQsKt = [&]() -> auto {
            if constexpr (CollectiveMainloop::Use_aiu) { return smem_thr_copy_Kt.partition_S(make_mix_tensor_like(sKt)); }
            else { return smem_thr_copy_Kt.partition_S(sKt); }
        }();

        // LSE / dPsum smem→reg partitioning.
        Tensor tSsLSEMma = [&]() -> auto {
#ifdef USE_PPU
            if constexpr (ArchTag::kMinComputeCapability >= 89) {
                return group_modes<0, 2>(logical_divide(thr_mma_SdP.partition_C(sLSEMma), Shape<_2>{})(make_coord(_, make_coord(_, _0{})), _, _, _));
            } else {
                return logical_divide(thr_mma_SdP.partition_C(sLSEMma), Shape<_4>{});
            }
#else
            return logical_divide(thr_mma_SdP.partition_C(sLSEMma), Shape<_2>{});
#endif
        }();
        Tensor tSsLSE = group_modes<0, 2>(cute::conditional_return<!SdP_swapAB>(
            tSsLSEMma(make_coord(_0{}, _), _, _0{}, _),   // (2, MMA_M, PIPE)
            tSsLSEMma(make_coord(_, _0{}), _0{}, _, _))); // (2, MMA_N, PIPE)
        Tensor tSsdPsumMma = [&]() -> auto {
#ifdef USE_PPU
            if constexpr (ArchTag::kMinComputeCapability >= 89) {
                return group_modes<0, 2>(logical_divide(thr_mma_SdP.partition_C(sdPsumMma), Shape<_2>{})(make_coord(_, make_coord(_, _0{})), _, _, _));
            } else {
                return logical_divide(thr_mma_SdP.partition_C(sdPsumMma), Shape<_4>{});
            }
#else
            return logical_divide(thr_mma_SdP.partition_C(sdPsumMma), Shape<_2>{});
#endif
        }();
        Tensor tSsdPsum = group_modes<0, 2>(cute::conditional_return<!SdP_swapAB>(
            tSsdPsumMma(make_coord(_0{}, _), _, _0{}, _),
            tSsdPsumMma(make_coord(_, _0{}), _0{}, _, _)));
        static constexpr int kStatsPerThread = cute::ceil_div(decltype(size(tSsLSE))::value, 8);

        // ── Predicates ──
        Tensor cQ = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
        Tensor tQcQ  = gmem_thr_copy_Q.partition_S(cQ);
        Tensor t0QcQ = gmem_thr0_copy_Q.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(ml.shape_Q); }
        Tensor cLSE = cute::make_identity_tensor(select<0>(TileShape_MNK{}));
        Tensor tLSEcLSE = gmem_thr_copy_lse.partition_S(cLSE);
        Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdOsdO)));
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(ml.shape_dO); }

        Tensor cKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
        Tensor tKVcKV  = gmem_thr_copy_K.partition_S(cKV);
        Tensor t0KVcKV = gmem_thr0_copy_KV.partition_S(cKV);
        Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));
        Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tVsV)));
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(ml.shape_K); }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(ml.shape_V); }
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(typename CollectiveMainloop::GmemLayoutAtom{})) == 0;

        static constexpr bool EvenM = kBlockM % CUTE_STATIC_V(shape<0>(typename CollectiveMainloop::GmemLayoutAtom{})) == 0;

        // ── Load Q / dO / LSE / dPsum ONCE (fixed for this m_block) ──
        auto load_Q_LSE = [&] (int const m_blk, int const smem_pipe_write) {
            Tensor tQsQ_cur = tQsQ(_, _, _, smem_pipe_write);
            Tensor tQgQ_cur = tQgQ(_, _, _, m_blk);
            int const seqlenq_row_limit = seqlen_info.seqlen_q - m_blk * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}));
#if defined(USE_PPU) && USE_AIU
            if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for Q.
                // Hierarchical tiling: 8x64 -> 16x64 (vertical x2) -> 16x128 (horizontal x2)
                //                      -> blockM x128 (vertical) -> blockM x headdim (horizontal)
                // offset(m,k) = (k/2)*kHalfSize + (m/2)*kTile16x128 + (k%2)*kTile16x64 + (m%2)*kAtomSize
                static_assert(kBlockKGmem == 64, "AIU hierarchical layout assumes 8x64 atom");
                const Element* block_q_ptr = mQ.data().get()
                    + (seqlen_info.offset_q + m_blk * kBlockM) * get<0>(ml.stride_Q);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumMTiles = kBlockM / kAiuTileH;
                static constexpr int kTotalTiles = kNumMTiles * kNumKTilesTotal;
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile16x64 = 2 * kAtomSize;                         // 16x64 = 1024
                static constexpr int kTile16x128 = 2 * kTile16x64;                       // 16x128 = 2048
                static constexpr int kHalfSize = kBlockM * 2 * kAiuTileW;                // blockM x 128
                static constexpr int kLog2NumKTiles = __builtin_ctz(kNumKTilesTotal);
                static constexpr int kLog2BlockKGmem = __builtin_ctz(kBlockKGmem);
                static constexpr int kLog2AtomSize = __builtin_ctz(kAtomSize);
                static constexpr int kLog2Tile16x64 = __builtin_ctz(kTile16x64);
                static constexpr int kLog2Tile16x128 = __builtin_ctz(kTile16x128);
                static constexpr int kMStride = CollectiveMainloop::NumMmaWarps >> kLog2NumKTiles;
                static_assert(kMStride % 2 == 0, "kMStride must be even for m&1 loop-invariance");
                static constexpr int kSmemMHalfDelta = (kMStride >> 1) << kLog2Tile16x128;

                auto const gmem_m_stride_Q = kAiuTileH * get<0>(ml.stride_Q);

                auto smem_loop_base = sQ_AIU_COPY.data() + smem_pipe_write * kBlockM * kHeadDim;
                int smem_m_half_offset;
                const Element* tile_gmem;
                {
                    int const k = warp_idx & (kNumKTilesTotal - 1);
                    int const m_init = warp_idx >> kLog2NumKTiles;
                    smem_loop_base = smem_loop_base + ((k >> 1) * kHalfSize + ((k & 1) << kLog2Tile16x64) + ((m_init & 1) << kLog2AtomSize));
                    smem_m_half_offset = (m_init >> 1) << kLog2Tile16x128;
                    tile_gmem = block_q_ptr
                        + (k << kLog2BlockKGmem)
                        + static_cast<int64_t>(m_init) * gmem_m_stride_Q;
                }

                int m_rows_cur = warp_idx >> kLog2NumKTiles;
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += CollectiveMainloop::NumMmaWarps) {
                    auto tile_smem = smem_loop_base + smem_m_half_offset;
                    // dim_h must bound the seqlen tail: AIU zero-fills rows past
                    // dim_h, keeping padded tail rows exact-0 (garbage rows would
                    // carry inf/nan into dS and poison dK/dV).
                    int const valid_rows = cute::min(kAiuTileH, cute::max(0,
                        seqlen_info.seqlen_q - (seqlen_info.offset_q + m_blk * kBlockM + m_rows_cur * kAiuTileH)));
                    if (valid_rows > 0) {
                        gmem_tiled_copy_Q.desc_.dim_h = valid_rows;
                        CollectiveMainloop::Gmem_copy_struct_Q::copy(
                            cute::raw_pointer_cast(tile_smem),
                            tile_gmem,
                            gmem_tiled_copy_Q.desc_,
                            0, 0);
                    } else {
                        uint32_t* zp = reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(tile_smem));
                        #pragma unroll
                        for (int i = threadIdx.x & 31; i < int(kAtomSize * sizeof(Element) / 4); i += 32) { zp[i] = 0u; }
                    }
                    tile_gmem += static_cast<int64_t>(kMStride) * gmem_m_stride_Q;
                    smem_m_half_offset += kSmemMHalfDelta;
                    m_rows_cur += kMStride;
                }
            } else {
                // Stock fp32 path: predicate rows past seqlen_q and zero-fill
                // them in smem (cp.async zfill), mirroring the AIU dim_h tail
                // zero-fill — garbage Q/dO tail rows poison dP/dS (0*inf=NaN).
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_Q, tQgQ_cur, tQsQ_cur, t0QcQ, tQpQ, seqlenq_row_limit);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tQsQ); ++m) {
                if (EvenM || m < size<1>(tQsQ) - 1 || get<0>(tQcQ(_0{}, m, _0{})) < kBlockM) {
                    bool const predicate_m = get<0>(t0QcQ(_0{}, m, _0{})) < seqlenq_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tQsQ); ++k) {
                        cute::copy(gmem_tiled_copy_Q.with(tQpQ(k) && predicate_m), tQgQ_cur(_, m, k), tQsQ_cur(_, m, k));
                    }
                }
            }
#endif
            Tensor tLSEgLSE_cur = tLSEgLSE(_, _, m_blk);
            Tensor tLSEsLSE_cur = tLSEsLSE(_, _, smem_pipe_write);
            #pragma unroll
            for (int m = 0; m < size<1>(tLSEsLSE); ++m) {
                if (get<0>(tLSEcLSE(_0{}, m)) < kBlockM) {
                    cute::copy(gmem_tiled_copy_lse, tLSEgLSE_cur(_, m), tLSEsLSE_cur(_, m));
                }
            }
        };

        auto load_dO_dPsum = [&] (int const m_blk, int const smem_pipe_write) {
            Tensor tdOsdO_cur = tdOsdO(_, _, _, smem_pipe_write);
            Tensor tdOgdO_cur = tdOgdO(_, _, _, m_blk);
            int const seqlenq_row_limit = seqlen_info.seqlen_q - m_blk * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}));
#if defined(USE_PPU) && USE_AIU
            if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for dO.
                // Hierarchical tiling: 8x64 -> 16x64 (vertical x2) -> 16x128 (horizontal x2)
                //                      -> blockM x128 (vertical) -> blockM x headdim (horizontal)
                // offset(m,k) = (k/2)*kHalfSize + (m/2)*kTile16x128 + (k%2)*kTile16x64 + (m%2)*kAtomSize
                const Element* block_do_ptr = mdO.data().get()
                    + (seqlen_info.offset_q + m_blk * kBlockM) * get<0>(ml.stride_dO);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumMTiles = kBlockM / kAiuTileH;
                static constexpr int kTotalTiles = kNumMTiles * kNumKTilesTotal;
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile16x64 = 2 * kAtomSize;                         // 16x64 = 1024
                static constexpr int kTile16x128 = 2 * kTile16x64;                       // 16x128 = 2048
                static constexpr int kHalfSize = kBlockM * 2 * kAiuTileW;                // blockM x 128
                static constexpr int kLog2NumKTiles = __builtin_ctz(kNumKTilesTotal);
                static constexpr int kLog2BlockKGmem = __builtin_ctz(kBlockKGmem);
                static constexpr int kLog2AtomSize = __builtin_ctz(kAtomSize);
                static constexpr int kLog2Tile16x64 = __builtin_ctz(kTile16x64);
                static constexpr int kLog2Tile16x128 = __builtin_ctz(kTile16x128);
                static constexpr int kMStride = CollectiveMainloop::NumMmaWarps >> kLog2NumKTiles;
                static_assert(kMStride % 2 == 0, "kMStride must be even for m&1 loop-invariance");
                static constexpr int kSmemMHalfDelta = (kMStride >> 1) << kLog2Tile16x128;

                auto const gmem_m_stride_dO = kAiuTileH * get<0>(ml.stride_dO);

                auto smem_loop_base = sdO_AIU_COPY.data() + smem_pipe_write * kBlockM * kHeadDim;
                int smem_m_half_offset;
                const Element* tile_gmem;
                {
                    int const k = warp_idx & (kNumKTilesTotal - 1);
                    int const m_init = warp_idx >> kLog2NumKTiles;
                    smem_loop_base = smem_loop_base + ((k >> 1) * kHalfSize + ((k & 1) << kLog2Tile16x64) + ((m_init & 1) << kLog2AtomSize));
                    smem_m_half_offset = (m_init >> 1) << kLog2Tile16x128;
                    tile_gmem = block_do_ptr
                        + (k << kLog2BlockKGmem)
                        + static_cast<int64_t>(m_init) * gmem_m_stride_dO;
                }

                int m_rows_cur = warp_idx >> kLog2NumKTiles;
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += CollectiveMainloop::NumMmaWarps) {
                    auto tile_smem = smem_loop_base + smem_m_half_offset;
                    // dim_h must bound the seqlen tail (see the Q copy above).
                    int const valid_rows = cute::min(kAiuTileH, cute::max(0,
                        seqlen_info.seqlen_q - (seqlen_info.offset_q + m_blk * kBlockM + m_rows_cur * kAiuTileH)));
                    if (valid_rows > 0) {
                        gmem_tiled_copy_dO.desc_.dim_h = valid_rows;
                        CollectiveMainloop::Gmem_copy_struct_Q::copy(
                            cute::raw_pointer_cast(tile_smem),
                            tile_gmem,
                            gmem_tiled_copy_dO.desc_,
                            0, 0);
                    } else {
                        uint32_t* zp = reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(tile_smem));
                        #pragma unroll
                        for (int i = threadIdx.x & 31; i < int(kAtomSize * sizeof(Element) / 4); i += 32) { zp[i] = 0u; }
                    }
                    tile_gmem += static_cast<int64_t>(kMStride) * gmem_m_stride_dO;
                    smem_m_half_offset += kSmemMHalfDelta;
                    m_rows_cur += kMStride;
                }
            } else {
                // Stock fp32 path: predicate rows past seqlen_q and zero-fill
                // them in smem (see the Q copy above).
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_dO, tdOgdO_cur, tdOsdO_cur, t0QcQ, tQpQ, seqlenq_row_limit);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tdOsdO); ++m) {
                if (EvenM || m < size<1>(tdOsdO) - 1 || get<0>(tQcQ(_0{}, m, _0{})) < kBlockM) {
                    bool const predicate_m = get<0>(t0QcQ(_0{}, m, _0{})) < seqlenq_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tdOsdO); ++k) {
                        cute::copy(gmem_tiled_copy_dO.with(tdOpdO(k) && predicate_m), tdOgdO_cur(_, m, k), tdOsdO_cur(_, m, k));
                    }
                }
            }
#endif
            Tensor tLSEgdPsum_cur = tLSEgdPsum(_, _, m_blk);
            Tensor tLSEsdPsum_cur = tLSEsdPsum(_, _, smem_pipe_write);
            #pragma unroll
            for (int m = 0; m < size<1>(tLSEsdPsum); ++m) {
                if (get<0>(tLSEcLSE(_0{}, m)) < kBlockM) {
                    cute::copy(gmem_tiled_copy_lse, tLSEgdPsum_cur(_, m), tLSEsdPsum_cur(_, m));
                }
            }
        };

        load_Q_LSE(m_block, 0);
        cute::cp_async_fence();
        load_dO_dPsum(m_block, 0);
        cute::cp_async_fence();
        flash::cp_async_wait<0>();
        __syncthreads();

        // ── K/V loading lambda (one n_block per step, single-stage smem) ──
        auto load_KV = [&] (int const n_block_cur) {
            int const seqlenk_row_limit = seqlen_k - n_block_cur * kBlockN - get<0>(tKVcKV(_0{}, _0{}, _0{}));
#if defined(USE_PPU) && USE_AIU
            if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for V.
                // Hierarchical tiling: 8x64 -> 8x128 (horizontal x2) -> blockN x128 (vertical)
                //                      -> blockN x headdim (horizontal)
                // offset(n,k) = (k/2)*kHalfSize + n*kTile8x128 + (k%2)*kAtomSize
                const Element* block_v_ptr = mV.data().get()
                    + (seqlen_info.offset_k + n_block_cur * kBlockN) * get<0>(ml.stride_V);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumNTiles = kBlockN / kAiuTileH;
                static constexpr int kTotalTiles = kNumNTiles * kNumKTilesTotal;
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile8x128 = kAiuTileH * 2 * kAiuTileW;             // 8x128 = 1024
                static constexpr int kHalfSize = kBlockN * 2 * kAiuTileW;                // blockN x 128
                static constexpr int kKTileLog = (kNumKTilesTotal == 4) ? 2
                                               : (kNumKTilesTotal == 2) ? 1 : 0;         // log2(kNumKTilesTotal)
                int const aiu_gmem_row_step = kAiuTileH * get<0>(ml.stride_V);           // 8-row gmem span
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += CollectiveMainloop::NumMmaWarps) {
                    // kNumKTilesTotal ∈ {1,2,4}: tile decode via shift/and.
                    int const n = tile_linear >> kKTileLog;
                    int const k = tile_linear & (kNumKTilesTotal - 1);
                    const Element* tile_gmem = block_v_ptr
                        + n * aiu_gmem_row_step
                        + k * kBlockKGmem;
                    auto tile_smem = sV_AIU_COPY.data()
                        + (k >> 1)*kHalfSize + n*kTile8x128 + (k & 1)*kAtomSize;
                    // dim_h must bound the seqlen tail: AIU zero-fills rows past
                    // dim_h, keeping padded tail rows exact-0 (garbage rows would
                    // carry inf/nan into dS and poison dK/dV).
                    int const valid_rows = cute::min(kAiuTileH, cute::max(0,
                        seqlen_info.seqlen_k - (seqlen_info.offset_k + n_block_cur * kBlockN + n * kAiuTileH)));
                    if (valid_rows > 0) {
                        gmem_tiled_copy_V.desc_.dim_h = valid_rows;
                        CollectiveMainloop::Gmem_copy_struct_KV::copy(
                            cute::raw_pointer_cast(tile_smem),
                            tile_gmem,
                            gmem_tiled_copy_V.desc_,
                            0, 0);
                    } else {
                        uint32_t* zp = reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(tile_smem));
                        #pragma unroll
                        for (int i = threadIdx.x & 31; i < int(kAtomSize * sizeof(Element) / 4); i += 32) { zp[i] = 0u; }
                    }
                }
            } else {
                // Stock fp32 path: predicate rows past seqlen_k and zero-fill
                // them in smem (see the dkdv kernel V copy).
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_V, tVgV(_, _, _, n_block_cur), tVsV, t0KVcKV, tVpV, seqlenk_row_limit);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tVsV); ++m) {
                if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
                    bool const predicate_n = get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tVsV); ++k) {
                        cute::copy(gmem_tiled_copy_V.with(tVpV(k) && predicate_n), tVgV(_, m, k, n_block_cur), tVsV(_, m, k));
                    }
                }
            }
#endif
            flash::cp_async_fence();
#if defined(USE_PPU) && USE_AIU
            if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for K.
                // Hierarchical tiling: 8x64 -> 8x128 (horizontal x2) -> blockN x128 (vertical)
                //                      -> blockN x headdim (horizontal)
                // offset(n,k) = (k/2)*kHalfSize + n*kTile8x128 + (k%2)*kAtomSize
                const Element* block_k_ptr = mK.data().get()
                    + (seqlen_info.offset_k + n_block_cur * kBlockN) * get<0>(ml.stride_K);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumNTiles = kBlockN / kAiuTileH;
                static constexpr int kTotalTiles = kNumNTiles * kNumKTilesTotal;
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile8x128 = kAiuTileH * 2 * kAiuTileW;             // 8x128 = 1024
                static constexpr int kHalfSize = kBlockN * 2 * kAiuTileW;                // blockN x 128
                static constexpr int kKTileLog = (kNumKTilesTotal == 4) ? 2
                                               : (kNumKTilesTotal == 2) ? 1 : 0;         // log2(kNumKTilesTotal)
                int const aiu_gmem_row_step = kAiuTileH * get<0>(ml.stride_K);           // 8-row gmem span
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += CollectiveMainloop::NumMmaWarps) {
                    // kNumKTilesTotal ∈ {1,2,4}: tile decode via shift/and.
                    int const n = tile_linear >> kKTileLog;
                    int const k = tile_linear & (kNumKTilesTotal - 1);
                    const Element* tile_gmem = block_k_ptr
                        + n * aiu_gmem_row_step
                        + k * kBlockKGmem;
                    auto tile_smem = sK_AIU_COPY.data()
                        + (k >> 1)*kHalfSize + n*kTile8x128 + (k & 1)*kAtomSize;
                    // dim_h must bound the seqlen tail (see the V copy above).
                    int const valid_rows = cute::min(kAiuTileH, cute::max(0,
                        seqlen_info.seqlen_k - (seqlen_info.offset_k + n_block_cur * kBlockN + n * kAiuTileH)));
                    if (valid_rows > 0) {
                        gmem_tiled_copy_K.desc_.dim_h = valid_rows;
                        CollectiveMainloop::Gmem_copy_struct_KV::copy(
                            cute::raw_pointer_cast(tile_smem),
                            tile_gmem,
                            gmem_tiled_copy_K.desc_,
                            0, 0);
                    } else {
                        uint32_t* zp = reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(tile_smem));
                        #pragma unroll
                        for (int i = threadIdx.x & 31; i < int(kAtomSize * sizeof(Element) / 4); i += 32) { zp[i] = 0u; }
                    }
                }
            } else {
                // Stock fp32 path: predicate rows past seqlen_k and zero-fill
                // them in smem (see the dkdv kernel V copy).
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_K, tKgK(_, _, _, n_block_cur), tKsK, t0KVcKV, tKpK, seqlenk_row_limit);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tKsK); ++m) {
                if (EvenN || m < size<1>(tKsK) - 1 || get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
                    bool const predicate_n = get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tKsK); ++k) {
                        cute::copy(gmem_tiled_copy_K.with(tKpK(k) && predicate_n), tKgK(_, m, k, n_block_cur), tKsK(_, m, k));
                    }
                }
            }
#endif
        };

        clear(tdQrdQ);

        bool any_work = false;

        // Mask for the S tile (SdP_swapAB ⇒ transposed fragment).
        MaskFlexFlash<kBlockM, kBlockN, TiledMmaSdP, /*Transposed=*/SdP_swapAB> mask;

        // ── Deterministic dK/dV turnstile state (LoopK) ──
        // Contributors to dKaccum[n_block, bidh_kv, bidb]: every CTA
        // (m_block, bidh, bidb) that computes (slice, n_block).  Fixed order:
        // (slice ascending, m_block ascending, q-head ascending).  Gap-free
        // turns via bwd_m_cover_count — dual to the fwd_valid_n_range
        // pruning used below, so every contributor is counted exactly once.
        using Barrier = ArbDetBarrier;
        int const num_batch = ml.num_batch;
        // shape_K = (seqlen_k, d, h_k, b) — h_k is get<2>, NOT get<1> (that's d).
        int const num_head_kv = get<2>(ml.shape_K);
        int const qhead_per_khead = cute::ceil_div(get<2>(ml.shape_Q), num_head_kv);
        int const head_idx_in_group = bidh - bidh_kv * qhead_per_khead;
        int* const dk_lock = !Deterministic ? nullptr : params.epilogue.dk_semaphore + (bidb * num_head_kv + bidh_kv);
        int* const dv_lock = !Deterministic ? nullptr : params.epilogue.dv_semaphore + (bidb * num_head_kv + bidh_kv);

        // Per-step dK/dV fragments (flushed every (slice, n_block) step).
        Tensor tdKrdK_step = partition_fragment_C(tiled_mma_dKV,
            select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
        Tensor tdVrdV_step = partition_fragment_C(tiled_mma_dKV,
            select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));

        // dK/dV flush destination (accum by default; with ReduceKV the host
        // passes the fp32 workspaces through the same epilogue pointers and
        // the scale multiply moves to FlexFlashAttentionBwdReduceKV).
        Tensor mdKaccum = make_tensor(make_gmem_ptr(params.epilogue.ptr_dKaccum),
            params.epilogue.shape_dKaccum, params.epilogue.stride_dKaccum)(_, bidh_kv, bidb);
        Tensor mdVaccum = make_tensor(make_gmem_ptr(params.epilogue.ptr_dVaccum),
            params.epilogue.shape_dVaccum, params.epilogue.stride_dVaccum)(_, bidh_kv, bidb);
        typename CollectiveEpilogue::R2GTiledCopydKVaccum r2g_tiled_copy_dKVaccum;
        auto r2g_thr_copy_dKVaccum = r2g_tiled_copy_dKVaccum.get_thread_slice(thread_idx);

        // ── Per-step dK/dV flush (scale here since there is no single
        //    end-of-tile dK accumulator; ReduceKV defers the scale to the
        //    reduce kernel) ──
        auto flush_dKV = [&](int n_block_cur, int turn_kv) {
            if constexpr (!ReduceKV) {
                #pragma unroll
                for (int i = 0; i < size(tdKrdK_step); ++i) { tdKrdK_step(i) *= ml.softmax_scale; }
                if constexpr (Is_dropout) {
                    // dV = rp · Pdrop^T dO (ReduceKV defers this scale to the
                    // dV postprocess, see the launch template).
                    #pragma unroll
                    for (int i = 0; i < size(tdVrdV_step); ++i) { tdVrdV_step(i) *= params.dropout.rp_dropout; }
                }
            }
            Tensor gdKaccum = local_tile(mdKaccum, Shape<Int<kBlockN * kHeadDim>>{}, make_coord(n_block_cur));
            Tensor gdVaccum = local_tile(mdVaccum, Shape<Int<kBlockN * kHeadDim>>{}, make_coord(n_block_cur));
            Tensor tdKrdK_atomic = r2g_thr_copy_dKVaccum.retile_S(tdKrdK_step);
            Tensor tdKgdK_atomic = r2g_thr_copy_dKVaccum.partition_D(gdKaccum);
            Tensor tdVrdV_atomic = r2g_thr_copy_dKVaccum.retile_S(tdVrdV_step);
            Tensor tdVgdV_atomic = r2g_thr_copy_dKVaccum.partition_D(gdVaccum);
            static_assert(CUTE_STATIC_V(size(tdKrdK_atomic)) == CUTE_STATIC_V(size(tdKgdK_atomic)));
            static_assert(CUTE_STATIC_V(size(tdVrdV_atomic)) == CUTE_STATIC_V(size(tdVgdV_atomic)));
            if constexpr (Deterministic) {
                Barrier::wait_eq(dk_lock, thread_idx, n_block_cur * num_batch * num_head_kv, turn_kv);
            }
            #pragma unroll
            for (int i = 0; i < size(tdKrdK_atomic); ++i) { atomicAdd(&tdKgdK_atomic(i), tdKrdK_atomic(i)); }
            if constexpr (Deterministic) {
                Barrier::arrive_inc(dk_lock, thread_idx, n_block_cur * num_batch * num_head_kv);
                Barrier::wait_eq(dv_lock, thread_idx, n_block_cur * num_batch * num_head_kv, turn_kv);
            }
            #pragma unroll
            for (int i = 0; i < size(tdVrdV_atomic); ++i) { atomicAdd(&tdVgdV_atomic(i), tdVrdV_atomic(i)); }
            if constexpr (Deterministic) {
                Barrier::arrive_inc(dv_lock, thread_idx, n_block_cur * num_batch * num_head_kv);
            }
        };

        // ── Outer slice loop (ascending slice_idx — required for the
        //    deterministic dK/dV turn order) ──
        for (int slice_idx = 0; slice_idx < sl.num_slices; ++slice_idx) {
            int const q_start = sl.q_starts[slice_idx];
            int const q_end   = min(sl.q_ends[slice_idx], seqlen_q);
            if (q_end <= m_block * kBlockM || q_start >= (m_block + 1) * kBlockM) { continue; }

            int const ks = sl.k_starts[slice_idx];
            int const ke = sl.k_ends[slice_idx];
            int const ks_eff = max(ks, 0);
            int const ke_eff = min(ke, seqlen_k);
            if (ke_eff <= ks_eff) { continue; }

            int const mask_type       = sl.mask_types[slice_idx];
            int const diagonal_offset = sl.diagonal_offsets[slice_idx];
            int const band_width      = sl.band_widths[slice_idx];

            // Inner n_block range: slice K range ∩ mask region (dual of the
            // LoopQ m-cover pruning; mask.apply() keeps running per tile).
            int n_lo_blk = ks_eff / kBlockN;
            int n_hi_blk = cute::ceil_div(ke_eff, kBlockN);
            fwd_valid_n_range(mask_type, diagonal_offset, band_width,
                              m_block * kBlockM, min(m_block * kBlockM + kBlockM, seqlen_q),
                              kBlockN, n_lo_blk, n_hi_blk);
            if (n_hi_blk <= n_lo_blk) { continue; }  // fully-masked (slice, m_block)
            int const n_cnt = n_hi_blk - n_lo_blk;

            // Direction-aware iteration (LoopK: kDirEff == kDir — the
            // deterministic turns only need per-(slice, n_block) monotonicity,
            // which the ascending outer slice loop already provides).
            auto blk_item = [&] (int k, int lo, int hi) {
                return (kDirEff == DispatchDirection::MaxToMin) ? hi - 1 - k : lo + k;
            };

            auto mask_fn = [&](auto& tSrS, int n_block_cur) {
                mask.template apply</*Seqlenk_mask=*/true>(
                    tSrS, m_block, n_block_cur,
                    mask_type, ks, ke,
                    diagonal_offset, band_width,
                    q_start, q_end,
                    thread_idx, seqlen_q, seqlen_k,
                    sl.mask_bits, sl.mask_row_stride);
            };

            // ── One bwd step for a single (slice, n_block) tile ──
            auto bwd_step_k = [&](int n_block_cur) {
                Tensor tSrS = partition_fragment_C(tiled_mma_SdP,
                    select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
                clear(tSrS);
                Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(thr_mma_SdP, sQ(_, _, _0{}));
                Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(thr_mma_SdP, sK);
                /** recompute s=Q*K^T (Q fixed in smem stage 0, K per-step) */
                flash::gemm_sm80<false /*A_in_regs*/, false /*B_in_regs*/, SdP_swapAB>(
                    tSrS, tSrQ, tSrK, tSsQ(_, _, _, _0{}), tSsK,
                    tiled_mma_SdP, smem_tiled_copy_QdO, smem_tiled_copy_KV,
                    smem_thr_copy_QdO, smem_thr_copy_KV, nullptr /*hook*/);
                Tensor tLSErLSE = cute::conditional_return<!ShuffleLSE>(
                    make_fragment_like(tSsLSE(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
                if constexpr (!ShuffleLSE) {
                    cute::copy(tSsLSE(_, _0{}), tLSErLSE);
                } else {
                    #pragma unroll
                    for (int i = 0; i < kStatsPerThread; ++i) {
                        tLSErLSE(i) = tSsLSE((thread_idx % 32) / 4 + i * 8, _0{});
                    }
                }

                // ── Softcap: S = tanh(S * scale/cap), then dtanh BEFORE mask ──
                if constexpr (Has_softcap) { flash::apply_softcap(tSrS, ml.softcap_val); }
                Tensor scores = make_tensor(tSrS.data(),
                    flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
                auto dtanh = [&] { if constexpr (Has_softcap) return flash::calculate_dtanh(scores); else return nullptr; }();

                mask_fn(tSrS, n_block_cur);

                /** recompute P = softmax(S) using fwd LSE */
                #pragma unroll
                for (int mi = 0; mi < size<0>(scores); ++mi) {
                    float const lse_scaled = [&] {
                        if constexpr (!ShuffleLSE) return tLSErLSE(mi);
                        else return __shfl_sync(0xffffffff, tLSErLSE(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                    }();
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ++ni) {
                        scores(mi, ni) = exp2f(scores(mi, ni) * ml.softmax_scale_log2 - lse_scaled);
                    }
                }

                // ── Dropout replay (LoopK): same stream as fwd/LoopM ──
                if constexpr (Is_dropout) {
                    DropoutFlexFlash<kBlockM, kBlockN, TiledMmaSdP, SdP_swapAB> dropout(
                        rng_seed_eff,
                        rng_offset_base_eff
                            + static_cast<unsigned long long>(bidb) * params.dropout.num_heads + bidh,
                        params.dropout.p_keep_in_uint8_t);
                    dropout.apply(tSrS, m_block, n_block_cur, thread_idx);
                }

                /** dP = dO*V^T (dO fixed in smem stage 0, V per-step) */
                Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP,
                    select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
                clear(tdPrdP);
                Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(thr_mma_SdP, sdO(_, _, _0{}));
                Tensor tdPrV_cur = mma_partition_fragment_AB</*A=*/SdP_swapAB>(thr_mma_SdP, sV);
                flash::gemm_sm80<false /*A_in_regs*/, false /*B_in_regs*/, SdP_swapAB>(
                    tdPrdP, tdPrdO, tdPrV_cur, tdPsdO(_, _, _, _0{}), tdPsV,
                    tiled_mma_SdP, smem_tiled_copy_QdO, smem_tiled_copy_KV,
                    smem_thr_copy_QdO, smem_thr_copy_KV, nullptr /*hook*/);
                Tensor tLSErdPsum = cute::conditional_return<!ShuffledPsum>(
                    make_fragment_like(tSsdPsum(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
                if constexpr (!ShuffledPsum) {
                    cute::copy(tSsdPsum(_, _0{}), tLSErdPsum);
                } else {
                    #pragma unroll
                    for (int i = 0; i < kStatsPerThread; ++i) {
                        tLSErdPsum(i) = tSsdPsum((thread_idx % 32) / 4 + i * 8, _0{});
                    }
                }

                /** dS = P * (dP - dPsum) [* dtanh with softcap]
                    Dropout: rp·dP convention, same as the LoopM mainloop. */
                Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
                #pragma unroll
                for (int mi = 0; mi < size<0>(dS); ++mi) {
                    float const dP_sum_cur = [&] {
                        if constexpr (!ShuffledPsum) return tLSErdPsum(mi);
                        else return __shfl_sync(0xffffffff, tLSErdPsum(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                    }();
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(dS); ++ni) {
                        float dp = dS(mi, ni);
                        if constexpr (Is_dropout) { dp *= params.dropout.rp_dropout; }
                        dS(mi, ni) = scores(mi, ni) * (dp - dP_sum_cur);
                        if constexpr (Has_softcap) { dS(mi, ni) *= dtanh(mi, ni); }
                    }
                }

                // Convert scores from fp32 to fp16/bf16
                Tensor rP = make_tensor_like<Element>(tSrS);
                flash::convert_type_out(tSrS, rP);
                if constexpr (!Mma_dKV_is_RS) {
                    Tensor tPaP = r2s_thr_copy_PdS.retile_S(rP);
                    cute::copy(r2s_tiled_copy_PdS, tPaP, tPsP);
                }
                Tensor rdS = make_tensor_like<Element>(tdPrdP);
                flash::convert_type_out(tdPrdP, rdS);
                if constexpr (!Mma_dKV_is_RS) { __syncthreads(); }
                Tensor tdSadS = r2s_thr_copy_PdS.retile_S(rdS);
                cute::copy(r2s_tiled_copy_PdS, tdSadS, tdSsdS);

                /** dV = P^T*dO */
                clear(tdVrdV_step);
                Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(thr_mma_dKV, sdOt(_, _, _0{}));
                Tensor tdVsdO_cur = tdVsdOt(_, _, _, _0{});
                if constexpr (Mma_dKV_is_RS) {
#ifdef USE_PPU
                    Tensor tdVrP_acc = [&]() -> auto {
                        if constexpr (ArchTag::kMinComputeCapability >= 89) {
                            return make_tensor_like<Element>(tSrS);
                        } else {
                            return flash::convert_acc<Element>(tSrS);
                        }
                    }();
                    Tensor tdVrP = [&]() -> auto {
                        if constexpr (ArchTag::kMinComputeCapability >= 89) {
                            return make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
                        } else {
                            return make_tensor(tdVrP_acc.data(), make_layout(get<0>(tSrQ.layout()), get<1>(tSrS.layout()), get<2>(tSrS.layout())));
                        }
                    }();
#else
                    Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
#endif
                    flash::gemm_rs_sm80(tdVrdV_step, tdVrP, tdVrdO, tdVsdO_cur,
                                        tiled_mma_dKV, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
                } else {
                    Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(thr_mma_dKV, sPt);
                    flash::gemm_sm80<false, false, /*SwapAB=*/dKV_swapAB>(
                        tdVrdV_step, tdVrP, tdVrdO, tdVsPt, tdVsdO_cur,
                        tiled_mma_dKV, smem_tiled_copy_PdSt, smem_tiled_copy_QdOt,
                        smem_thr_copy_PdSt, smem_thr_copy_QdOt, nullptr);
                }
                __syncthreads();  // make sure sdS is written
                /** dQ = dS*K (register-accumulated across all steps/slices) */
                Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(thr_mma_dQ, sdS);
                Tensor tdQrK  = mma_partition_fragment_AB</*A=*/dQ_swapAB>(thr_mma_dQ, sKt);
                flash::gemm_sm80<false /*A_in_regs*/, false /*B_in_regs*/, /*SwapAB=*/dQ_swapAB>(
                    tdQrdQ, tdQrdS, tdQrK, tdQsdS, tdQsKt, tiled_mma_dQ,
                    smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt, nullptr);
                /** dK = dS^T * Q */
                clear(tdKrdK_step);
                Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(thr_mma_dKV, sQt(_, _, _0{}));
                Tensor tdKsQ_cur = tdKsQt(_, _, _, _0{});
                if constexpr (Mma_dKV_is_RS) {
#ifdef USE_PPU
                    Tensor tdKrdS_acc = [&]() -> auto {
                        if constexpr (ArchTag::kMinComputeCapability >= 89) {
                            return make_tensor_like<Element>(tdPrdP);
                        } else {
                            return flash::convert_acc<Element>(tdPrdP);
                        }
                    }();
                    Tensor tdKrdS = [&]() -> auto {
                        if constexpr (ArchTag::kMinComputeCapability >= 89) {
                            return make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
                        } else {
                            return make_tensor(tdKrdS_acc.data(), make_layout(get<0>(tdPrdO.layout()), get<1>(tdPrdP.layout()), get<2>(tdPrdP.layout())));
                        }
                    }();
#else
                    Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
#endif
                    flash::gemm_rs_sm80(tdKrdK_step, tdKrdS, tdKrQ, tdKsQ_cur,
                                        tiled_mma_dKV, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
                } else {
                    Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(thr_mma_dKV, sdSt);
                    flash::gemm_sm80<false, false, /*SwapAB=*/dKV_swapAB>(
                        tdKrdK_step, tdKrdS, tdKrQ, tdKsdSt, tdKsQ_cur,
                        tiled_mma_dKV, smem_tiled_copy_PdSt, smem_tiled_copy_QdOt,
                        smem_thr_copy_PdSt, smem_thr_copy_QdOt, nullptr);
                }
            };

            // ── Stage the first n_block's K/V, then iterate ──
            load_KV(blk_item(0, n_lo_blk, n_hi_blk));
            cute::cp_async_fence();

            CUTLASS_PRAGMA_NO_UNROLL
            for (int step = 0; step < n_cnt; ++step) {
                int const n_block_cur = blk_item(step, n_lo_blk, n_hi_blk);
                flash::cp_async_wait<0>();
                __syncthreads();

                bwd_step_k(n_block_cur);

                // Deterministic turn: gap-free rank of (slice, m_block, head)
                // among all contributors of dKaccum[n_block_cur] — prior
                // slices' cover counts plus the within-slice m rank.
                int turn_kv = 0;
                if constexpr (Deterministic) {
                    for (int s = 0; s < slice_idx; ++s) {
                        turn_kv += qhead_per_khead * bwd_m_cover_count(
                            sl.mask_types[s], sl.diagonal_offsets[s],
                            sl.band_widths[s],
                            sl.q_starts[s], sl.q_ends[s],
                            sl.k_starts[s], sl.k_ends[s],
                            n_block_cur, kBlockM, kBlockN, seqlen_q, seqlen_k);
                    }
                    int m_lo_c, m_hi_c;
                    bwd_m_cover_interval(mask_type, diagonal_offset, band_width,
                                         q_start, q_end, ks, ke,
                                         n_block_cur * kBlockN, (n_block_cur + 1) * kBlockN,
                                         kBlockM, seqlen_q, seqlen_k, m_lo_c, m_hi_c);
                    turn_kv += qhead_per_khead * (m_block - m_lo_c) + head_idx_in_group;
                }
                flush_dKV(n_block_cur, turn_kv);

                __syncthreads();  // smem_k/smem_v fully consumed; safe to overwrite
                if (step + 1 < n_cnt) {
                    load_KV(blk_item(step + 1, n_lo_blk, n_hi_blk));
                }
                cute::cp_async_fence();
            }
            any_work = true;
        }  // end slice loop

        // ── Single dQ flush: registers hold the exact (RangeMerge-ordered)
        //    sum across all slices, so one atomicAdd is bitwise
        //    deterministic — no dQ semaphore needed. ──
        if (any_work) {
            Tensor tdQrdQ_atomic = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
            static_assert(CUTE_STATIC_V(size(tdQrdQ_atomic)) == CUTE_STATIC_V(size(tdQgdQaccum(_, _, 0))));
            if constexpr (CollectiveMainloop::Use_CVT_SWZL_LD) {
                // CVT SHUFFLING_GAIT causes bit3<->bit6 headdim swap: each acc
                // half belongs to the own or partner warp's 48x8 tile, so write
                // through the matching thread_slice partition (partner = warp^4).
                constexpr int partner_warp_mask = kHeadDim == 256 ? 256 : 128;
                int partner_tid = thread_idx ^ partner_warp_mask;
                auto r2s_thr_copy_partner = r2s_tiled_copy_dQaccum.get_thread_slice(partner_tid);
                Tensor tdQgdQaccum_partner = r2s_thr_copy_partner.partition_D(gdQaccum);
                Tensor tdQgdQaccum_atomic_own = tdQgdQaccum(_, _, m_block);
                Tensor tdQgdQaccum_atomic_partner = tdQgdQaccum_partner(_, _, m_block);
                constexpr int kTotal = CUTE_STATIC_V(size(decltype(tdQrdQ_atomic){}));
                if ((thread_idx & partner_warp_mask) == 0) {
                    #pragma unroll
                    for (int i = 0; i < kTotal; ++i) {
                        if ((i % 8) < 4) { atomicAdd(&tdQgdQaccum_atomic_own(i), tdQrdQ_atomic(i)); }
                        else { atomicAdd(&tdQgdQaccum_atomic_partner(i - 4), tdQrdQ_atomic(i)); }
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < kTotal; ++i) {
                        if ((i % 8) < 4) { atomicAdd(&tdQgdQaccum_atomic_partner(i + 4), tdQrdQ_atomic(i)); }
                        else { atomicAdd(&tdQgdQaccum_atomic_own(i), tdQrdQ_atomic(i)); }
                    }
                }
            } else {
                Tensor tdQgdQaccum_atomic = tdQgdQaccum(_, _, m_block);
                #pragma unroll
                for (int i = 0; i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
            }
        }
    }

};

} // namespace flash
