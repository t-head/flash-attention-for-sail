/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Independent flex flash attention FWD kernel for PPU 1.5 (SM89).
//
// Architecture:
//   - NOT a wrapper around FA3's CollectiveMainloopFwdSm80 — this kernel has
//     its own operator() with its own slice-aware mainloop.
//   - Uses CollectiveMainloopFwdSm80 ONLY as a TYPE PROVIDER (TiledMma,
//     SmemLayoutQ/K/V, GmemTiledCopy, etc.) to pick up PPU SM89-optimized
//     MMA atoms, copy atoms, swizzle layouts, and shared memory sizing.
//   - Reuses CollectiveEpilogueFwd (safe: writes O/LSE to gmem, no coupling).
//   - Reuses flash::Softmax (online softmax + rescale_o).
//
// Methodology reference: MagiAttention FFA flash_fwd_kernel_sm90.h —
//   - Per-slice Q range dispatch via row_to_slice[] O(1) lookup
//   - K-block inner loop with configurable direction (MaxToMin / MinToMax)
//   - Zone splitting mask dispatch (no-mask / diagonal / boundary zones)
//   Adapted from Hopper (TMA + GMMA + warp specialization) to PPU 1.5
//   (cp.async + MMA + single warp group).
//
// Control flow (per work tile = (m_block, bidh, bidb)):
//   1. Load Q once (cp.async → smem_q, wait)
//   2. RangeMerge (per MagiAttention DenseBlockMeta batch loop):
//      Outer loop over all slices covering this m_block:
//        RangeMerge: vb in [row_to_vbatch_start[row], row_to_vbatch_end[row])
//                    → slice = vbatch_to_slice[vb]
//        Legacy:     single slice from row_to_slice[m_block * kBlockM]
//      The online softmax state (tOrO, row_max, row_sum) persists across
//      slices, so partial results of Q-overlapping slices are merged via LSE.
//      Per-slice K-block inner loop (direction = kDir template param),
//      software-pipelined like the native FA sm80 mainloop (kStages=1):
//        For each n_block:
//          2a. K[n_block] was prefetched by the previous iteration (or the
//              slice prologue): wait + syncthreads
//          2b. flash::gemm QK → tSrS; the V[n_block] cp.async load is issued
//              from inside the gemm (hook) so it overlaps with the QK MMA
//          2c. K[n_block+1] is issued right after the QK gemm so it overlaps
//              with mask/softmax/PV (smem_k is free once all warps finished
//              the QK smem reads)
//          2d. Zone dispatch: boundary → seqlenk_mask, diagonal → mask.apply(),
//              no-mask zone → skip mask entirely
//          2e. softmax.max_get_scale + rescale_o + online_softmax
//          2f. wait V[n_block]; flash::gemm PV → tOrO
//   3. softmax.finalize
//   4. epilogue.store (write O + LSE)

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/kernel_hardware_info.h>

#include "../hopper/seqlen.h"
#include "../hopper/utils.h"
#include "../hopper/softmax.h"
#include "../hopper/tile_scheduler.hpp"
#include "../hopper/mainloop_fwd_sm80_f32.hpp"   // TYPE PROVIDER ONLY — we don't call mma()
#include "../hopper/epilogue_fwd_f32.hpp"

#include "attn_slice.h"
#include "mask_flex_flash.h"
#include "block_meta_flex_flash.h"
#include "dropout_flex_flash.h"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          DispatchDirection kDir = DispatchDirection::MaxToMin,
          bool Has_softcap_ = false, bool Is_dropout_ = false>
class FlashAttnFlexFlashFwdSm80 {

public:

    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Is_dropout = Is_dropout_;

    // ─── TYPE PROVIDER ALIASES ───────────────────────────────────────────────
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    using TileScheduler = TileScheduler_;

    // Mainloop derived types (reused from FA3 for PPU SM89 optimization).
    using TileShape_MNK    = typename CollectiveMainloop::TileShape_MNK;
    using TileShape_MNK_PV = typename CollectiveMainloop::TileShape_MNK_PV;
    using TiledMma         = typename CollectiveMainloop::TiledMma;
    using ArchTag          = typename CollectiveMainloop::ArchTag;
    using Element          = typename CollectiveMainloop::Element;

    // Shared memory copy atoms (PPU LDSM / TSM_LD variants).
    using SmemCopyAtomQ   = typename CollectiveMainloop::SmemCopyAtomQ;
    using SmemCopyAtomK   = typename CollectiveMainloop::SmemCopyAtomK;
    using SmemCopyAtomKVt = typename CollectiveMainloop::SmemCopyAtomKVt;

    // Global memory copy atoms (cp.async or AIU desc).
    using GmemTiledCopyQ = typename CollectiveMainloop::GmemTiledCopyQ;
    using GmemTiledCopyK = typename CollectiveMainloop::GmemTiledCopyKV;
#if defined(USE_PPU) && !defined(FLASHATTENTION_DISABLE_FP8)
    using GmemTiledCopyV_t = typename CollectiveMainloop::GmemTiledCopyV;
#else
    using GmemTiledCopyV_t = typename CollectiveMainloop::GmemTiledCopyKV;
#endif

    // Shared memory layouts (with PPU-specific swizzle).
    using SmemLayoutQ  = typename CollectiveMainloop::SmemLayoutQ;
    using SmemLayoutK  = typename CollectiveMainloop::SmemLayoutK;
    using SmemLayoutV  = typename CollectiveMainloop::SmemLayoutV;
    using SmemLayoutVt = typename CollectiveMainloop::SmemLayoutVt;

    // Mainloop arguments/params types (for accessing QKV pointers, shapes, strides).
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams    = typename CollectiveMainloop::Params;

    static constexpr bool Is_causal = false;
    static constexpr bool Is_local  = false;
    static constexpr bool Varlen    = CollectiveMainloop::Varlen;
    static constexpr bool PagedKV   = CollectiveMainloop::PagedKV;
    static constexpr bool Split     = false;
    static constexpr bool Is_FP8    = CollectiveMainloop::Is_FP8;
    static constexpr bool PackGQA   = CollectiveMainloop::PackGQA;
    static constexpr bool V_colmajor = CollectiveMainloop::V_colmajor;
    static constexpr int  NumProducerThreads = CollectiveMainloop::NumMmaThreads;
    static constexpr int  kBlockKGmem = CollectiveMainloop::kBlockKGmem;
    // Q_in_regs (== Share_QV_Smem): with the kBlockM=128 tiles Q stays in
    // smem (SeparateQV storage) and is re-read per K-block; with kBlockM=64
    // tiles smem_q/smem_v share storage and Q is hoisted into registers.
    static constexpr bool Q_in_regs = CollectiveMainloop::Share_QV_Smem;

    // ISOLATED fp32 (TF32) kernel set gate: mirror the mainloop's Use_aiu so
    // float rides the stock (cp.async + LDSM) path; 16-bit builds untouched.
    static constexpr bool Use_aiu = CollectiveMainloop::Use_aiu;

    using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;

    // Epilogue derived types.
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams    = typename CollectiveEpilogue::Params;

    static_assert(ArchTag::kMinComputeCapability >= 80);

    using TileSchedulerArguments = typename flash::TileSchedulerArguments;
    using TileSchedulerParams    = typename TileScheduler::Params;

    static constexpr int kBlockM   = get<0>(TileShape_MNK{});
    static constexpr int kBlockN   = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim  = get<2>(TileShape_MNK{});
    static constexpr int kHeadDimV = get<1>(TileShape_MNK_PV{});

    static constexpr uint32_t NumThreads                 = CUTE_STATIC_V(size(TiledMma{}));
    static constexpr uint32_t MaxThreadsPerBlock         = NumThreads;
    static constexpr uint32_t MinBlocksPerMultiprocessor = NumThreads == 128 ? 2 : 1;

    // ─── SHARED MEMORY STORAGE ────────────────────────────────────────────────
    static constexpr int mainloop_smem_padding_ =
        int(sizeof(typename CollectiveEpilogue::TensorStorage))
      - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v)))
      - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k)));
    static constexpr int mainloop_smem_padding = mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;

    struct CUTE_ALIGNAS(128) TensorStorageWoPadding {
        union {
            typename CollectiveMainloop::TensorStorage mainloop;
            typename CollectiveEpilogue::TensorStorage epilogue;
        };
    };
    struct CUTE_ALIGNAS(128) TensorStorageWPadding {
        union {
            struct {
                cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
                typename CollectiveMainloop::TensorStorage mainloop;
            };
            typename CollectiveEpilogue::TensorStorage epilogue;
        };
    };
    using TensorStorage = std::conditional_t<(mainloop_smem_padding > 0),
                                             TensorStorageWPadding, TensorStorageWoPadding>;

    struct SharedStorage {
        union {
            TensorStorage tensors;
            alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
        };
    };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // ─── HOST-SIDE ARGUMENTS & DEVICE-SIDE PARAMS ────────────────────────────
    // Dropout state is passed by value; see flash::DropoutArbArgs
    // (dropout_flex_flash.h) — shared with the bwd kernel so both replay the
    // exact same Philox stream.
    using DropoutArgs = flash::DropoutArbArgs;

    struct Arguments {
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
        AttnSliceParams slices{};
        DropoutArgs dropout{};
    };

    struct Params {
        MainloopParams mainloop{};
        EpilogueParams epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
        AttnSliceParams slices{};
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
            args.dropout
        };
    }

    static dim3 get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(
            params.scheduler, params.hw_info.sm_count * MinBlocksPerMultiprocessor);
    }

#ifdef USE_PPU
    static dim3 get_grid_shape(Params const& params, int blocks_per_sm) {
        return TileScheduler::get_grid_shape(
            params.scheduler,
            params.hw_info.sm_count * (blocks_per_sm <= 1 ? MinBlocksPerMultiprocessor : blocks_per_sm));
    }
#endif

    static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

    // ─── KERNEL ENTRY POINT ──────────────────────────────────────────────────
    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf) {

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        CollectiveEpilogue epilogue;
        TileScheduler scheduler(
            reinterpret_cast<typename TileScheduler::SharedStorage*>(
                &shared_storage.smem_scheduler));

        TiledMma tiled_mma;
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
            // O accumulator (register resident).
            Tensor tOrO = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK_PV{}));

            auto block_coord = work_tile_info.get_block_coord(params.scheduler);
            int const m_block = get<0>(block_coord);
            int const bidh    = get<1>(block_coord);
            int const bidb    = get<2>(block_coord);

            // Online softmax state.
            // With softcap, the launch template already converted
            // softmax_scale_log2 to cap * log2(e), so no change here.
            flash::Softmax<2 * (2 * kBlockM / NumThreads), /*Max_offset=*/0> softmax(
                params.mainloop.softmax_scale_log2);

            // Sequence info (handles Varlen / batch offsets).
            SeqlenInfo_t seqlen_info{
                bidb,
                get<0>(params.mainloop.shape_Q),
                !PagedKV ? size<0>(params.mainloop.shape_K)
                         : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable),
                get<0>(params.mainloop.shape_K_new),
                params.mainloop.cu_seqlens_q, params.mainloop.cu_seqlens_k,
                params.mainloop.cu_seqlens_k_new,
                params.mainloop.seqused_q, params.mainloop.seqused_k,
                params.mainloop.leftpad_k, params.mainloop.seqlens_rotary
            };

            bool tile_valid = run_mainloop(
                params, shared_storage, tiled_mma, softmax,
                tOrO, thread_idx, m_block, bidh, bidb, seqlen_info);

            scheduler.prefetch_next_work(params.scheduler, work_tile_info);

            if (tile_valid) {
                if constexpr (Is_dropout) {
                    // E[out_drop] = E[mask] * out = p_keep * out, so divide
                    // by p_keep (multiply by rp_dropout) to stay unbiased.
                    float const rp = params.dropout.rp_dropout;
                    #pragma unroll
                    for (int i = 0; i < size(tOrO); ++i) { tOrO(i) *= rp; }
                }
                epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage,
                               tiled_mma, thread_idx, block_coord);
            } else {
                epilogue.store_zero(params.epilogue, thread_idx, block_coord);
            }
        }
    }

    // ─── SLICE-AWARE MAINLOOP ────────────────────────────────────────────────
    template <typename Softmax, typename FrgTensorO>
    CUTLASS_DEVICE bool run_mainloop(
        Params const& params,
        SharedStorage& shared_storage,
        TiledMma& tiled_mma,
        Softmax& softmax,
        FrgTensorO& tOrO,
        int thread_idx,
        int m_block, int bidh, int bidb,
        SeqlenInfo_t const& seqlen_info)
    {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        // ── Shared memory tensors ──
        Tensor sQ  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
                                 SmemLayoutQ{});
        Tensor sK  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
                                 SmemLayoutK{});
        Tensor sV  = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
                                 SmemLayoutV{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
                                 SmemLayoutVt{});

        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;

        // Dropout stream resolve (CUDA-graph-safe; see dropout_flex_flash.h):
        // graph mode reads the generator's extragraph tensors at kernel
        // start so every replay draws the refreshed stream, and publishes
        // the pair for the captured bwd graph.  Eager keeps the by-value
        // scalars baked at launch.
        unsigned long long rng_seed_eff = 0, rng_offset_base_eff = 0;
        if constexpr (Is_dropout) {
            arb_resolve_rng_stream(params.dropout, rng_seed_eff,
                                   rng_offset_base_eff);
            if (params.dropout.rng_state_dev != nullptr && thread_idx == 0) {
                params.dropout.rng_state_dev[0] = rng_seed_eff;
                params.dropout.rng_state_dev[1] = rng_offset_base_eff;
            }
        }

        // P3 per-(b,h) heterogeneous masks: rebase the slice layout onto
        // this (batch, head) pair's layout group.  With a shared layout
        // (bh_to_group == nullptr) this is a zero-cost identity copy, so
        // the legacy single-mask path keeps its exact semantics.
        // shape_Q = (seqlen_q, d, h, b): num_heads from get<2> (always
        // populated, unlike dropout.num_heads which is Is_dropout-only).
        AttnSliceParams const sl = rebase_slices(
            params.slices,
            bidb * cute::get<2>(params.mainloop.shape_Q) + bidh, seqlen_q);
        // rebase_slices shifts the slice ARRAY pointers by the group's slice
        // offset but the vbatch_to_slice / row_to_slice ENTRIES still hold
        // FLAT (global) slice indices — un-rebase them at the single point
        // where slice_idx is resolved (below), so they stay consistent with
        // the rebased (per-group) num_slices and array bases.
        int const slice_idx_base = params.slices.bh_to_group == nullptr
            ? 0
            : params.slices.group_slice_offsets[
                  params.slices.bh_to_group[
                      bidb * cute::get<2>(params.mainloop.shape_Q) + bidh]];

        // ── Gmem tensors (Q, K, V) ──
        bool const is_varlen_q = Varlen && params.mainloop.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.mainloop.cu_seqlens_k;
        int const bidh_kv = !PackGQA
            ? params.mainloop.qhead_per_khead_divmod.divide(bidh)
            : bidh;
        int const bidb_kv = params.mainloop.kv_batch_idx == nullptr
            ? bidb
            : params.mainloop.kv_batch_idx[bidb];

        auto shape_V = make_shape(
            get<0>(params.mainloop.shape_K),
            params.mainloop.headdim_v,
            get<2>(params.mainloop.shape_K),
            get<3>(params.mainloop.shape_K));

        Tensor mQ = make_tensor(
            make_gmem_ptr(params.mainloop.ptr_Q
                          + seqlen_info.offset_q * get<0>(params.mainloop.stride_Q)),
            params.mainloop.shape_Q_packed, params.mainloop.stride_Q_packed)(
            _, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor gQ = [&]() -> auto {
            if constexpr (Use_aiu) {
                return local_tile(make_mix_tensor_like(mQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
            } else {
                return local_tile(mQ, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
            }
        }();

        Tensor mK = make_tensor(
            make_gmem_ptr(params.mainloop.ptr_K
                          + seqlen_info.offset_k * get<0>(params.mainloop.stride_K)),
            params.mainloop.shape_K, params.mainloop.stride_K)(
            _, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor gK = [&]() -> auto {
            if constexpr (Use_aiu) {
                return local_tile(make_mix_tensor_like(mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
            } else {
                return local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
            }
        }();

        Tensor mV = make_tensor(
            make_gmem_ptr(params.mainloop.ptr_V
                          + seqlen_info.offset_k * get<0>(params.mainloop.stride_V)),
            shape_V, params.mainloop.stride_V)(
            _, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        Tensor gV = [&]() -> auto {
            if constexpr (Use_aiu) {
                return local_tile(make_mix_tensor_like(mV), select<2, 1>(TileShape_MNK_PV{}), make_coord(_, _0{}));  // (N, K, _)
            } else {
                return local_tile(mV, select<2, 1>(TileShape_MNK_PV{}), make_coord(_, _0{}));  // (N, K, _)
            }
        }();

        // ── Gmem tiled copy (cp.async / AIU) ──
        GmemTiledCopyQ gmem_tiled_copy_Q;
        GmemTiledCopyK gmem_tiled_copy_K;
        GmemTiledCopyV_t gmem_tiled_copy_V;

#if defined(USE_PPU) && USE_AIU
        // PPU SM89 AIU descriptor initialization (required before any AIU copy).
        if constexpr (Use_aiu) {
        if constexpr (ArchTag::kMinComputeCapability >= 89) {
            gmem_tiled_copy_Q.desc_.init(nullptr, seqlen_info.seqlen_q, get<1>(params.mainloop.shape_Q), get<0>(params.mainloop.stride_Q));
            gmem_tiled_copy_K.desc_.init(nullptr, seqlen_info.seqlen_k, get<1>(params.mainloop.shape_Q), get<0>(params.mainloop.stride_K));
#if !defined(FLASHATTENTION_DISABLE_FP8)
            if constexpr (V_colmajor) {
                gmem_tiled_copy_V.desc_.init(nullptr, get<1>(shape_V), seqlen_info.seqlen_k, get<1>(params.mainloop.stride_V));
            } else {
                gmem_tiled_copy_V.desc_.init(nullptr, seqlen_info.seqlen_k, get<1>(shape_V), get<0>(params.mainloop.stride_V));
            }
#else
            gmem_tiled_copy_V.desc_.init(nullptr, seqlen_info.seqlen_k, get<1>(shape_V), get<0>(params.mainloop.stride_V));
#endif
        } else {
            int aiu_offset_q = get<1>(params.mainloop.shape_Q) == kHeadDim ? 0 : (get<0>(params.mainloop.stride_Q) - get<1>(params.mainloop.shape_Q));
            int aiu_offset_k = get<1>(params.mainloop.shape_Q) == kHeadDim ? 0 : (get<0>(params.mainloop.stride_K) - get<1>(params.mainloop.shape_Q));
            int aiu_offset_v = get<1>(shape_V) == kHeadDimV ? 0 : (get<0>(params.mainloop.stride_V) - get<1>(shape_V));
            using AiuDesc = decltype(gmem_tiled_copy_Q.desc_);
            gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, seqlen_info.seqlen_q, get<0>(params.mainloop.stride_Q), kBlockM, kBlockKGmem, aiu_offset_q};
            gmem_tiled_copy_K.desc_ = AiuDesc{nullptr, seqlen_info.seqlen_k, get<0>(params.mainloop.stride_K), kBlockN, kBlockKGmem, aiu_offset_k};
            gmem_tiled_copy_V.desc_ = AiuDesc{nullptr, seqlen_info.seqlen_k, get<0>(params.mainloop.stride_V), kBlockN, kBlockKGmem, aiu_offset_v};
        }
        }
        const int tid_thread_slice = [&] {
            if constexpr (Use_aiu) { return __ppu_read_firstlane(threadIdx.x / 32) * 32; }
            else { return thread_idx; }
        }();
#else
        const int tid_thread_slice = thread_idx;
#endif

        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(thread_idx);
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(thread_idx);
        auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_Q  = gmem_tiled_copy_Q.get_thread_slice(_0{});  // For index calculation
        auto gmem_thr0_copy_KV = gmem_tiled_copy_K.get_thread_slice(_0{});  // For index calculation

        Tensor tKgK = gmem_thr_copy_K.partition_S(gK);
        Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
        Tensor tVgV = gmem_thr_copy_V.partition_S(gV);
        Tensor tVsV = gmem_thr_copy_V.partition_D(sV);

        // ── MMA fragments ──
        auto thr_mma  = tiled_mma.get_slice(thread_idx);
        Tensor tSrQ   = thr_mma.partition_fragment_A(sQ);
        Tensor tSrK   = thr_mma.partition_fragment_B(sK(_, _, _0{}));
        Tensor tOrV   = thr_mma.partition_fragment_B(sVt(_, _, _0{}));
        Tensor tSrS   = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));

        // Smem → MMA copy atoms.
        auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma);
        auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(tid_thread_slice);
        auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomK{}, tiled_mma);
        auto smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(tid_thread_slice);
        auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomKVt{}, tiled_mma);
        auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(tid_thread_slice);

        // PPU AIU: use make_mix_tensor_like for smem partition.
        Tensor tSsQ = [&]() -> auto {
            if constexpr (Use_aiu) { return smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ)); }
            else { return smem_thr_copy_Q.partition_S(sQ); }
        }();
        Tensor tSsK = [&]() -> auto {
            if constexpr (Use_aiu) { return smem_thr_copy_K.partition_S(make_mix_tensor_like(sK)); }
            else { return smem_thr_copy_K.partition_S(sK); }
        }();
        Tensor tOsVt = [&]() -> auto {
            if constexpr (Use_aiu) { return smem_thr_copy_V.partition_S(make_mix_tensor_like(sVt)); }
            else { return smem_thr_copy_V.partition_S(sVt); }
        }();

        // K/V predicates for flash::copy (bound check on head_dim).
        Tensor cK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
        Tensor tKcK = gmem_thr_copy_K.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_KV.partition_S(cK);
        Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(params.mainloop.shape_K); }
        Tensor cV = cute::make_identity_tensor(select<2, 1>(TileShape_MNK_PV{}));
        Tensor tVcV = gmem_thr_copy_V.partition_S(cV);
        Tensor t0VcV = gmem_thr0_copy_KV.partition_S(cV);
        Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tVsV)));
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(_0{}, _0{}, k)) < get<1>(shape_V); }

        // ── Load Q once per work tile ───────────────────────────────────────
        {
            Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
            Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
            Tensor cQ = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
            Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
            Tensor t0QcQ = gmem_thr0_copy_Q.partition_S(cQ);
            Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
#if defined(USE_PPU) && USE_AIU
            if constexpr (Use_aiu) {
                flash::copy</*Is_even_MN=*/true>(
                    gmem_tiled_copy_Q, tQgQ, tQsQ, t0QcQ, tQpQ);
            } else {
                #pragma unroll
                for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.mainloop.shape_Q); }
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                    gmem_tiled_copy_Q, tQgQ, tQsQ, t0QcQ, tQpQ, seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{})));
            }
#else
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.mainloop.shape_Q); }
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                gmem_tiled_copy_Q, tQgQ, tQsQ, t0QcQ, tQpQ, seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{})));
#endif
            cute::cp_async_fence();
            flash::cp_async_wait<0>();
            __syncthreads();
        }

        // ── Hoist Q into registers (Q_in_regs) ──────────────────────────────
        // CRITICAL: with Q_in_regs the mainloop TensorStorage shares smem_q
        // and smem_v (union).  Every V load below overwrites smem_q, so Q
        // must be latched into registers BEFORE the first V load and the QK
        // gemm must run with A_in_regs=true (same as the standard mainloop's
        // preprocess_Q).  Re-reading Q from smem per K-block would silently
        // use V data and corrupt S/LSE for all but the first block.
        // With !Q_in_regs (kBlockM=128 tiles) smem_q is separate and stays
        // live for the whole mainloop; the QK gemm re-reads it per K-block.
        if constexpr (Q_in_regs) {
            Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
            cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        }

        // ── Slice iteration with zone splitting ─────────────────────────────
        // Reference: MagiAttention mask_dispatch — partition K-block range into
        // zones with different masking requirements to skip mask.apply() in
        // fully-visible regions.
        //
        // RangeMerge (Magi DenseBlockMeta NeedsBatchLoop): the same Q tile may
        // be covered by multiple slices.  We iterate over all covering slices
        // with a persistent online-softmax state (tOrO / softmax carry over),
        // which merges the per-slice partial results via LSE.
        MaskFlexFlash<kBlockM, kBlockN, TiledMma> mask;

        bool any_tile_processed = false;
        bool is_first_iter = true;
        clear(tOrO);

        // Perf-debug counters (FLASH_ATTENTION_ARB_DBG=1): accumulated per
        // CTA, flushed with one atomicAdd at the end of the mainloop.
        unsigned long long *dbg = sl.dbg_counters;
        unsigned long long dbg_c0 = dbg ? clock64() : 0;
        long long dbg_steps = 0, dbg_masked = 0;

        int const row0 = m_block * kBlockM;

        // Per-(slice) K-block inner loop, shared verbatim by the generic
        // slice mechanism and the trivial single-slice specialization (the
        // body is polymorphic only over the block_meta type).  Softmax state
        // persists across run_slice() calls (RangeMerge LSE merging).
        auto run_slice = [&](auto& block_meta) {
        // Determine the last block index (for boundary/seqlenk_mask handling).
        // MaxToMin: last block = inner_block_min (processed last).
        // MinToMax: last block = inner_block_cnt - 1 (processed last).
        int const last_block = (kDir == DispatchDirection::MaxToMin)
            ? block_meta.inner_block_min
            : block_meta.inner_block_cnt - 1;
        bool const last_block_is_seqlen_boundary =
            (last_block * kBlockN + kBlockN > seqlen_k);

        // ── Pipelined block loads (mirrors the native FA sm80 mainloop with
        // kStages=1): K[next] is issued right after the QK gemm (overlapping
        // with mask/softmax/PV) and V[cur] is issued from inside the QK gemm
        // hook (overlapping with the QK MMA).  Fences are ALWAYS issued even
        // when a load is skipped so the cp.async commit-group count stays
        // fixed and the cp_async_wait<N> below waits for the right group.
        auto load_K_block = [&] (int const nb) {
            Tensor tKsK_cur = tKsK(_, _, _, 0);
#if defined(USE_PPU) && USE_AIU
            if constexpr (Use_aiu) {
                flash::copy</*Is_even_MN=*/true>(
                    gmem_tiled_copy_K, tKgK(_, _, _, nb), tKsK_cur, t0KcK, tKpK);
            } else {
                // fp32 stock path: mask the tail K-block rows (seqlen_k may not
                // be a multiple of kBlockN) to avoid OOB cp.async reads.
                int const seqlenk_row_limit = -int(get<0>(tKcK(_0{}, _0{}, _0{})))
                    + std::min(seqlen_k - nb * kBlockN, kBlockN);
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                    gmem_tiled_copy_K, tKgK(_, _, _, nb), tKsK_cur, t0KcK, tKpK, seqlenk_row_limit);
            }
#else
            static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(typename CollectiveMainloop::GmemLayoutAtom{})) == 0;
            flash::copy</*Is_even_MN=*/EvenN, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                gmem_tiled_copy_K, tKgK(_, _, _, nb), tKsK_cur, t0KcK, tKpK);
#endif
        };
        auto load_V_block = [&] (int const nb) {
            Tensor tVsV_cur = tVsV(_, _, _, 0);
#if defined(USE_PPU) && USE_AIU
            if constexpr (Use_aiu) {
                flash::copy</*Is_even_MN=*/true>(
                    gmem_tiled_copy_V, tVgV(_, _, _, nb), tVsV_cur, t0VcV, tVpV);
            } else {
                // fp32 stock path: mask the tail V-block rows as in load_K_block.
                int const seqlenk_row_limit = seqlen_k - nb * kBlockN - get<0>(tVcV(_0{}, _0{}, _0{}));
                #pragma unroll
                for (int m = 0; m < size<1>(tVsV); ++m) {
                    bool const predicate_n = get<0>(t0VcV(_0{}, m, _0{})) < seqlenk_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tVsV); ++k) {
                        cute::copy(gmem_tiled_copy_V.with(tVpV(k) && predicate_n), tVgV(_, m, k, nb), tVsV_cur(_, m, k));
                    }
                }
            }
#else
            flash::copy</*Is_even_MN=*/true, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                gmem_tiled_copy_V, tVgV(_, _, _, nb), tVsV_cur, t0VcV, tVpV);
#endif
        };

        // K-block inner loop for this (m_block, slice) work entry.
        // Softmax state persists across vbatch iterations (LSE merging).
        //
        // Like the native mainloop, the body is a templated `step` lambda and
        // the iteration is split into phases so that the hot no-mask zone runs
        // with compile-time Check_inf=false and no per-block runtime dispatch:
        //   phase 1: blocks needing the algebraic (diagonal) mask
        //   phase 2: fully-visible blocks (no mask.apply at all)
        // The seqlenk boundary (last processed block) is handled at whichever
        // phase reaches it.
        auto step = [&](auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
            static constexpr bool Is_first_iter_t = decltype(is_first_iter_type)::value;
            static constexpr bool Check_inf = decltype(check_inf_type)::value;
            int const n_block = block_meta.inner_block_idx;
            clear(tSrS);

            // ── K[n_block]: prefetched in the previous iteration (or the
            // slice prologue above) — wait for it here.
            flash::cp_async_wait<0>();
            __syncthreads();

            // ── gemm QK: tSrS = Q @ K^T (Q in regs, K from smem) ──
            // The V[n_block] load is issued from the gemm hook so its cp.async
            // latency overlaps with the QK MMA.
            auto load_V_cur = [&]() {
                load_V_block(n_block);
                cute::cp_async_fence();
            };
            flash::gemm_sm80</*A_in_regs=*/Q_in_regs, /*B_in_regs=*/false>(
                tSrS, tSrQ, tSrK, tSsQ, tSsK(_, _, _, 0),
                tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K, load_V_cur);

            // Look ahead the next K-block within this slice (same direction).
            int n_block_next = n_block;
            advance_block_cur<kDir>(n_block_next);
            bool const has_next = !is_block_finish<kDir>(
                n_block_next, block_meta.inner_block_min, block_meta.inner_block_cnt);

            // ── Softcap: S = tanh(S * scale/cap) * cap ──
            // Must run BEFORE masking: mask fills masked entries with -inf,
            // and tanh(-inf) would be meaningless.  The mainloop params
            // already carry the premultiplied softmax_scale/cap value.
            // Issued before the V wait so its tanh/mul flops overlap the
            // in-flight V load (native mainloop ordering).
            if constexpr (Has_softcap) {
                flash::apply_softcap(tSrS, params.mainloop.softcap_val);
            }

            // All warps are done reading smem_k → issue K[next]; it overlaps
            // with the mask/softmax/PV work below.
            if constexpr (kHeadDim <= 96) {
                // Short QK gemm: defer waiting for V[n_block] until just
                // before the PV gemm (same as the native mainloop).
                __syncthreads();
                if (has_next) { load_K_block(n_block_next); }
                cute::cp_async_fence();
            } else {
#ifdef USE_PPU
                __ppu_sched_bound();   // keep the V wait out of the QK gemm
#endif
                flash::cp_async_wait<0>();   // V[n_block] arrived
                __syncthreads();
                if (has_next) { load_K_block(n_block_next); }
                cute::cp_async_fence();
            }

            // ── Mask: phase-specific (diagonal / none) + seqlenk boundary ──
            bool const is_boundary = (n_block == last_block && last_block_is_seqlen_boundary);
            if (is_boundary) {
                // Boundary block: apply both seqlenk and algebraic mask.
                mask.template apply</*Seqlenk_mask=*/true>(
                    tSrS, m_block, n_block,
                    block_meta.cur_mask_type,
                    block_meta.cur_k_start, block_meta.cur_k_end,
                    block_meta.cur_diagonal_offset, block_meta.cur_band_width,
                    block_meta.cur_q_start, block_meta.cur_q_end,
                    thread_idx, seqlen_q, seqlen_k,
                    sl.mask_bits, sl.mask_row_stride);
            } else {
                mask_fn(tSrS, n_block);
            }

            // ── Online softmax ──
            // Mirrors the native FA4SkipRescaleO pattern (always on here, the
            // kernel is bf16/fp16 only): max_get_scale_skip skips both the
            // exp2f and the row_sum rescale when the running max did not grow
            // significantly, and rescale_o is deferred until AFTER the P
            // conversion below (the PPU mma then interleaves with the
            // exp2/add/fma).  On the first iteration need_rescale=false, so
            // Is_first_iter is handled without any runtime branch here.
            auto scores_scale = softmax.template max_get_scale_skip<Is_first_iter_t, Check_inf>(tSrS);
            softmax.template online_softmax<Is_first_iter_t, Check_inf>(tSrS);
            if constexpr (Is_first_iter_t) { is_first_iter = false; }

            // ── Dropout (coordinate-keyed Philox, FA2 semantics) ──
            // Applied AFTER online softmax so row_sum accumulates the
            // UN-dropped P (LSE stays dropout-free, matching FA2); the
            // rp_dropout compensation is applied once on tOrO in operator().
            if constexpr (Is_dropout) {
                DropoutFlexFlash<kBlockM, kBlockN, TiledMma> dropout(
                    rng_seed_eff,
                    rng_offset_base_eff
                        + static_cast<unsigned long long>(bidb) * params.dropout.num_heads + bidh,
                    params.dropout.p_keep_in_uint8_t);
                dropout.apply(tSrS, m_block, n_block, thread_idx);
            }
            // Deferred rescale (FA4SkipRescaleO): placed right after the
            // softmax (native position) so the PV mma interleaves with the
            // exp2/add/fma below.
            if (softmax.need_rescale) { softmax.rescale_o(tOrO, scores_scale); }

            // ── gemm PV: tOrO += P @ V (A in regs, B from smem) ──
            // Convert QK accumulator to Element type for P operand.
#ifdef USE_PPU
            Tensor tOrP = [&]() -> auto {
                if constexpr (ArchTag::kMinComputeCapability >= 89) {
                    if constexpr (std::is_same_v<Element, float>) {
                        // TF32 m16n16k8: acc C-regs cannot be re-viewed as
                        // A-regs (t%4 coefficient differs); move data within
                        // the quad via shfl (verified by micro/e2e tests).
                        // The float fragment is consumed by the mma atom via
                        // bitcast (tf32 truncation happens in HW).
                        return flash::convert_acc_to_tf32_Aregs_f32<TiledMma>(tSrS);
                    } else {
                        Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
                        Tensor tOrP_ = make_tensor_like<Element>(tOrP_acc);
                        convert_type_out(tOrP_acc, tOrP_);
                        return tOrP_;
                    }
                } else {
                    Tensor tOrP_acc = flash::convert_acc<Element>(tSrS);
                    return make_tensor(tOrP_acc.data(), make_layout(get<0>(tSrQ.layout()), get<1>(tSrS.layout()), get<2>(tSrS.layout())));
                }
            }();
#else
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);
#endif
            if constexpr (kHeadDim <= 96) {
                // Deferred V wait: V[n_block] must have arrived; K[next]
                // (most recent group) may still be in flight.
                flash::cp_async_wait<1>();
                __syncthreads();
            }
            flash::gemm_rs_sm80(tOrO, tOrP, tOrV, tOsVt(_, _, _, 0),
                                tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

            any_tile_processed = true;
            block_meta.advance_inner();
            if (dbg) { ++dbg_steps; }
        };

        // Prologue: prefetch K of the first block of this slice.
        load_K_block(block_meta.inner_block_idx);
        cute::cp_async_fence();

        auto diag_mask_fn = [&](auto& tSrS_, int n_block) {
            mask.template apply</*Seqlenk_mask=*/false>(
                tSrS_, m_block, n_block,
                block_meta.cur_mask_type,
                block_meta.cur_k_start, block_meta.cur_k_end,
                block_meta.cur_diagonal_offset, block_meta.cur_band_width,
                block_meta.cur_q_start, block_meta.cur_q_end,
                thread_idx, seqlen_q, seqlen_k,
                sl.mask_bits, sl.mask_row_stride);
        };
        auto no_mask_fn = [&](auto& /*tSrS_*/, int /*n_block*/) {};
        auto needs_mask = [&](int n_block) {
            return !block_meta.is_no_mask_block(n_block);
        };

        // Zone runs: the no-mask zone is ONE contiguous interval in n_block
        // space, so iterating in either direction visits at most three runs
        // (masked / no-mask / masked, e.g. BICAUSAL).  Each run's inner loop
        // is specialized: masked runs keep Check_inf=true and apply the
        // algebraic mask; the no-mask run skips mask.apply() entirely and
        // runs the hot iterations with Check_inf=false (compile-time).
        // NOTE: a no-mask block's scores are always finite (all cols visible,
        // and is_no_mask_block()==false whenever q_partial), so entering the
        // no-mask run with row_max==-inf is safe without Check_inf.
        if (!block_meta.is_inner_finish()) {
            // Hoisted first step keeps the runtime is_first branch out of the
            // hot run loops below.  is_first_iter persists across vbatch
            // iterations (RangeMerge LSE merging), so a later covering slice
            // starts with Is_first=false (softmax state already live).
            bool const first_needs_mask = needs_mask(block_meta.inner_block_idx);
            if (dbg && first_needs_mask) { ++dbg_masked; }
            if (is_first_iter) {
                if (first_needs_mask) {
                    step(diag_mask_fn, cute::true_type{}, cute::true_type{});
                } else {
                    step(no_mask_fn, cute::true_type{}, cute::true_type{});
                }
            } else {
                if (first_needs_mask) {
                    step(diag_mask_fn, cute::false_type{}, cute::true_type{});
                } else {
                    step(no_mask_fn, cute::false_type{}, cute::true_type{});
                }
            }
        }
        while (!block_meta.is_inner_finish()) {
            if (needs_mask(block_meta.inner_block_idx)) {
                #pragma unroll 1
                do {
                    step(diag_mask_fn, cute::false_type{}, cute::true_type{});
                    if (dbg) { ++dbg_masked; }
                } while (!block_meta.is_inner_finish() && needs_mask(block_meta.inner_block_idx));
            } else {
                #pragma unroll 1
                do {
                    step(no_mask_fn, cute::false_type{}, cute::false_type{});
                } while (!block_meta.is_inner_finish() && !needs_mask(block_meta.inner_block_idx));
            }
        }
        };  // end run_slice

        // Trivial single-slice specialization (host-proven dense / causal,
        // see AttnSliceParams::trivial_mask): the whole mask is ONE FULL /
        // CAUSAL slice over the rectangle, so the mainloop runs with pure
        // host-constant metadata — no descriptor / row-table global loads
        // per work tile and no vbatch bookkeeping.
        if (sl.trivial_mask != 0) {
            TrivialSliceMeta<kDir> tmeta(
                sl.trivial_mask, sl.trivial_diagonal,
                seqlen_q, seqlen_k, kBlockM, kBlockN, m_block);
            if (tmeta.has_work()) {
                run_slice(tmeta);
            }
        } else {
        // Resolve the vbatch range covering this m_block.  Coverage is
        // constant within an m_block (kBlockM-aligned slice Q boundaries).
        int vb_lo, vb_hi;
        if (sl.vbatch_to_slice != nullptr) {
            vb_lo = sl.row_to_vbatch_start[row0];
            vb_hi = sl.row_to_vbatch_end[row0];
        } else {
            // Legacy single-slice dispatch: synthesize one iteration.
            vb_lo = -1;
            vb_hi = 0;
        }
        // Outer vbatch loop: one iteration per covering slice.
        for (int vb = vb_lo; vb < vb_hi; ++vb) {
        int const slice_idx = ((vb < 0)
            ? sl.row_to_slice[row0]   // legacy path
            : sl.vbatch_to_slice[vb]) // RangeMerge path
            - slice_idx_base;         // flat index → group-local index

        SliceBlockMetaFlexFlash<kDir> block_meta(
            sl, seqlen_q, seqlen_k, kBlockM, kBlockN, slice_idx, m_block);

        // Empty K range or invalid slice → try the next covering slice.
        if (block_meta.skip_to_first_valid()) {
            continue;
        }

        run_slice(block_meta);
        }  // end vbatch loop (RangeMerge)
        }  // end generic slice dispatch

        // Finalize softmax (warp-reduce sum + compute LSE into row_sum).
        auto scores_scale = softmax.finalize();
        softmax.rescale_o(tOrO, scores_scale);

        if (dbg && thread_idx == 0) {
            atomicAdd(dbg + 0, (unsigned long long)dbg_steps);
            atomicAdd(dbg + 1, (unsigned long long)dbg_masked);
            atomicAdd(dbg + 4, clock64() - dbg_c0);
        }

        return any_tile_processed;
    }
};

} // namespace flash
