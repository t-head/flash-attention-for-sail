/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Dynamic persistent tile scheduler for the flex flash attention fwd/bwd kernels.
//
// Methodology reference: DynamicPersistentTileSchedulerSM80 (tile_scheduler.hpp).
// A fixed pool of CTAs (= number of SMs) grabs tiles dynamically through a
// global semaphore counter, instead of the static round-robin used by
// StaticPersistentTileScheduler.  This balances the uneven per-tile workload
// caused by diagonal masks (causal / bicausal tiles have very different cost).
//
// The linear tile index decomposes as (block, head, batch):
//   fwd: block = m_block,  bwd: block = n_block
// No varlen / split logic: the flex flash attention path is fixed-shape only.
//
// Interface is identical to SingleTileScheduler / StaticPersistentTileScheduler
// (SharedStorage, Params, WorkTileInfo::is_valid/get_block_coord,
// get_initial_work/get_next_work/init_consumer/prefetch_next_work), so the
// kernels only need a different `using Scheduler = ...` in the launch template.

#pragma once

#include "cutlass/fast_math.h"

#include "../hopper/tile_scheduler.hpp"

namespace flash {

template<int kBlock = 128>
class FlexFlashTileSchedulerSM80 {

public:

    using SharedStorage = int;

protected:
    SharedStorage* const tile_count_smem;

public:

    // Device side kernel params
    struct Params {
        int const total_tiles;
        int const num_head, num_batch;
        cutlass::FastDivmod m_block_divmod, head_divmod;
        int* const tile_count_semaphore;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.tile_count_semaphore != nullptr);
        return {args.num_blocks * args.num_head * args.num_batch,
                args.num_head, args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(std::min(params.total_tiles, num_sm))};
    }

    struct WorkTileInfo {
        int tile_idx_next;
        int tile_idx, block_idx, bidh, bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) {
            if (tile_idx_next >= params.total_tiles) { return false; }
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block_idx, tile_idx_next));
            tile_idx = tile_idx_next;
            return true;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            // No block_idx reversal here: the inner-loop direction is already
            // controlled by the kernel's DispatchDirection (kDir) template.
            return {block_idx, bidh, bidb, 0 /*split_idx*/};
        }
    };

    CUTLASS_DEVICE
    FlexFlashTileSchedulerSM80(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {};

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x), 0, 0, 0, 0};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        int const warp_idx = cutlass::canonical_warp_idx_sync();
        if (warp_idx == 0) {
            // Only lane 0 claims the next tile: the semaphore is a single
            // int (host-side torch::zeros({1})), so a per-lane atomicAdd on
            // [lane_id] would write past the allocation (and only lane 0's
            // return value is consumed anyway).
            if (__ppu_read_laneid() == 0) {
                *tile_count_smem = atomicAdd(params.tile_count_semaphore, 1);
            }
        }
        __syncthreads();
        int tile_idx = __ppu_read_firstlane(*tile_count_smem) + int(gridDim.x);
        return {tile_idx, current_work.tile_idx, current_work.block_idx, current_work.bidh, current_work.bidb};
    }

};

} // namespace flash
