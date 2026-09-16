/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#include "cutlass/fast_math.h"
#include "cutlass/barrier.h"
#include "cutlass/arch/barrier.h"

#ifndef FLASHATTENTION_DISABLE_SM90
#include "cutlass/arch/grid_dependency_control.h"
#endif

#include "flash.h"
#ifdef USE_PPU
#include "heuristics.h"
#endif

namespace flash {

__global__ void prepare_varlen_num_blocks_kernel(
        int seqlen_q_static, int seqlen_k_static, int seqlen_k_new_static,
        int const* const cu_seqlens_q, int const* const cu_seqlens_k, int const* const cu_seqlens_k_new,
        int const* const seqused_q, int const* const seqused_k, int const* const leftpad_k_ptr,
#ifdef USE_PPU
        int num_batch, int num_head, int qhead_per_khead, int num_sm, int num_splits_static, int num_splits_average,
#else
        int num_batch, int num_head, int qhead_per_khead, int num_sm, int num_splits_static,
#endif
        cutlass::FastDivmod blockm_divmod, cutlass::FastDivmod blockn_divmod,
        int* const tile_count_semaphore,
        // int* const num_m_blocks_ptr,
        int* const num_splits_dynamic_ptr,
        bool enable_pdl) {

    static constexpr int kNumBatchPerWarp = cutlass::NumThreadsPerWarp - 1;
    static constexpr int kSmemSize = 1;
    // Assume that there's only one block in the grid
    __shared__ int total_blocks_smem[kSmemSize];
#ifdef USE_PPU
    __shared__ int max_num_splits_dynamic_smem;
    __shared__ int total_blocks_dynamic_smem;
    __shared__ int total_blocks_average_smem;
#endif

#ifndef FLASHATTENTION_DISABLE_SM90
    // There's only 1 block in the grid, so might as well start launching the main attn kernel
    if (enable_pdl) { cutlass::arch::launch_dependent_grids(); }
#endif

    if (threadIdx.x < kSmemSize) { total_blocks_smem[threadIdx.x] = 0; }
#ifdef USE_PPU
    if (threadIdx.x == 0) {
        max_num_splits_dynamic_smem = 0;
        total_blocks_dynamic_smem = 0;
        total_blocks_average_smem = 0;
    }
#endif
    __syncthreads();

    if (threadIdx.x == 0 && tile_count_semaphore) { *tile_count_semaphore = 0; }

    int lane = threadIdx.x % cutlass::NumThreadsPerWarp;

#ifdef USE_PPU
    auto get_num_m_blocks = [&](int bidb_start, int& seqlen_q_dynamic) {
#else
    auto get_num_m_blocks = [&](int bidb_start) {
#endif
        int batch_idx = lane + bidb_start;
        int seqlen;
        if (seqused_q) {
#ifdef USE_PPU
            seqlen = batch_idx < num_batch ? __ldg(&seqused_q[batch_idx]) : 0;
#else
            seqlen = batch_idx < num_batch ? seqused_q[batch_idx] : 0;
#endif
        } else if (cu_seqlens_q) {
#ifdef USE_PPU
            int cur_cu_seqlen = batch_idx <= num_batch ? __ldg(&cu_seqlens_q[batch_idx]) : 0;
#else
            int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_q[batch_idx] : 0;
#endif
            int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
            seqlen = next_cu_seqlen - cur_cu_seqlen;
        } else {
            seqlen = seqlen_q_static;
        }
#ifdef USE_PPU
        seqlen_q_dynamic = seqlen;
#endif
        seqlen *= qhead_per_khead;
        return batch_idx < num_batch && lane < kNumBatchPerWarp
            ? blockm_divmod.div(seqlen + blockm_divmod.divisor - 1) : 0;
    };

    auto get_num_n_blocks = [&](int bidb_start) {
        int batch_idx = lane + bidb_start;
#ifdef USE_PPU
        int leftpad_k = batch_idx < num_batch && leftpad_k_ptr != nullptr ? __ldg(&leftpad_k_ptr[batch_idx]) : 0;
#else
        int leftpad_k = batch_idx < num_batch && leftpad_k_ptr != nullptr ? leftpad_k_ptr[batch_idx] : 0;
#endif
        int seqlen;
        if (seqused_k) {
#ifdef USE_PPU
            seqlen = batch_idx < num_batch ? __ldg(&seqused_k[batch_idx]) : 0;
#else
            seqlen = batch_idx < num_batch ? seqused_k[batch_idx] : 0;
#endif
        } else if (cu_seqlens_k) {
#ifdef USE_PPU
            int cur_cu_seqlen = batch_idx <= num_batch ? __ldg(&cu_seqlens_k[batch_idx]) : 0;
#else
            int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_k[batch_idx] : 0;
#endif
            int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
            seqlen = next_cu_seqlen - cur_cu_seqlen;
        } else {
            seqlen = seqlen_k_static;
        }
        int seqlen_new;
        if (cu_seqlens_k_new) {
#ifdef USE_PPU
            int cur_cu_seqlen_new = batch_idx <= num_batch ? __ldg(&cu_seqlens_k_new[batch_idx]) : 0;
#else
            int cur_cu_seqlen_new = batch_idx <= num_batch ? cu_seqlens_k_new[batch_idx] : 0;
#endif
            int next_cu_seqlen_new = __shfl_down_sync(0xffffffff, cur_cu_seqlen_new, 1);
            seqlen_new = next_cu_seqlen_new - cur_cu_seqlen_new;
        } else {
            seqlen_new = seqlen_k_new_static;
        }
        // if (threadIdx.x == 0) { printf("seqlen = %d, seqlen_new = %d, leftpad_k = %d\n", seqlen, seqlen_new, leftpad_k); }
        seqlen = seqlen - leftpad_k + seqlen_new;
        return batch_idx < num_batch && lane < kNumBatchPerWarp
            ? blockn_divmod.div(seqlen + blockn_divmod.divisor - 1) : 0;
    };

    int warp_idx = threadIdx.x / cutlass::NumThreadsPerWarp;
    int bidb_start = kNumBatchPerWarp * warp_idx;
#ifdef USE_PPU
    int seqlen_q_dynamic;
    int num_m_blocks = get_num_m_blocks(bidb_start, seqlen_q_dynamic);
    // We do not split for prefill, to get rid of too much costs of dynamic split combine kernel.
    int num_n_blocks = get_num_n_blocks(bidb_start);
    num_n_blocks = num_n_blocks < 1 ? 0 : (seqlen_q_dynamic <= 16 ? num_n_blocks : 1);
#else
    int num_m_blocks = get_num_m_blocks(bidb_start);
    int num_n_blocks = get_num_n_blocks(bidb_start);
#endif

    int total_blocks = num_m_blocks * num_n_blocks;
    // Warp sum
    #pragma unroll
    for (int i = cutlass::NumThreadsPerWarp / 2; i >= 1; i /= 2) {
        total_blocks += __shfl_down_sync(0xffffffff, total_blocks, i);
    }
    if (lane == 0) { atomicAdd(total_blocks_smem, total_blocks); }
    __syncthreads();
    total_blocks = total_blocks_smem[0];
#ifdef USE_PPU
    int blocks_per_sm = static_cast<int>(ceilf(float(total_blocks) * 1.2f * float(num_head) / float(num_sm)));
#else
    // 10% margin
    int blocks_per_sm = static_cast<int>(ceilf(float(total_blocks) * 1.1f * float(num_head) / float(num_sm)));
#endif
    // blocks_per_sm = std::max(1, blocks_per_sm);  // 1 is the minimum number of blocks per SM
    int num_splits_dynamic = std::max(std::min((num_n_blocks + blocks_per_sm - 1) / blocks_per_sm, num_splits_static), 1);
    if (bidb_start + lane < num_batch && lane < kNumBatchPerWarp) {
        num_splits_dynamic_ptr[bidb_start + lane] = num_splits_dynamic;
        // printf("idx = %d, num_m_blocks = %d, num_n_blocks = %d, num_split_static = %d, num_splits_dynamic = %d\n", bidb_start + lane, num_m_blocks_ptr[bidb_start + lane], num_n_blocks, num_splits_static, num_splits_dynamic);
    }
#ifdef USE_PPU
    int max_num_splits_dynamic = num_splits_dynamic;
    int total_blocks_dynamic = num_m_blocks * num_splits_dynamic;
    int total_blocks_average = num_m_blocks * num_splits_average;
    if (bidb_start < num_batch) {
        #pragma unroll
        for (int i = cutlass::NumThreadsPerWarp / 2; i >= 1; i /= 2) {
            int tmp_num_splits_dynamic = __shfl_down_sync(0xffffffff, max_num_splits_dynamic, i);
            max_num_splits_dynamic = max_num_splits_dynamic < tmp_num_splits_dynamic ? tmp_num_splits_dynamic : max_num_splits_dynamic;
            total_blocks_dynamic += __shfl_down_sync(0xffffffff, total_blocks_dynamic, i);
            total_blocks_average += __shfl_down_sync(0xffffffff, total_blocks_average, i);
        }
        if (lane == 0) {
            atomicMax(&max_num_splits_dynamic_smem, max_num_splits_dynamic);
            atomicAdd(&total_blocks_dynamic_smem, total_blocks_dynamic);
            atomicAdd(&total_blocks_average_smem, total_blocks_average);
        }
    }
    __syncthreads();
    max_num_splits_dynamic = max_num_splits_dynamic_smem;
    total_blocks_dynamic = total_blocks_dynamic_smem;
    total_blocks_average = total_blocks_average_smem;
    if ((max_num_splits_dynamic < num_splits_average ||
        // "+ 3" is empirical that when max_num_splits_dynamic >= num_splits_average + 3, there's no need to check the following condition
        (max_num_splits_dynamic < num_splits_average + 3 && total_blocks_dynamic * num_head > num_sm && total_blocks_average < total_blocks_dynamic)) &&
        num_splits_average <= num_splits_static) {
        if (bidb_start + lane < num_batch && lane < kNumBatchPerWarp) {
            num_splits_dynamic_ptr[bidb_start + lane] = seqlen_q_dynamic <= 16 ? num_splits_average : 1;
        }
        max_num_splits_dynamic = num_splits_average;
    }
    if (threadIdx.x == 0) {
        num_splits_dynamic_ptr[num_batch] = max_num_splits_dynamic;
    }
#endif
}

} // flash

void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream, bool packgqa,
                               int blockM, int blockN, bool enable_pdl) {
    // Only support batch <= 992 (32 warps, each with 31 batches)
    int qhead_per_khead = !packgqa ? 1 : cutlass::ceil_div(params.h, params.h_k);
#ifdef USE_PPU
    int const occ = (blockN == 16) ? (params.d_rounded <= 64 && params.dv_rounded > 256 ? 14 : 16) :
        (params.d_rounded <= 64 ? (params.dv_rounded <= 64 ? 8 : (params.dv_rounded <= 256 ? 6 : 14)) :
        (params.d_rounded <= 96 ? 10 : (params.d_rounded <= 128 ? 8 : (params.d_rounded <= 192 ? 5 : 16))));
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + blockM));
    int const num_n_blocks = (seqlen_k_loaded + blockN - 1) / blockN;
    int const num_m_blocks = (seqlen_q_packgqa + blockM - 1) / blockM;
    int total_mblocks = params.b * params.h_k * ((params.total_q / params.b * (params.h / params.h_k) + blockM - 1) / blockM);
    int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
    int num_splits_average = num_splits_heuristic(total_mblocks, params.num_sm, occ, num_n_blocks, num_m_blocks, size_one_kv_head, params.is_causal || params.is_local, 128);
#endif
    flash::prepare_varlen_num_blocks_kernel<<<1 /*grid*/, 1024 /*block*/, 0, stream>>>(
        params.seqlen_q, params.seqlen_k, params.seqlen_knew,
        params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
        params.seqused_q, params.seqused_k, params.leftpad_k,
#ifdef USE_PPU
        params.b, !packgqa ? params.h : params.h_k, qhead_per_khead, params.num_sm * occ, params.num_splits, num_splits_average,
#else
        params.b, !packgqa ? params.h : params.h_k, qhead_per_khead, params.num_sm, params.num_splits,
#endif
        cutlass::FastDivmod(blockM), cutlass::FastDivmod(blockN),
        params.tile_count_semaphore,
        // params.num_m_blocks_ptr,
        params.num_splits_dynamic_ptr, enable_pdl);
}
