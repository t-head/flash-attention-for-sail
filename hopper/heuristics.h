/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <vector>

#ifdef USE_PPU
inline bool should_pack_gqa(bool varlen_q, int seqlen_q, int num_qhead, int num_khead, int blockM_nopack, int blockM_pack, int num_SMs) {
    // If varlen, we don't actually know seqlen_q but only max_seqlen_q.
    if (varlen_q) return true;
    if (num_qhead <= num_khead) return false;
    // Heuristic: PackGQA is a bit slower but can help if seqlen_q is small or not near a multiple of kBlockM
    auto round_up = [](int a, int b) { return (a + b - 1) / b * b; };
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    int qhead_per_khead = num_qhead / num_khead;
    float nopack_gqa_efficiency = float(seqlen_q) / float(round_up(seqlen_q, blockM_nopack));
    float pack_gqa_efficiency = float(seqlen_q * qhead_per_khead) / float(round_up(seqlen_q * qhead_per_khead, blockM_pack));
    float nopack_gqa_n_waves = float(ceil_div(seqlen_q, blockM_nopack) * num_qhead) / num_SMs;
    float nopack_gqa_wave_eff = nopack_gqa_n_waves / ceil(nopack_gqa_n_waves);
    float pack_gqa_n_waves = float(ceil_div(seqlen_q * qhead_per_khead, blockM_pack) * num_khead) / num_SMs;
    float pack_gqa_wave_eff = pack_gqa_n_waves / ceil(pack_gqa_n_waves);
    // when qhead_per_khead < 4 (common occupancy in prefill instance), packgqa=false will break the load balance and L2 cache locality.
    return nopack_gqa_efficiency < 0.9 * pack_gqa_efficiency || nopack_gqa_wave_eff < 0.95 * pack_gqa_wave_eff || qhead_per_khead < 4;
};
#else
inline bool should_pack_gqa(bool varlen_q, int seqlen_q, int qhead_per_khead, int blockM) {
    // If varlen, we don't actually know seqlen_q but only max_seqlen_q.
    if (varlen_q) return true;
    // Heuristic: PackGQA is a bit slower but can help if seqlen_q is small or not near a multiple of kBlockM
    auto round_up = [](int a, int b) { return (a + b - 1) / b * b; };
    float nopack_gqa_efficiency = float(seqlen_q) / float(round_up(seqlen_q, blockM));
    float pack_gqa_efficiency = float(seqlen_q * qhead_per_khead) / float(round_up(seqlen_q * qhead_per_khead, blockM));
    return nopack_gqa_efficiency < 0.9 * pack_gqa_efficiency;
};
#endif

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
#ifdef USE_PPU
inline int num_splits_heuristic(int total_mblocks, int num_CUs, int occ, int num_n_blocks, int num_m_blocks, int size_one_kv_head, bool is_causal_or_local, int max_splits) {
    // printf("total_mblocks = %d, num_n_blocks = %d\n", total_mblocks, num_n_blocks);
    int num_SMs = num_CUs * occ;
    bool has_enough_mblocks = false;
    if (occ > 8) {
        has_enough_mblocks = (total_mblocks >= 0.8f * num_SMs);
    } else {
        float n_waves = float(total_mblocks) / num_SMs;
        float eff = n_waves / ceil(n_waves);
        has_enough_mblocks = (eff > 0.9f);
    }
    if (has_enough_mblocks) {
#else
inline int num_splits_heuristic(int total_mblocks, int num_SMs, int num_n_blocks, int num_m_blocks, int size_one_kv_head, bool is_causal_or_local, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    // However, in the case of super long seqlen where each head of KV doesn't even fit into
    // L2 (we assume that L2 size is 50MB), we want to split.
    if (total_mblocks >= 0.8f * num_SMs) {
#endif
        int const size_l2 = 50 * 1024 * 1024;
        // Only split if there are enough queries to go over the KV at least twice
        // Don't split if causal
        if (size_one_kv_head > size_l2 && num_m_blocks >= num_SMs * 2 && !is_causal_or_local) {
            return std::min((size_one_kv_head + size_l2 - 1) / size_l2, max_splits);
        } else {
            return 1;
        }
    }
    // If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if (num_n_blocks <= 4) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        float n_waves = float(total_mblocks * num_splits) / num_SMs;
        float eff = n_waves / ceil(n_waves);
        // printf("num_splits = %d, eff = %f\n", num_splits, eff);
        if (eff > max_efficiency) { max_efficiency = eff; }
        efficiency.push_back(eff);
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
#ifdef USE_PPU
        if (efficiency[num_splits - 1] >= 0.90 * max_efficiency) {
            // minimize num_splits to reduce the overhead of combine kernel
            int num_n_blocks_per_split = (num_n_blocks + num_splits - 1) / num_splits;
            int num_splits_worktile_balanced = (num_n_blocks + num_n_blocks_per_split - 1) / num_n_blocks_per_split;
            unsigned long long next_power = 1ULL << (64 - __builtin_clzll(num_splits_worktile_balanced - 1));
            if (next_power <= 128 && next_power < num_splits && efficiency[num_splits_worktile_balanced - 1] / efficiency[num_splits - 1] >= 0.8) {
                num_splits = num_splits_worktile_balanced;
            }
#else
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
#endif
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}
