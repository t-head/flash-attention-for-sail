/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

// Return {kBlockM, kBlockN, Mma1_is_RS, IntraWGOverlap}
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(
        int headdim, bool is_causal, bool is_local, int element_size=2,
        bool v_colmajor=false, bool paged_kv=false, bool softcap=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {192, 128, true, true};
            // Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen
            // return {192, is_causal || is_local ? 192 : 176, true, false};
        } else if (headdim <= 96) {
            return {192, is_local || paged_kv ? 128 : 144, false, true};
        } else if (headdim <= 128) {
            return {128, is_causal || is_local || paged_kv ? 128 : 176, true, true};
            // {128, 192, false, false} and {192, 128, false, true} are quite good too
            // 128 x 192 hits the limit of smem if Mma1_is_RS, 128 x 144 hits the limit if !Mma1_is_RS
        } else if (headdim <= 192) {
            return {128, paged_kv || is_local ? 96 : 112, true, true};  // 128 x 112 hits the limit of smem
        } else {
            return {128, is_local ? 64 : 80, true, true};  // 128 x 80 hits the limit of smem
        }
    } else {
        if (headdim <= 64) {
            return {192, 160, true, true};
        } else if (headdim <= 96) {
            return {192, 128, true, true};
        } else if (headdim <= 128) {
            return {128, paged_kv ? 160 : (v_colmajor || (softcap && is_local) ? 192 : 224), true, true};
        } else if (headdim <= 192) {
            return {128, (paged_kv || softcap) && is_local ? 128 : 160, true, true};
        } else {
            return {128, is_local ? 64 : 128, true, !paged_kv};  // PagedKV uses more registers so we disabled IntraWGOverlap
        }
    }
}

// Return {kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
        bool sm86_or_89, int headdim, bool is_causal, bool is_local, int element_size=2,
        bool paged_kv=false, bool varlen_and_split=false,
        bool softcap=false, bool append_kv=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {128, varlen_and_split ? 80 : (is_local ? 96 : 112), 4, 1, false};
        } else if (headdim <= 96) {
            return {128, varlen_and_split || is_local ? 48 : 64, 4, 1, false};
        } else if (headdim <= 128) {
            bool const use_8_warps = sm86_or_89 | varlen_and_split;
            return {128, use_8_warps ? (varlen_and_split ? (is_local ? 96 : 112) : (is_local ? 96 : 128)) : (is_local ? 48 : 64), use_8_warps ? 8 : 4, 1, use_8_warps};
        } else if (headdim <= 192) {
            bool const kBlockN_64 = append_kv || is_local || varlen_and_split || paged_kv;
            return {128, kBlockN_64 ? 64 : 96, 8, sm86_or_89 ? 1 : 2, !kBlockN_64};
        } else {
            return {128, sm86_or_89 ? (append_kv ? 32 : (varlen_and_split || is_local ? 48 : 64)) : (append_kv ? 48 : (varlen_and_split || is_local ? 64 : 96)), 8, 1, sm86_or_89 && !append_kv};
        }
    } else {
        // Placeholder for now
        return {128, 64, 8, 2, false};
    }
}

#ifdef USE_PPU
// Return {kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_ppu(
        int arch, int headdim, bool is_causal, bool is_local, int element_size=2,
        bool paged_kv=false, bool varlen=false, bool split=false,
        bool softcap=false, bool append_kv=false, bool pack_gqa = false,
        bool kBlockM_128=false, bool kBlockM_16=false) {
    if (element_size == 2) {
        bool const vreg_strain = paged_kv || pack_gqa || varlen || is_local;
        if (kBlockM_16) {
            if (headdim <= 64) {
                return {16, split ? 16 : 128, 1, 1, true};
            } else if (headdim <= 96) {
                return {16, split ? 16 : 96, 1, 1, true};
            } else if (headdim <= 128) {
                return {16, split ? 16 : 64, 1, 1, true};
            } else {
                return {16, 16, 1, 1, true};
            }
        }
        if (headdim <= 64) {
            if ((is_causal && !varlen) || !kBlockM_128) {
                return {64, 128, 4, 1, vreg_strain ? false : true};
            } else {
                return {128, vreg_strain ? 64 : 96, 4, 1, vreg_strain ? false : true};
            }
        } else if (headdim <= 96) {
            if ((is_causal && !varlen) || !kBlockM_128) {
                return {64, vreg_strain ? 96 : 128, 4, 1, vreg_strain ? false : true};
            } else {
                return {128, vreg_strain ? 32 : 64, 4, 1, vreg_strain ? false : true};
            }
        } else if (headdim <= 128) {
            if ((is_causal && !varlen) || !kBlockM_128) {
                return {64, is_causal || vreg_strain ? 64 : 96, 4, 1, vreg_strain ? false : true};
            } else {
                return {128, 64, 4, 1, false};
            }
        } else if (headdim <= 192) {
            return {64, vreg_strain ? 32 : 64, 4, 1, vreg_strain ? false : true};
        } else {
            return {64, vreg_strain ? 16 : 32, 4, 1, vreg_strain ? false : true};
        }
    } else {
        // Placeholder for now
        return {128, 64, 8, 2, false};
    }
}
#endif