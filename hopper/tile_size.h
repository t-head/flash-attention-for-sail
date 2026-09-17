/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

// Return {kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap}
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(
        int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
        bool v_colmajor=false, bool paged_kv_non_TMA=false, bool softcap=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            // return {same_hdim ? 192 : 64, same_hdim ? 128 : 64, same_hdim, same_hdim};
            // With this workaround in Cutlass 3.8, tile size 192 x 128 got slower for non-causal, idk why
            // https://github.com/NVIDIA/cutlass/blob/833f6990e031b48b4cd2fcf55e0849c51ef6bac2/include/cute/container/tuple.hpp#L131
            if (headdim_v == 512) {
                return {64, 64, false, false};
            } else if (headdim_v == 256) {
                return {128, 96, true, false};
            } else {
                // Switch to tile size 192 x 192 for now
                bool const use_blockN_128 = is_causal || is_local;
                return {192, use_blockN_128 ? 128 : 192, use_blockN_128, true};
            }
            // Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen
            // return {192, is_causal || is_local ? 192 : 176, true, false};
        } else if (headdim <= 96) {
            return {192, is_local || paged_kv_non_TMA ? 128 : 144, false, true};
        } else if (headdim <= 128) {
            return {128, is_causal || is_local || paged_kv_non_TMA ? 128 : 176, true, true};
            // {128, 192, false, false} and {192, 128, false, true} are quite good too
            // 128 x 192 hits the limit of smem if MmaPV_is_RS, 128 x 144 hits the limit if !MmaPV_is_RS
        } else if (headdim <= 192) {
            return {128, paged_kv_non_TMA || is_local ? 96 : (headdim_v <= 128 ? 128 : 112), true, true};  // 128 x 112 hits the limit of smem
        } else {
            return {128, is_local ? 64 : 80, true, true};  // 128 x 80 hits the limit of smem
        }
    } else {
        if (headdim <= 64) {
            return {192, 160, true, true};
        } else if (headdim <= 96) {
            return {192, 128, true, true};
        } else if (headdim <= 128) {
            return {128, paged_kv_non_TMA ? 160 : (v_colmajor || (softcap && is_local) ? 192 : 224), true, true};
        } else if (headdim <= 192) {
            return {128, (paged_kv_non_TMA || softcap) && is_local ? 128 : 160, true, true};
        } else {
            return {128, is_local ? 64 : 128, true, !paged_kv_non_TMA};  // PagedKV uses more registers so we disabled IntraWGOverlap
        }
    }
}

// Return {kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
        bool sm86_or_89, int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
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
        int arch, int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
        bool paged_kv=false, bool varlen=false, bool split=false,
        bool softcap=false, bool append_kv=false, bool pack_gqa = false,
        bool kBlockM128=false, bool kBlockM16=false, bool kBlockN16=false,
        bool PagedKVAiu=false) {
    if (element_size == 2) {
        bool const vreg_strain = paged_kv || pack_gqa || varlen || is_local;
        if (kBlockM16) {
            if (headdim <= 64) {
                return {16, kBlockN16 ? 16 : (headdim_v <= 64 ? 128 : (headdim_v <= 256 ? 64 : 16)), 1, 1, true};
            } else if (headdim <= 192) {
                return {16, kBlockN16 ? 16 : 64, 1, 1, true};
            } else {
                return {16, 16, 1, 1, true};
            }
        }
        if (headdim <= 64) {
            if (headdim_v == 512) {
                return {64, 16, 4, 1, false};
            } else if (headdim_v == 256) {
                return {64, vreg_strain ? 16 : 32, 4, 1, vreg_strain ? false : true};
            } else {
                if (arch == 80 && !paged_kv && !varlen) {
                    return {64, vreg_strain ? 64 : 96, 2, 1, vreg_strain ? false : true};
                }
                if (!kBlockM128) {
                    return {64, 128, 4, 1, vreg_strain ? false : true};
                } else {
                    return {128, vreg_strain ? 64 : 96, 4, 1, vreg_strain ? false : true};
                }
            }
        } else if (headdim <= 96) {
            if (!kBlockM128) {
                return {64, vreg_strain ? 96 : 128, 4, 1, vreg_strain ? false : true};
            } else {
                return {128, vreg_strain ? 32 : 64, 4, 1, vreg_strain ? false : true};
            }
        } else if (headdim <= 128) {
            if (!kBlockM128) {
                return {64, is_causal || vreg_strain ? 64 : 96, 4, 1, vreg_strain ? false : true};
            } else {
                return {128, 64, 4, 1, false};
            }
        } else if (headdim <= 192) {
            if (!kBlockM128 || arch != 89) {
                return {64, 64, 4, 1, true};
            } else {
                return {128, 64, 8, 1, true};
            }
        } else {
            // hdim 256: the kBlockM=128 arm mirrors the FA2 A100/PPU config
            // (128x64, 8 warps, Q in smem) — halves the KV reread per Q row
            // versus the default 64-wide tile.  Gated by the launch-template
            // heuristic (FLASH_ATTENTION_ARB_FORCE_KBM64 falls back below).
            if (kBlockM128) {
                return {128, 64, 8, 1, false};
            }
            return {64, arch == 89 ? 48 : 32, 4, 1, true};
        }
    } else {
        if (kBlockM16) {
            if (headdim <= 128) {
                if (paged_kv && (!PagedKVAiu)) {
                    return {16, 32, 1, 2, true};
                } else {
                    return {64, 64, 4, 2, true};
                }
            } else if (headdim <= 192) {
                if (paged_kv && (!PagedKVAiu)) {
                    return {16, 32, 1, 2, true};
                } else {
                    return {64, 32, 4, 2, true};
                }
            } else {
                if (paged_kv && (!PagedKVAiu)) {
                    return {32, 32, 2, 2, true};  // head 256, use 2 warps to faster tranpose
                } else {
                    return {64, 32, 4, 2, true};
                }
            }
        }
        if (headdim <= 128) {
            return {64, 64, 4, 2, true};
        } else if (headdim <= 192) {
            return {64, 32, 4, 2, true};
        } else {
            // use more threads to faster transpose
            return {128, 32, 8, 2, true};
        }
    }
}
#endif