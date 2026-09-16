/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

// We consolidate all the info related to sequence length here. This is so that we can do all
// the gmem reads once at the beginning of each tile, rather than having to repeat these reads
// to compute various things like n_block_min, n_block_max, etc.

template <bool Varlen, int kBlock>
struct SeqlenInfo {

    int const offset, offset_padded;
    int const seqlen;

    CUTLASS_DEVICE
    SeqlenInfo(int const bidb, int const seqlen_static, int const* const cu_seqlens, int const* const seqused)
#ifdef USE_PPU
        : offset(!Varlen || cu_seqlens == nullptr ? 0 : __ldg(&cu_seqlens[bidb]))
        , offset_padded(!Varlen || cu_seqlens == nullptr ? 0 : (__ldg(&cu_seqlens[bidb]) + bidb * kBlock) / kBlock * kBlock)
#else
        : offset(!Varlen || cu_seqlens == nullptr ? 0 : cu_seqlens[bidb])
        , offset_padded(!Varlen || cu_seqlens == nullptr ? 0 : (cu_seqlens[bidb] + bidb * kBlock) / kBlock * kBlock)
#endif
        , seqlen(!Varlen
                 ? seqlen_static
#ifdef USE_PPU
                 : (seqused ? __ldg(&seqused[bidb]) : (cu_seqlens ? __ldg(&cu_seqlens[bidb + 1]) - __ldg(&cu_seqlens[bidb]) : seqlen_static)))
#else
                 : (seqused ? seqused[bidb] : (cu_seqlens ? cu_seqlens[bidb + 1] - cu_seqlens[bidb] : seqlen_static)))
#endif
    {
    }

};

template <bool Varlen, int kBlockM>
struct SeqlenInfoQK {

    int const offset_q, offset_k, offset_q_padded;
    int const seqlen_q, seqlen_k;

    CUTLASS_DEVICE
    SeqlenInfoQK(int const bidb, int const seqlen_q_static, int const seqlen_k_static,
                 int const* const cu_seqlens_q, int const* const cu_seqlens_k,
                 int const* const seqused_q, int const* const seqused_k
                 )
#ifdef USE_PPU
        : offset_q(!Varlen || cu_seqlens_q == nullptr ? 0 : __ldg(&cu_seqlens_q[bidb]))
        , offset_k(!Varlen || cu_seqlens_k == nullptr ? 0 : __ldg(&cu_seqlens_k[bidb]))
#else
        : offset_q(!Varlen || cu_seqlens_q == nullptr ? 0 : cu_seqlens_q[bidb])
        , offset_k(!Varlen || cu_seqlens_k == nullptr ? 0 : cu_seqlens_k[bidb])
#endif
        // If varlen, the layout for dPSum, LSE_log2, and dQaccum is that we pad each sequence in the batch
        // by an extra kBlockM, so that the write for each sequence doesn't touch the next sequence.
        // Sequence i starts at cu_seqlens[i] + i * kBlockM and ends at cu_seqlens[i + 1] + i * kBlockM
        // However, the start must align to multiples of kBlockM.
#ifdef USE_PPU
        , offset_q_padded(!Varlen || cu_seqlens_q == nullptr ? 0 : (__ldg(&cu_seqlens_q[bidb]) + bidb * kBlockM) / kBlockM * kBlockM)
        , seqlen_q(!Varlen
                   ? seqlen_q_static
                   : (seqused_q ? __ldg(&seqused_q[bidb]) : (cu_seqlens_q ? __ldg(&cu_seqlens_q[bidb + 1]) - __ldg(&cu_seqlens_q[bidb]) : seqlen_q_static)))
        , seqlen_k(!Varlen
                   ? seqlen_k_static
                   : (seqused_k ? __ldg(&seqused_k[bidb]) : (cu_seqlens_k ? __ldg(&cu_seqlens_k[bidb + 1]) - __ldg(&cu_seqlens_k[bidb]) : seqlen_k_static)))
#else
        , offset_q_padded(!Varlen || cu_seqlens_q == nullptr ? 0 : (cu_seqlens_q[bidb] + bidb * kBlockM) / kBlockM * kBlockM)
        , seqlen_q(!Varlen
                   ? seqlen_q_static
                   : (seqused_q ? seqused_q[bidb] : (cu_seqlens_q ? cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb] : seqlen_q_static)))
        , seqlen_k(!Varlen
                   ? seqlen_k_static
                   : (seqused_k ? seqused_k[bidb] : (cu_seqlens_k ? cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb] : seqlen_k_static)))
#endif
    {
    }

};

template <bool Varlen, bool AppendKV>
struct SeqlenInfoQKNewK {

    static_assert(!(AppendKV && !Varlen), "AppendKV is only supported with Varlen");

    int const leftpad_k;
    int const offset_q, offset_k, offset_k_new;
#ifndef FA3_HLLM_BUILD
    int const seqlen_q, seqlen_k_og, seqlen_k_new, seqlen_k, seqlen_rotary;
#else
    int seqlen_q, seqlen_k_og, seqlen_k_new, seqlen_k, seqlen_rotary;
    int const mrope_delta;
#endif // FA3_HLLM_BUILD

    CUTLASS_DEVICE
    SeqlenInfoQKNewK(int const bidb, int const seqlen_q_static, int const seqlen_k_static, int const shape_K_new_0,
                     int const* const cu_seqlens_q, int const* const cu_seqlens_k, int const* const cu_seqlens_k_new,
                     int const* const seqused_q, int const* const seqused_k, int const* const ptr_leftpad_k,
                     int const* const seqlens_rotary
#ifdef FA3_HLLM_BUILD
                     , bool is_generation_phase = false, int const* const mrope_position_deltas = nullptr
#endif // FA3_HLLM_BUILD
                     )
#ifdef USE_PPU
        : leftpad_k(ptr_leftpad_k ? __ldg(&ptr_leftpad_k[bidb]) : 0)
        , offset_q(!Varlen || cu_seqlens_q == nullptr ? 0 : __ldg(&cu_seqlens_q[bidb]))
        , offset_k(!Varlen ? 0 : (cu_seqlens_k ? __ldg(&cu_seqlens_k[bidb]) : 0) + leftpad_k)
        , offset_k_new(!AppendKV || cu_seqlens_k_new == nullptr ? 0 : __ldg(&cu_seqlens_k_new[bidb]))
        , seqlen_q(!Varlen
                   ? seqlen_q_static
                   : (seqused_q ? __ldg(&seqused_q[bidb]) : (cu_seqlens_q ? __ldg(&cu_seqlens_q[bidb + 1]) - __ldg(&cu_seqlens_q[bidb]) : seqlen_q_static)))
        , seqlen_k_og(!Varlen
                      ? seqlen_k_static
                      : (seqused_k ? __ldg(&seqused_k[bidb]) : (cu_seqlens_k ? __ldg(&cu_seqlens_k[bidb + 1]) - __ldg(&cu_seqlens_k[bidb]) : seqlen_k_static)) - leftpad_k)
        , seqlen_k_new(!AppendKV
                       ? 0
                       : (cu_seqlens_k_new ? __ldg(&cu_seqlens_k_new[bidb + 1]) - __ldg(&cu_seqlens_k_new[bidb]) : shape_K_new_0))
        , seqlen_k(!AppendKV ? seqlen_k_og : seqlen_k_og + seqlen_k_new)
        , seqlen_rotary(!AppendKV || !seqlens_rotary ? seqlen_k_og + leftpad_k : __ldg(&seqlens_rotary[bidb]))
#ifdef FA3_HLLM_BUILD
        , mrope_delta(mrope_position_deltas ? mrope_position_deltas[bidb] : 0)
#endif // FA3_HLLM_BUILD
#else
        : leftpad_k(ptr_leftpad_k ? ptr_leftpad_k[bidb] : 0)
        , offset_q(!Varlen || cu_seqlens_q == nullptr ? 0 : cu_seqlens_q[bidb])
        , offset_k(!Varlen ? 0 : (cu_seqlens_k ? cu_seqlens_k[bidb] : 0) + leftpad_k)
        , offset_k_new(!AppendKV || cu_seqlens_k_new == nullptr ? 0 : cu_seqlens_k_new[bidb])
        , seqlen_q(!Varlen
                   ? seqlen_q_static
                   : (seqused_q ? seqused_q[bidb] : (cu_seqlens_q ? cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb] : seqlen_q_static)))
        , seqlen_k_og(!Varlen
                      ? seqlen_k_static
                      : (seqused_k ? seqused_k[bidb] : (cu_seqlens_k ? cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb] : seqlen_k_static)) - leftpad_k)
        , seqlen_k_new(!AppendKV
                       ? 0
                       : (cu_seqlens_k_new ? cu_seqlens_k_new[bidb + 1] - cu_seqlens_k_new[bidb] : shape_K_new_0))
        , seqlen_k(!AppendKV ? seqlen_k_og : seqlen_k_og + seqlen_k_new)
        , seqlen_rotary(!AppendKV || !seqlens_rotary ? seqlen_k_og + leftpad_k : seqlens_rotary[bidb])
#endif
    {
#ifdef FA3_HLLM_BUILD
        // in hllm, we should sub seqlen_q from seqlen_k in generation phase to avoid out-of-bounds access to the KV cache
        // by default, seq_len_q is 1. When speculative decoding is enabled, seq_len_q = 1 + draft_tokens
        if (is_generation_phase) {
            seqlen_k_og -= seqlen_q;
            seqlen_k -= seqlen_q;
            seqlen_rotary -= seqlen_q;
        }
#endif // FA3_HLLM_BUILD
    }

};

} // namespace flash
