/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ni++) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); mi++) {
            summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, bool warp_reduce=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
    if constexpr (warp_reduce) { quad_allreduce_(sum, sum, sum_op); }
}

// Apply the exp to all the elements.
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0,
        typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    // For FP8, we can subtract max by 8.0 so that the value after exp2 is in the range of [0, 256].
    // This lets us use more of the FP8 range (instead of just [0, 1]) to reduce underflow.
    static constexpr float max_offset = float(Max_offset);  // We can only template on int, not float
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
    // max * log_2(e)). This allows the compiler to use the ffma
    // instruction instead of fadd and fmul separately.
    // Split into two loops: all FMAs first (independent, fully pipelinable),
    // then all exp2f (SFU can issue back-to-back). This breaks the serial
    // FMA->exp2f->FMA->exp2f dependency chain and improves ILP.
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
            : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = tensor(mi, ni) * scale - max_scaled;
        }
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = exp2f(tensor(mi, ni));
        }
    }
}

__device__ __forceinline__
__forceinline__ __device__ void exp2_approx_f32x2_ptx(float x, float y, float &out_x, float &out_y) {
    asm volatile(
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"

        // 1) clamp: f1=max(x,-127), f2=max(y,-127)
        "ppu.max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
        "ppu.max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"

        // 2) pack (x,y) into l1
        "ppu.mov.b64 l1, {f1, f2};\n\t"

        // 3) magic = 2^23 + 2^22, pack into l2
        "ppu.mov.f32 f3, 0f4B400000;\n\t"
        "ppu.mov.b64 l2, {f3, f3};\n\t"

        // 4) floor trick with RM, then recover frac with RN
        "ppu.add.rtn.ftz.f32x2 l7, l1, l2;\n\t"
        "ppu.sub.rtte.ftz.f32x2 l8, l7, l2;\n\t"
        "ppu.sub.rtte.ftz.f32x2 l9, l1, l8;\n\t"   // l9 = frac(x), frac(y)

        // 5) polynomial coeffs: c3,c2,c1,c0
        "ppu.mov.f32 f7, 0f3D9DF09D;\n\t"        // c3
        "ppu.mov.b64 l6, {f7, f7};\n\t"
        "ppu.mov.f32 f6, 0f3E6906A4;\n\t"        // c2
        "ppu.mov.b64 l5, {f6, f6};\n\t"
        "ppu.mov.f32 f5, 0f3F31F519;\n\t"        // c1
        "ppu.mov.b64 l4, {f5, f5};\n\t"
        "ppu.mov.f32 f4, 0f3F800000;\n\t"        // c0
        "ppu.mov.b64 l3, {f4, f4};\n\t"

        // 6) Horner (packed)
        "ppu.fma.rtte.ftz.f32x2 l10, l9,  l6, l5;\n\t"
        "ppu.fma.rtte.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "ppu.fma.rtte.ftz.f32x2 l10, l10, l9, l3;\n\t"

        // 7) bit-combine: (int_part<<23) + poly_bits
        "ppu.mov.b64 {r1, r2}, l7;\n\t"          // r1 bits for lane0, r2 for lane1
        "ppu.mov.b64 {r3, r4}, l10;\n\t"         // poly bits

        "ppu.shl.b32 r5, r1, 23;\n\t"
        "ppu.add.s32 r7, r5, r3;\n\t"            // out_x bits

        "ppu.shl.b32 r6, r2, 23;\n\t"
        "ppu.add.s32 r8, r6, r4;\n\t"            // out_y bits

        // 8) write outputs as float (bitcast through "=f" outputs)
        "ppu.mov.b32 %0, r7;\n\t"
        "ppu.mov.b32 %1, r8;\n\t"

        "}\n\t"
        : "=f"(out_x), "=f"(out_y)
        : "f"(x), "f"(y)
    );
}

// Apply the ex2_emulation_2 to all the elements.
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0,
        typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_ex2_emulation_2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    // For FP8, we can subtract max by 8.0 so that the value after exp2 is in the range of [0, 256].
    // This lets us use more of the FP8 range (instead of just [0, 1]) to reduce underflow.
    static constexpr float max_offset = float(Max_offset);  // We can only template on int, not float
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));

    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
            : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ni += 2) {
            float x0 = tensor(mi, ni + 0) * scale - max_scaled;
            float x1 = tensor(mi, ni + 1) * scale - max_scaled;

            float y0, y1;
            flash::exp2_approx_f32x2_ptx(x0, x1, y0, y1);

            tensor(mi, ni) = y0;
            tensor(mi, ni + 1) = y1;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows, int Max_offset=0>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;
    bool need_rescale;
    float const softmax_scale_log2;

    CUTLASS_DEVICE Softmax(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale_skip(Tensor0 &acc_s) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        TensorT scores_scale;
        if constexpr (Is_first) {
            need_rescale = false;
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            cute::fill(scores_scale, 1.f);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            need_rescale = false;
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = (scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2;
                need_rescale |= (scores_scale(mi) < -8.0);
            }
            if (need_rescale) {
               #pragma unroll
                for (int mi = 0; mi < size(row_max); ++mi) {
                    scores_scale(mi) = exp2f(scores_scale(mi));
                    row_sum(mi) *= scores_scale(mi);
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size(row_max); ++mi) {
                    row_max(mi) = scores_max_prev(mi);
                    scores_scale(mi) = 1.0f;
                }
            }
        }
        return scores_scale;
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        TensorT scores_scale;
        if constexpr (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            cute::fill(scores_scale, 1.f);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
        }
        return scores_scale;
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
        // We don't do the reduce across threads here since we don't need to use the row_sum.
        // We do that reduce at the end when we need to normalize the softmax.
        flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);
    };


#ifdef USE_PPU
    // N-split O warps own column slabs, so their row scales come from
    // the QK warps through shared memory. The caller publishes these scales
    // with a CTA barrier. Issue all scale loads before the SFU work to avoid
    // reusing live exp2 input registers for their destinations (SFU WAR).
    // This boundary constrains compiler scheduling; it is not a CTA barrier.
    template<bool Check_inf, typename TensorS, typename TensorO>
    __forceinline__ __device__ void online_softmax_rescale_o_nsplit(
            TensorS &acc_s, TensorO &acc_o, const float* scale_smem) {
        Tensor output = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        constexpr int OutputRows = decltype(size<0>(output))::value;
        const int p = (threadIdx.x % 32) / 4;
        float scales[OutputRows];
        #pragma unroll
        for (int row = 0; row < OutputRows; ++row) {
            scales[row] = scale_smem[16 * (row / 2) + p + 8 * (row % 2)];
        }
        __ppu_sched_bound();
        online_softmax</*Is_first=*/false, Check_inf>(acc_s);
        #pragma unroll
        for (int row = 0; row < OutputRows; ++row) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(output); ++ni) {
                output(row, ni) *= scales[row];
            }
        }
    }
#endif

    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.
            if constexpr (Max_offset != 0) {
                static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                sum *= sum_scale;
            }
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    };

    __forceinline__ __device__ TensorT finalize_aux(TensorT const& tSrSAux, float const final_scale=1.f) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            if (row_max(mi) == -INFINITY) { row_max(mi) = 0.f; }
            const float max_scaled = row_max(mi) * softmax_scale_log2 - Max_offset;
            float sum = row_sum(mi) + exp2f(float(M_LOG2E) * tSrSAux(mi) - max_scaled);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.
            if constexpr (Max_offset != 0) {
                static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                sum *= sum_scale;
            }
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    };

#ifdef USE_PPU
    // N-split scale broadcast: write this warp's per-row scales into the
    // smem broadcast area so every warp can rescale its N-split slice of O
    // against all kBlockM rows (flash::rescale_o_nsplit in utils.h). Each
    // lane holds the scales of rows p and p+8 within its warp's 16 M-split
    // rows (nrow mode (v1, MMA_M=1): row = warp_idx*16 + p + 8*mi); only
    // q==0 lanes store (redundant stores of the same value are harmless).
    // Pass the scores_scale returned by finalize() / the running rescale
    // factors — NOT row_sum, which finalize() turns into LSE.
    template<typename TensorScales>
    __forceinline__ __device__ void write_scales_to_smem(float* scale_ptr, int warp_idx,
                                                         TensorScales const& scores_scale) const {
        static_assert(decltype(size(scores_scale))::value == kNRows);
        const int lane = threadIdx.x % 32;
        const int p = lane / 4;
        const int q = lane % 4;
        if (q == 0) {
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
                scale_ptr[warp_idx * kNRows * 8 + mi * 8 + p] = scores_scale(mi);
            }
        }
    }
#endif

    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
        }
    };

};

}  // namespace flash
