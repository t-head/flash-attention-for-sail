/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <hggc_fp16.h>

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 100
#include <hggc_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cuda_check.h"

#ifdef USE_PPU
#include "ppu_include.hpp"
#include <cute/arch/copy_ppu.hpp>
#endif
#if !USE_AIU
#define make_mix_tensor make_tensor
#define make_mix_tensor_like(x)  x
#endif
#ifdef USE_PPU
#include "acc_vreg_fraga.h"
#endif

#ifdef USE_PPU
// Epilogue-only twin of the PPU1.5 FP8 MMA atom with a permuted CLayout that
// matches the pi_n column order produced by the zero-shfl V-direct PV path
// (after flash::permute_output_fp8): lane (q=lane%4, p=lane/4) register slot
// d[v0+2*v1+4*v2] holds (m = p+8*v1, n = 4*q + v0 + 2*v2) instead of the
// standard (m = p+8*v1, n = 2*q + v0 + 8*v2). Never execute MMA through this
// atom; it only re-interprets accumulator coordinates for the epilogue.
namespace cute {
struct PPU0015_16x16x32_F32E4M3E4M3F32_TN_EpiPerm : PPU0015_16x16x32_F32E4M3E4M3F32_TN {};
template <>
struct MMA_Traits<PPU0015_16x16x32_F32E4M3E4M3F32_TN_EpiPerm>
    : MMA_Traits<PPU0015_16x16x32_F32E4M3E4M3F32_TN> {
  using CLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,_2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_8,_32>>>;
};
} // namespace cute
#endif

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// A wrapper for the kernel that is used to guard against compilation on
// architectures that will never use the kernel. The purpose of this is to
// reduce the size of the compiled binary.
// Adapted from https://github.com/vllm-project/vllm/blob/4d29e91be84d27ca313d657eee92c067439a4c23/csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cuh#L55
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ > 150)
        Kernel::operator()(std::forward<Args>(args)...);
#endif
    }
};

template <typename Kernel>
struct enable_sm80_to_sm89 : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100) && (__HGGC_ARCH__ <= 150)
        Kernel::operator()(std::forward<Args>(args)...);
#endif
    }
};

#ifdef USE_PPU
template <typename Kernel>
struct enable_sm80 : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
        Kernel::operator()(std::forward<Args>(args)...);
#endif
    }
};

template <typename Kernel>
struct enable_sm89 : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
        Kernel::operator()(std::forward<Args>(args)...);
#endif
    }
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
int div_floor(cutlass::FastDivmod const& divmod, int dividend) {
    // Take care of the negative case: https://stackoverflow.com/questions/39304681/division-with-negative-dividend-but-rounded-towards-negative-infinity
    // Maybe the compiler will turn the -1 - * into bit negation operation, I haven't checked.
    return dividend >= 0 ? divmod.divide(dividend) : -1 - divmod.divide(-1 - dividend);
}

CUTLASS_HOST_DEVICE
int round_down(cutlass::FastDivmod const& divmod, int dividend) {
    return div_floor(divmod, dividend) * divmod.divisor;
}

CUTLASS_HOST_DEVICE
int round_up(cutlass::FastDivmod const& divmod, int dividend) {
    return div_floor(divmod, dividend - 1) * divmod.divisor + divmod.divisor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// For SM80, convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
// For SM90, convert acc_layout from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
template<bool Transposed=false, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout0 acc_layout) {
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90, also applies to ppu1.5
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = acc_layout;
        if constexpr (!Transposed) {
            return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
        } else {
             return make_layout(make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));
        }

    } else {  // SM80
#ifdef USE_PPU   // applies to ppu1.0
        // acc is ppu c layout, size0 is 8, MMA_N size is A100 MMA_N/2
        static_assert(decltype(size<0>(acc_layout))::value == 8);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_4>{}); //((2, 4), MMA_M, MMA_N)
#else
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
#endif
        if constexpr (!Transposed) {
            return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
        } else {
            return make_layout(make_layout(get<0, 0>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// For SM80, convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
// For SM90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
// For SM90, FP8, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((4, 2, 2), MMA_M, (N / 32, MMA_N))
template<typename MMA_Traits, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {
    using X = Underscore;
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90, also applies to ppu1.5
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        static_assert(decltype(rank(acc_layout))::value == 3);
        static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
        if constexpr (sizeof(typename MMA_Traits::ValTypeA) == 2) {
            auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});  // ((2, N / 16))
            return make_layout(make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)), get<1>(acc_layout), coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
        } else {
#if !defined(USE_PPU)
            static_assert(sizeof(typename MMA_Traits::ValTypeA) == 1);
            static_assert(decltype(stride<0, 0>(acc_layout))::value == 1);
            static_assert(decltype(stride<0, 1>(acc_layout))::value == 2);
            auto l = logical_divide(get<0, 2>(acc_layout), Tile<Layout<Shape<_2, _2>>>{});  // (((2, 2), N / 32))
            // This combines the first two modes (<0, 0> and <0, 1>) into one mode.
            // Will require register shuffling later to be correct.
            return make_layout(make_layout(Layout<_4>{}, get<0, 0, 0>(l), get<0, 0, 1>(l)),
                               get<1>(acc_layout),
                               coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));  // ((4, 2, 2), MMA_M, N / 32 * MMA_N)
            // This combination is right but doesn't work with register shuffling.
            // return make_layout(make_layout(coalesce(make_layout(get<0, 0>(acc_layout), get<0, 0, 0>(l))), get<0, 1>(acc_layout), get<0, 0, 1>(l)),
            //                    get<1>(acc_layout),
            //                    coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
#else
            static_assert(sizeof(typename MMA_Traits::ValTypeA) == 1);
            static_assert(decltype(stride<0, 0>(acc_layout))::value == 1);
            static_assert(decltype(stride<0, 1>(acc_layout))::value == 2);
            auto l = logical_divide(select<0, 2>(acc_layout), Layout<Shape<_4, _2, _2>>{});
            auto res = make_layout(get<0>(l), get<1>(acc_layout), get<1>(l));
            return res;
#endif
        }
    } else {  // SM80
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        constexpr int mma_shape_K = get<2>(typename MMA_Traits::Shape_MNK{});
        static_assert(mma_shape_K == 8 || mma_shape_K == 16);
        if constexpr (mma_shape_K == 8) {
            return acc_layout;
        } else {
            auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
            return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_PPU
template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_acc(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    NumericArrayConverterPPU<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type_unsafe(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    // Unsafe because we're returning a tensor with memory allocated on the stack. If the compiler does not
    // inline this function, then the memory might not be valid.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const &tensor, Tensor<EngineOut, Layout> &out) {
    // Somehow if we allocate out inside this function and return it, e2e is slower and the output can be wrong.
    using From_type = typename Engine::value_type;
    using To_type = typename EngineOut::value_type;
    static constexpr int FragmentSize = std::max(sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
    static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0, "Fragment size does not vectorize properly");
    Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
    Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
    static_assert(size(frag) == size(out_frg));
    cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
    #pragma unroll
    for (int i = 0; i < size(frag); ++i) { out_frg[i] = convert_op(frag[i]); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_PPU_ENABLED)
    asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool A, class Mma, class Tensor0>
CUTLASS_DEVICE
auto mma_partition_fragment_AB(Mma const& mma, Tensor0 const& tensor0) {
    if constexpr (A) {
        return mma.partition_fragment_A(tensor0);
    } else {
        return mma.partition_fragment_B(tensor0);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool zero_init=false, int wg_wait=0, bool SwapAB=false, int M_slice=-1,
        typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTLASS_DEVICE void gemm(TiledMma& tiled_mma, Tensor0 const& tCrA, Tensor1 const& tCrB, Tensor2& tCrC) {
#ifndef USE_PPU
    if constexpr (M_slice >= 0) {
        static constexpr int MMA_M = decltype(size<1>(tCrC))::value;
        static_assert(M_slice < MMA_M);
        // After logical_divide, C has shape ((2,2,V), (MMA_M, 1), MMA_N)
        Tensor tCrC_slice = cute::logical_divide(tCrC, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
        if constexpr (!SwapAB) {
            Tensor tCrA_slice = cute::logical_divide(tCrA, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
            gemm<zero_init, wg_wait, SwapAB, /*M_slice=*/-1>(tiled_mma, tCrA_slice, tCrB, tCrC_slice);
        } else {
            Tensor tCrB_slice = cute::logical_divide(tCrB, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
            gemm<zero_init, wg_wait, SwapAB, /*M_slice=*/-1>(tiled_mma, tCrA, tCrB_slice, tCrC_slice);
        }
    } else {
        constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
        // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
        if constexpr (Is_RS) {
            if constexpr (!SwapAB) {
                warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA));
            } else {
                warpgroup_fence_operand(const_cast<Tensor1 &>(tCrB));
            }
        }
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        if constexpr (zero_init) {
            tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        }
        static constexpr int kNumKIters = CUTE_STATIC_V(size<2>(tCrA));
        static constexpr int kMaxKIters = 16;
        // Unroll the K mode manually to set scale D to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < std::min(kNumKIters, kMaxKIters); ++k_block) {
            if constexpr (!SwapAB) {
                cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            } else {
                cute::gemm(tiled_mma, tCrB(_,_,k_block), tCrA(_,_,k_block), tCrC);
            }
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        // In the case of large kNumKIters, the compiler chooses to store the smem addresses
        // in registers, causing spills. This loop forces the compiler to recompute the addresses.
        if constexpr (kNumKIters > kMaxKIters) {
            // This will always be zero, just a way to force the compiler to recompute the smem
            // addresses. This results in USEL instructions. There's probably a better way to do this.
            int const k_offset = cutlass::canonical_warp_group_idx() < 128 ? 0 : 1;
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = kMaxKIters; k_block < kNumKIters; ++k_block) {
                if constexpr (!SwapAB) {
                    cute::gemm(tiled_mma, tCrA(_,_,k_block + k_offset), tCrB(_,_,k_block + k_offset), tCrC);
                } else {
                    cute::gemm(tiled_mma, tCrB(_,_,k_block + k_offset), tCrA(_,_,k_block + k_offset), tCrC);
                }
                tiled_mma.accumulate_ = GMMA::ScaleOut::One;
            }
        }
        warpgroup_commit_batch();
        if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
        warpgroup_fence_operand(tCrC);
        if constexpr (Is_RS) {
            if constexpr (!SwapAB) {
                warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA));
            } else {
                warpgroup_fence_operand(const_cast<Tensor1 &>(tCrB));
            }
        }
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, bool SwapAB=false,
         typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB, typename Hook>
CUTLASS_DEVICE void gemm_sm80(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                              Tensor4 const& tCsB, TiledMma tiled_mma,
                              TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                              ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B, Hook fn) {
    if constexpr (SwapAB) {
        gemm_sm80<B_in_regs, A_in_regs>(acc, tCrB, tCrA, tCsB, tCsA, tiled_mma, smem_tiled_copy_B, smem_tiled_copy_A, smem_thr_copy_B, smem_thr_copy_A, fn);
    } else {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
#ifdef USE_PPU
        #pragma hggc mmatiestrictly
        {
        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i), tCrA_copy_view(_, _, i)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i), tCrB_copy_view(_, _, i)); }
            if constexpr (!std::is_same_v<Hook, std::nullptr_t>) {
                if (i == 0) { fn(); }
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
        }
#else
        if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
        if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
                if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
            }
            if constexpr (!std::is_same_v<Hook, std::nullptr_t>) {
                if (i == 0) { fn(); }
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_PPU
template<int kBlockN, int kBlockNPagedPerAiuLoad, int kHeadDim, int kBlockKSmem,
         bool A_in_regs=false, bool B_in_regs=false, bool SwapAB=false,
         typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB, typename Hook>
CUTLASS_DEVICE void gemm_sm80_kv_paged_aiu(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                                           Tensor4 const& tCsB, TiledMma tiled_mma,
                                           TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                                           ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B, Hook fn) {
    if constexpr (SwapAB) {
        gemm_sm80_kv_paged_aiu<kBlockN, kBlockNPagedPerAiuLoad, kHeadDim, kBlockKSmem, B_in_regs, A_in_regs>(acc, tCrB, tCrA, tCsB, tCsA, tiled_mma, smem_tiled_copy_B, smem_tiled_copy_A, smem_thr_copy_B, smem_thr_copy_A, fn);
    } else {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
        constexpr int OFFSET_MMA_N = kBlockNPagedPerAiuLoad * kBlockKSmem;
        constexpr int OFFSET_TILE = kBlockN * kBlockKSmem;
        int MMA_K_PER_TILE = size<2>(tCrB) / (kHeadDim / kBlockKSmem);
        Tensor tCsB_rTile = make_tensor(tCsB.data(), tCsB.layout());
        Tensor tCsB_sTile = make_tensor(tCsB.data(), tCsB.layout());
        #pragma hggc mmatiestrictly
        {
        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i), tCrA_copy_view(_, _, i)); }
            if (!B_in_regs) {
                tCsB_rTile.data() = tCsB_sTile.data();
                for (int n = 0; n < size<1>(tCrB) / (kBlockNPagedPerAiuLoad / 16); n++) {
                    #pragma unroll
                    for (int j = 0; j < kBlockNPagedPerAiuLoad / 16; j++) {
                        cute::copy(smem_tiled_copy_B, tCsB_rTile(_, j, i % MMA_K_PER_TILE), tCrB_copy_view(_, n * (kBlockNPagedPerAiuLoad / 16) + j, i));
                    }
                    tCsB_rTile.data() = tCsB_rTile.data() + OFFSET_MMA_N;
                }
                if ((i + 1) % MMA_K_PER_TILE == 0) {
                    tCsB_sTile.data() = tCsB_sTile.data() + OFFSET_TILE;
                }
            }
            if constexpr (!std::is_same_v<Hook, std::nullptr_t>) {
                if (i == 0) { fn(); }
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
        }
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
CUTLASS_DEVICE void gemm_rs_sm80(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                 TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                 ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
#ifdef USE_PPU
    #pragma hggc mmatiestrictly
    {
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, i), tCrB_copy_view(_, _, i));
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
    }
#else
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_PPU
template<int kBlockN, int kBlockNPagedPerAiuLoad, int kHeadDim, int kBlockKSmem,
         typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
CUTLASS_DEVICE void gemm_rs_sm80_kv_paged_aiu(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                       TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                       ThrCopy smem_thr_copy_B) {
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    constexpr int OFFSET_MMA_N = kBlockNPagedPerAiuLoad * kBlockKSmem;
    constexpr int OFFSET_TILE = kBlockN * kBlockKSmem;
    int MMA_K_PER_TILE = size<1>(tCrB) / (kHeadDim / kBlockKSmem);
    Tensor tCsB_rTile = make_tensor(tCsB.data(), tCsB.layout());
    Tensor tCsB_sTile = make_tensor(tCsB.data(), tCsB.layout());
    #pragma hggc mmatiestrictly
    {
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA) / (kBlockNPagedPerAiuLoad / 16); ++i) {
        tCsB_rTile.data() = tCsB_sTile.data();
        for (int k = 0; k < size<1>(tCrB); k++) {
            #pragma unroll
            for (int j = 0; j < kBlockNPagedPerAiuLoad / 16; ++j) {
                cute::copy(smem_tiled_copy_B, tCsB_rTile(_, k % MMA_K_PER_TILE, j), tCrB_copy_view(_, k, i * (kBlockNPagedPerAiuLoad / 16) + j));
            }
            if ((k + 1) % MMA_K_PER_TILE == 0) {
                tCsB_rTile.data() = tCsB_rTile.data() + OFFSET_TILE;
            }
        }
        tCsB_sTile.data() = tCsB_sTile.data() + OFFSET_MMA_N;
        #pragma unroll
        for (int j = 0; j < kBlockNPagedPerAiuLoad / 16; ++j) {
            cute::gemm(tiled_mma, tCrA(_, _, i * (kBlockNPagedPerAiuLoad / 16) + j), tCrB(_, _, i * (kBlockNPagedPerAiuLoad / 16) + j), acc);
        }
    }
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP8 PV: register-direct V load, zero-shfl pi_k/pi_n permutation scheme.
//
// Device-measured hardware truths (PPU1.5, hopper/test_tsm_trans_semantics.cu):
//   * TSM `ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans` request
//     (coord_h=kv0, coord_w=w0 in b16 units, cube ic) on the cube-blocked raw
//     sV tile ([NCUBE][kBlockN][kBlockKGmem] e4m3, as written by the AIU g2s
//     load) delivers to lane (q=lane%4, p=lane/4):
//       r0 = {V[kv0+2q][2t], V[kv0+2q][2t+1], V[kv0+2q+1][2t], V[kv0+2q+1][2t+1]}
//       with t = ic*(CUBE_W) + w0 + p; r1: kv+8; r2: t+8; r3: kv+8 and t+8.
//   * One x4 request covers a 16-row x 16-b16-col (32B) window that must lie
//     fully inside its cube (coord_h+16<=CUBE_H, coord_w+16<=CUBE_W), so
//     requests are issued per n-tile PAIR (j0=32*np): r0/r1 serve the even
//     n-tile, r2/r3 the odd one. The TSM bulk address must be warp-uniform.
//   * MMA m16n16k32 e4m3 fragment layouts (measured):
//       A(lane u32 j, byte kq) = (m=p+8*(j&1), k=4q+kq+16*(j>>1))
//       B(lane u32 w, byte kq) = (k=4q+kq+16*(w&1), n=p+8*(w>>1))
//       C(lane d[v0+2v1+4v2])  = (m=p+8*v1, n=2q+v0+8*v2)
//
// pi_k/pi_n framework: the reduction axis (k) may be permuted freely as long
// as A and B carry the same real k at every (lane,reg,byte) slot; the n axis
// may be permuted if the accumulator is un-permuted once at the end.
//   pi_k: slot k=4q+kq (+16 for j>>1 / w&1) carries real k = 2q+(kq&1)+8*(kq>>1)
//         (+16 for the high half). A (permute_Cregs_fp8 + the plain
//         convert_layout_acc_Aregs view, see mainloop_fwd_sm80.hpp) and B
//         (below) agree on this slot-for-slot, so the MMA sum is exact.
//   pi_n: slot n=p carries real n=2*(t0+p), slot n=p+8 carries real
//         n=2*(t0+p)+1 (t0 = n16-tile base in b16 units). The accumulator then
//         holds columns {4q,4q+2,4q+1,4q+3} at d{d0,d1,d4,d5};
//         permute_output_fp8 (d1<->d4, d3<->d6) sorts them into
//         n = 4q+v0+2*v2, which the epilogue consumes through the permuted
//         CLayout atom PPU0015_16x16x32_F32E4M3E4M3F32_TN_EpiPerm.
//
// Assembly (per n-tile pair, kv0=32*mk, j0=32*np), lane (q,p), h in {0,1}
// selecting n-tile mn=2*np+h, requests A(kv0)/B(kv0+16):
//   w0 = byte_perm(rA[2h], rA[2h+1], 0x6420)  // real k {2q,2q+1,2q+8,2q+9}, real n=2t
//   w1 = byte_perm(rB[2h], rB[2h+1], 0x6420)  // same, k+16
//   w2 = byte_perm(rA[2h], rA[2h+1], 0x7531)  // real n=2t+1
//   w3 = byte_perm(rB[2h], rB[2h+1], 0x7531)  // same, k+16
// Cost per n-tile: 1 TSM_LD_SWZL + 4 byte_perm, zero shfl/sel
// (per n-tile pair: 2 TSM_LD_SWZL + 8 byte_perm).
//
// Issue granularity: the MMAs are emitted PER N-TILE (atom-granular) right
// after that tile's B fragment is assembled, not in one batch over the full
// N width. Live B registers drop from MMA_N*4 u32 (64 for hdim256) to ~8 u32
// (one TSM pair rA/rB + one assembled fragment), and the compiler is free to
// software-pipeline the next pair's TSM loads against the in-flight MMAs.
// The mk loop stays outermost so every (m,n) accumulator sees ascending k.
//
// UseTsmLd selects the smem read instruction:
//   true  (non-paged / paged-AIU): TSM ldmatrix.swzl b16.trans on the
//          cube-blocked raw tile written by the AIU g2s load.
//   false (paged non-AIU): plain ldmatrix.x4.trans (PPU_U16x8_LDSM_T) with
//          per-lane row addresses evaluated through SmemLayoutVRaw (the raw
//          XOR-swizzle row layout; its swizzle permutes whole 16B chunks
//          only, so j0 %% 16 == 0 addresses are 16B-aligned chunk starts).
//          Per (mk, np), lane l issues two x4 loads at &sV(32mk+l, j0) and
//          &sV(32mk+l, j0+16) and remaps the halves to the TSM pair:
//            rA = {ldA0, ldA1, ldB0, ldB1}, rB = {ldA2, ldA3, ldB2, ldB3}.
//          Device-verified byte-for-byte identical to the TSM pair
//          (hopper/test_tsm_trans_semantics.cu phase 5, all four tiles).
//          The byte_perm assembly + per-atom MMA emission below is shared.
#ifdef USE_PPU
template<int kBlockN, int kHeadDimV, int kBlockKGmem,
         bool UseTsmLd, typename SmemLayoutVRaw,
         typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTLASS_DEVICE void gemm_rs_sm80_pv_fp8_vdirect(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB,
                                                void *smem_v, int stage, TiledMma tiled_mma) {
    static_assert(kBlockN % 32 == 0 && kHeadDimV % 32 == 0 && kHeadDimV % kBlockKGmem == 0);
    static_assert(decltype(size<1>(tCrB))::value == kHeadDimV / 16, "tCrB MMA_N mismatch");
    static_assert(decltype(size<2>(tCrB))::value == kBlockN / 32, "tCrB MMA_K mismatch");
    static_assert(decltype(size<2>(tCrA))::value == kBlockN / 32, "tCrA MMA_K mismatch");
    // b16 element view of the raw sV cube: CUBE_W in b16 units = kBlockKGmem/2.
    using TsmOp = cute::PPU0015_TSM_LD_SWZL<cute::bfloat16_t, kBlockN, kBlockKGmem / 2,
                                            true /*Swap*/, true /*Trans*/, kHeadDimV / kBlockKGmem>;
    constexpr int MMA_K = kBlockN / 32;
    constexpr int MMA_NP = kHeadDimV / 32;  // n-tiles processed in pairs
    Tensor tCrB32 = cute::recast<uint32_t>(tCrB);
    CUTE_STATIC_ASSERT_V(size<0>(tCrB32) == Int<4>{});
    // Raw b8 view of sV for the LDSM_T path (type-identical to SmemLayoutV in
    // the paged non-AIU FP8 instantiation this path is compiled for).
    Tensor sVraw = make_tensor(make_smem_ptr(reinterpret_cast<uint8_t const*>(smem_v)), SmemLayoutVRaw{});
    const int lane = threadIdx.x % 32;
    #pragma hggc mmatiestrictly
    {
    #pragma unroll
    for (int mk = 0; mk < MMA_K; ++mk) {
        #pragma unroll
        for (int np = 0; np < MMA_NP; ++np) {
            const int j0 = 32 * np;
            uint32_t rA[4], rB[4];
            if constexpr (UseTsmLd) {
                const int cube = j0 / kBlockKGmem;
                const int wcoord = (j0 % kBlockKGmem) / 2;  // b16 units
                TsmOp::copy(rA, smem_v, /*coord_h=*/32 * mk,      /*coord_w=*/wcoord, cube, stage);
                TsmOp::copy(rB, smem_v, /*coord_h=*/32 * mk + 16, /*coord_w=*/wcoord, cube, stage);
            } else {
                const int kv0 = 32 * mk;
                uint32_t ldA[4], ldB[4];
                const uint8_t* addrA = &sVraw(kv0 + lane, j0,      stage);
                const uint8_t* addrB = &sVraw(kv0 + lane, j0 + 16, stage);
                cute::PPU_U16x8_LDSM_T::copy(*reinterpret_cast<cute::uint128_t const*>(addrA),
                                             ldA[0], ldA[1], ldA[2], ldA[3]);
                cute::PPU_U16x8_LDSM_T::copy(*reinterpret_cast<cute::uint128_t const*>(addrB),
                                             ldB[0], ldB[1], ldB[2], ldB[3]);
                // Remap the two 32-row x 16B LDSM results to the TSM pair:
                // rA serves rows kv0..kv0+15, rB rows kv0+16..kv0+31, both
                // covering byte cols j0..j0+31 (r2/r3 = the j0+16 halves).
                rA[0] = ldA[0]; rA[1] = ldA[1]; rA[2] = ldB[0]; rA[3] = ldB[1];
                rB[0] = ldA[2]; rB[1] = ldA[3]; rB[2] = ldB[2]; rB[3] = ldB[3];
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int mn = 2 * np + h;
                // Zero-shfl pi_k/pi_n B-fragment assembly (see header comment).
                tCrB32(0, mn, mk) = __byte_perm(rA[2 * h + 0], rA[2 * h + 1], 0x6420u);
                tCrB32(1, mn, mk) = __byte_perm(rB[2 * h + 0], rB[2 * h + 1], 0x6420u);
                tCrB32(2, mn, mk) = __byte_perm(rA[2 * h + 0], rA[2 * h + 1], 0x7531u);
                tCrB32(3, mn, mk) = __byte_perm(rB[2 * h + 0], rB[2 * h + 1], 0x7531u);
                // Emit the 16x16x32 atom MMA(s) for this n-tile immediately:
                // rank-1 fragment slices hit cute::gemm dispatch [1] (V)x(V)->(V)
                // which forwards straight to MMA_Atom::call. The (mn, mk) B slot
                // is dead right after the call, so the compiler can recycle its
                // registers instead of keeping the whole N width live.
                #pragma unroll
                for (int m = 0; m < size<1>(acc); ++m) {
                    cute::gemm(tiled_mma, tCrA(_, m, mk), tCrB(_, mn, mk), acc(_, m, mn));
                }
            }
        }
    }
    }
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool zero_init=false, typename Atom, typename TA, typename TB, typename TC>
CUTLASS_DEVICE void gemm_sm100(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
    static constexpr int rA = decltype(rank(tA))::value;
    static constexpr int rB = decltype(rank(tB))::value;
    static constexpr int rC = decltype(rank(tC))::value;
    static_assert(rA == 3 && rB == 3 && rC == 3);

    if constexpr (zero_init) { atom.accumulate_ = decltype(atom.accumulate_)::Zero; }
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tA); k_block++) {
        cute::gemm(atom, tA(_,_,k_block), tB(_,_,k_block), tC);
        atom.accumulate_ = decltype(atom.accumulate_)::One;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef FLASHATTENTION_DISABLE_SM90
template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr
auto
to_tiled_mma_sm100_ts(
    TiledMMA<MMA_Atom<
      MMA_Traits<SM100_MMA_F8F6F4_SS, a_type, b_type, c_type,
                    cute::C<M>, cute::C<N>,
                    cute::integral_constant<UMMA::Major, a_major>,
                    cute::integral_constant<UMMA::Major, b_major>,
                    cute::integral_constant<UMMA::ScaleIn, a_neg>,
                    cute::integral_constant<UMMA::ScaleIn, b_neg>>,
      TAs...>, TMs...>) {

  return TiledMMA<MMA_Atom<
    MMA_Traits<SM100_MMA_F8F6F4_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, UMMA::Saturate::False>>,
    TAs...>, TMs...>{};
}

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr
auto
to_tiled_mma_sm100_ts(
    TiledMMA<MMA_Atom<
      SM100_MMA_F16BF16_SS<a_type, b_type, c_type,
                    M, N,
                    a_major,
                    b_major,
                    a_neg,
                    b_neg>,
      TAs...>, TMs...>) {
  return TiledMMA<MMA_Atom<
    SM100_MMA_F16BF16_TS<a_type, b_type, c_type,
                                M, N,
                                a_major, b_major,
                                a_neg, b_neg, UMMA::Saturate::False>,
    TAs...>, TMs...>{};
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

//PPU: shared memory not support init by zero, need clear if not align.
#ifdef USE_PPU
template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=true, bool Clear_OOB_K=true,
#else
template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
#endif
          class CopyAtom, class TV, class Tiler, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void copy(TiledCopy<CopyAtom, TV, Tiler> &tiled_copy, Tensor<Engine0, Layout0> const &S,
                         Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                         Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
// support AIU on PPU
#if defined(USE_PPU) && USE_AIU
    if constexpr (is_mix_iterator<typename Engine0::iterator>::value) {
        const int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
        if (warp_idx == 0) {
            if constexpr (!Is_even_MN) {
                tiled_copy.desc_.dim_h = max_MN;
            }
            cute::copy(tiled_copy, S, D);
        }
        return;
    }
#endif
    // Decay TiledCopy to CopyAtom
    auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    auto has_with_bool = cute::is_valid([](auto t)->void_t<decltype(declval<typename decltype(t)::Traits>().with(true))>{}, copy_atom);
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool predicate_mn = Is_even_MN || get<0>(identity_MN(_0{}, m, _0{})) < max_MN;
        if constexpr (Is_even_MN || !Clear_OOB_MN) {
            if (Is_even_MN || predicate_mn) {
                #pragma unroll
                for (int k = 0; k < size<2>(S); ++k) {
                    if constexpr (Is_even_K || !Clear_OOB_K) {
                        if (Is_even_K || predicate_K(k)) { cute::copy(copy_atom, S(_, m, k), D(_, m, k)); }
                    } else {  // Clear_OOB_K == true && Is_even_K == false
                        // If copy traits can be transformed with a predicate value, do it, otherwise branch here
                        if constexpr (has_with_bool) {
                            cute::copy(copy_atom.with(predicate_K(k)), S(_, m, k), D(_, m, k));
                        } else {
                            if (predicate_K(k)) {
                                cute::copy(copy_atom, S(_, m, k), D(_, m, k));
                            } else {
                                cute::clear(D(_, m, k));
                            }
                        }
                    }
                }
            }
        } else {  // Clear_OOB_MN == true && Is_even_MN == false, also implies Clear_OOB_K == true
            if constexpr (!has_with_bool) {
                if (predicate_mn) {
                    #pragma unroll
                    for (int k = 0; k < size<2>(S); ++k) {
                        if (Is_even_K || predicate_K(k)) {
                            cute::copy(copy_atom, S(_, m, k), D(_, m, k));
                        } else if (Clear_OOB_K) {
                            cute::clear(D(_, m, k));
                        }
                    }
                } else {
                    cute::clear(D(_, m, _));
                }
            } else {  // combine the mn predicate with the k predicate
                #pragma unroll
                for (int k = 0; k < size<2>(S); ++k) {
                    cute::copy(copy_atom.with(predicate_mn && (Is_even_K || predicate_K(k))), S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Byte permute and shuffle to match register layout of
// (FP8 downcasted) accumulator of GEMM-I to FP8 operand A of GEMM-II.
template <typename Fragment>
CUTLASS_DEVICE void permute_Aregs_fp8(Fragment &frag) {
    // frag has shape ((4, 2, 2), MMA_M, MMA_N), each element is 8 bits
    static_assert(decltype(size<0, 0>(frag))::value == 4);
    static_assert(decltype(size<0, 1>(frag))::value == 2);
    static_assert(decltype(stride<0, 0>(frag))::value == 1);
    static_assert(decltype(stride<0, 1>(frag))::value == 4);
    static_assert(sizeof(typename Fragment::value_type) == 1);

    int quad_idx = threadIdx.x % 4;
    bool lane_03 = quad_idx == 0 || quad_idx == 3;
    int selector_upper = lane_03 ? 0x5410 : 0x1054;
    int selector_lower = lane_03 ? 0x7632 : 0x3276;

    static constexpr int upper_map[4] = {0, 3, 1, 2};
    // static constexpr int lower_map[4] = {1, 2, 0, 3};

    Tensor frag_64b = recast<uint2>(frag);  // ((1, 1, 2), MMA_M, MMA_N)
    #pragma unroll
    for (int i = 0; i < size(frag_64b); ++i) {
        uint32_t upper = frag_64b[i].x;
        uint32_t lower = frag_64b[i].y;
        uint32_t upper0 = lane_03 ? upper : lower;
        uint32_t lower0 = lane_03 ? lower : upper;
        upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[quad_idx], 4);
        // lower0 = __shfl_sync(uint32_t(-1), lower0, lower_map[quad_idx], 4);
        lower0 = __shfl_sync(uint32_t(-1), lower0, upper_map[quad_idx] ^ 1, 4);
        frag_64b[i].x = __byte_perm(upper0, lower0, selector_upper);
        frag_64b[i].y = __byte_perm(upper0, lower0, selector_lower);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Fragment>
CUTLASS_DEVICE void permute_Cregs_fp8(Fragment &frag) {
    // frag has shape ((2, 2, N / 8), MMA_M, MMA_N), each element is 32 bits
    static_assert(decltype(size<0, 0>(frag))::value == 2);
    static_assert(decltype(size<0, 1>(frag))::value == 2);
    static_assert(decltype(size<0, 2>(frag))::value % 2 == 0);
    static_assert(decltype(stride<0, 0>(frag))::value == 1);
    static_assert(sizeof(typename Fragment::value_type) == 4);
    Tensor frag_64b = group_modes<1, 3>(recast<uint2>(frag));  // ((1, 2, N / 8), (MMA_M, MMA_N))
    #pragma unroll
    for (int mi = 0; mi < size<1>(frag_64b); ++mi) {
        #pragma unroll
        for (int i = 0; i < size<0, 2>(frag_64b) / 2; ++i) {
            cutlass::swap(frag_64b(make_coord(_0{}, _1{}, 2 * i), mi), frag_64b(make_coord(_0{}, _0{}, 2 * i + 1), mi));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Fragment>
CUTLASS_DEVICE void permute_output_fp8(Fragment &out) {
    // out has shape ((2, 2, N / 8), MMA_M, MMA_N), each element is 32 bits
    static_assert(decltype(size<0, 0>(out))::value == 2);
    static_assert(decltype(size<0, 1>(out))::value == 2);
    static_assert(decltype(size<0, 2>(out))::value % 2 == 0);
    static_assert(decltype(stride<0, 0>(out))::value == 1);
    static_assert(sizeof(typename Fragment::value_type) == 4);
    Tensor frag = group_modes<1, 3>(out);  // ((2, 2, N / 8), (MMA_M, MMA_N))
    #pragma unroll
    for (int mi = 0; mi < size<1>(frag); ++mi) {
        #pragma unroll
        for (int j = 0; j < size<0, 1>(frag); ++j) {
            #pragma unroll
            for (int i = 0; i < size<0, 2>(frag) / 2; ++i) {
                cutlass::swap(frag(make_coord(_1{}, j, 2 * i), mi), frag(make_coord(_0{}, j, 2 * i + 1), mi));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Fragment>
CUTLASS_DEVICE void permute_output_fp8_Vcolmajor(Fragment &frag) {
    // frag has shape ((2, 2, N / 8), MMA_M, MMA_N), each element is 16 bits
    static_assert(decltype(size<0, 0>(frag))::value == 2);
    static_assert(decltype(size<0, 1>(frag))::value == 2);
    static_assert(decltype(stride<0, 0>(frag))::value == 1);
    static_assert(sizeof(typename Fragment::value_type) == 2 || sizeof(typename Fragment::value_type) == 4);

    int quad_idx = threadIdx.x % 4;
    bool lane_03 = quad_idx == 0 || quad_idx == 3;

    static constexpr int upper_map[4] = {0, 2, 3, 1};
    // static constexpr int lower_map[4] = {2, 0, 1, 3};

    // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(frag); }
    using type2 = std::conditional_t<sizeof(typename Fragment::value_type) == 2, uint32_t, uint64_t>;
    Tensor frag_2 = group_modes<1, 3>(recast<type2>(frag));  // ((1, 2, N / 8), (MMA_M, MMA_N))
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(frag); printf("\n"); print(frag_2); }
    #pragma unroll
    for (int mi = 0; mi < size<1>(frag_2); ++mi) {
        #pragma unroll
        for (int j = 0; j < size<0, 1>(frag_2); ++j) {
            #pragma unroll
            for (int i = 0; i < size<0, 2>(frag_2) / 2; ++i) {
                type2 upper = frag_2(make_coord(_0{}, j, 2 * i), mi);
                type2 lower = frag_2(make_coord(_0{}, j, 2 * i + 1), mi);
                type2 upper0 = lane_03 ? upper : lower;
                type2 lower0 = lane_03 ? lower : upper;
                upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[quad_idx], 4);
                // lower0 = __shfl_sync(uint32_t(-1), lower0, lower_map[quad_idx], 4);
                lower0 = __shfl_sync(uint32_t(-1), lower0, upper_map[quad_idx] ^ 2, 4);
                frag_2(make_coord(_0{}, j, 2 * i), mi) = lane_03 ? upper0 : lower0;
                frag_2(make_coord(_0{}, j, 2 * i + 1), mi) = lane_03 ? lower0 : upper0;
            }
        }
    }
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(frag); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
CUTLASS_DEVICE void apply_softcap(Tensor<Engine, Layout> &tensor, float const softcap){
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
    }
}

template <typename Engine, typename Layout>
CUTLASS_DEVICE auto calculate_dtanh(Tensor<Engine, Layout> &tensor){
    Tensor out = make_fragment_like<float>(tensor);
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        out(i) = 1.f - (tensor(i) * tensor(i));
    }
    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
CUTE_DEVICE T warp_prefix_sum(T val) {
    int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < cutlass::NumThreadsPerWarp; i <<= 1) {
        T partial_sum = __shfl_up_sync(0xffffffff, val, i);
        if (lane >= i) { val += partial_sum; }
    }
    return val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
CUTE_DEVICE T warp_uniform(T a) {
    return __shfl_sync(0xffffffff, a, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_DEVICE
int canonical_warp_group_idx_nosync() {
    return threadIdx.x / cutlass::NumThreadsPerWarpGroup;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace flash
