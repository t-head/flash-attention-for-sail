// Copyright (c) 2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
#pragma once
#include "cute/atom/mma_traits_ppu0010.hpp"
namespace cute {
struct QsaSM80_8x16x16_F32BF16BF16F32_TN {
    using DRegisters = float[4];
    using ARegisters = uint32_t[2];
    using BRegisters = uint32_t[4];
    using CRegisters = float[4];
    CUTE_HOST_DEVICE static void fma(
        float &d0, float &d1, float &d2, float &d3,
        uint32_t const& a0, uint32_t const& a1,
        uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
        float const& c0, float const& c1, float const& c2, float const& c3) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
        asm volatile("ppu.tc01.mma.sync.aligned.m8n16k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(b2), "r"(b3),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
        CUTE_INVALID_CONTROL_PATH("QSA M8 requires PPU0010");
#endif
    }
};
template<> struct MMA_Traits<QsaSM80_8x16x16_F32BF16BF16F32_TN> {
    using ValTypeD = float; using ValTypeC = float;
    using ValTypeA = bfloat16_t; using ValTypeB = bfloat16_t;
    using Shape_MNK = Shape<_8,_16,_16>;
    using ThrID = Layout<_32>;
    using ALayout = Layout<Shape<Shape<_4,_8>,Shape<_2,_2>>,
                           Stride<Stride<_16,_1>,Stride<_8,_64>>>;
    using BLayout = typename MMA_Traits<PPU0010_16x16x16_F32BF16BF16F32_TN>::BLayout;
    using CLayout = Layout<Shape<Shape<_4,_8>,Shape<_4,_1>>,Stride<Stride<_8,_1>,Stride<_32,_8>>>;
};

template<typename To, typename Engine, typename Layout>
CUTE_DEVICE __forceinline__ auto qsa_convert_m8(Tensor<Engine,Layout> const& scores) {
    static_assert(decltype(size(scores))::value == 4);
    cutlass::NumericArrayConverter<To,float,4> convert;
    auto values = convert(*reinterpret_cast<cutlass::Array<float,4> const*>(scores.data()));
    auto out = make_tensor_like<To>(scores);
    auto const* packed = reinterpret_cast<uint32_t const*>(values.data());
    auto* dst = reinterpret_cast<uint32_t*>(out.data());
    // Convert the four M8 C values into two BF16 MMA A registers.
    int lane = threadIdx.x & 31;
    int src = (lane & ~3) + 2 * (lane & 1);
    int shift = (lane & 2) * 8;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        uint32_t a = __shfl_sync(0xffffffff, packed[j], src);
        uint32_t b = __shfl_sync(0xffffffff, packed[j], src + 1);
        dst[j] = ((a >> shift) & 0xffff) | ((b >> shift) << 16);
    }
    return out;
}
}
