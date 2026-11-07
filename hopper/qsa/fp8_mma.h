// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Copyright (c) 2024, Tri Dao.
// PPU QSA FP8 Tensor Cell operations with explicit FP32 accumulation.
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

typedef unsigned qsa_v4uint __attribute__((ext_vector_type(4)));

namespace flash {
// Match FA's V-direct pi_k/pi_n mapping using hardware b16 matrix loads
// and lane-local byte permutations, without the software b8 transpose.
// P uses permute_Cregs_fp8; the epilogue restores the permuted columns.
template <int RowStride, class Element>
CUTLASS_DEVICE qsa_v4uint qsa_fp8_load_v_direct(Element const* base) {
    static_assert(sizeof(Element) == 1 && RowStride % 16 == 0);
    auto const* row = base + (threadIdx.x % 32) * RowStride;
    uint32_t r[4];
    cute::PPU_U16x8_LDSM_T::copy(*reinterpret_cast<cute::uint128_t const*>(row),
                               r[0], r[1], r[2], r[3]);
    return {__byte_perm(r[0], r[1], 0x6420u),
            __byte_perm(r[2], r[3], 0x6420u),
            __byte_perm(r[0], r[1], 0x7531u),
            __byte_perm(r[2], r[3], 0x7531u)};
}

template <class A, class B, class C>
CUTLASS_DEVICE void qsa_fp8_mma_fp32_add(A const &a, B const &b, C &c,
                                         unsigned operand_dependency) {
    auto aa = cute::recast<uint32_t>(a);
    auto bb = cute::recast<uint32_t>(b);
    static_assert(decltype(cute::size(aa))::value == 4);
    static_assert(decltype(cute::size(bb))::value == 4);
    static_assert(decltype(cute::size(c))::value == 8);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    asm volatile(
        "{\n"
        ".reg .f32 x0,x1,x2,x3,x4,x5,x6,x7,z;\n"
        "ppu.mov.f32 z, 0f00000000;\n"
        "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.f32.e4m3.e4m3.f32 {x0,x1,x2,x3,x4,x5,x6,x7}, "
        "{%8,%9,%10,%11}, {%12,%13,%14,%15}, {z,z,z,z,z,z,z,z};\n"
        "ppu.add.rtte.ftz.f32 %0, %0, x0;\n"
        "ppu.add.rtte.ftz.f32 %1, %1, x1;\n"
        "ppu.add.rtte.ftz.f32 %2, %2, x2;\n"
        "ppu.add.rtte.ftz.f32 %3, %3, x3;\n"
        "ppu.add.rtte.ftz.f32 %4, %4, x4;\n"
        "ppu.add.rtte.ftz.f32 %5, %5, x5;\n"
        "ppu.add.rtte.ftz.f32 %6, %6, x6;\n"
        "ppu.add.rtte.ftz.f32 %7, %7, x7;\n"
        "}\n"
        : "+r"(c(0)), "+r"(c(1)), "+r"(c(2)), "+r"(c(3)), "+r"(c(4)), "+r"(c(5)), "+r"(c(6)),
          "+r"(c(7))
        : "r"(aa(0) ^ operand_dependency), "r"(aa(1)), "r"(aa(2)), "r"(aa(3)), "r"(bb(0)),
          "r"(bb(1)), "r"(bb(2)), "r"(bb(3)));
#else
    assert(false && "QSA exact cross-tile add requires PPU0015");
#endif
}

template <class A, class B, class C>
CUTLASS_DEVICE void qsa_fp8_mma_fp32_add_sixhead(A const &a, B const &b, C &c,
                                                 unsigned operand_dependency) {
    auto aa = cute::recast<uint32_t>(a);
    auto bb = cute::recast<uint32_t>(b);
    static_assert(decltype(cute::size(aa))::value == 4);
    static_assert(decltype(cute::size(bb))::value == 4);
    static_assert(decltype(cute::size(c))::value == 8);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    asm volatile("{\n"
                 ".reg .f32 x0,x1,x2,x3,x4,x5,x6,x7,z;\n"
                 "ppu.mov.f32 z, 0f00000000;\n"
                 "ppu.tc02.mma.sync.aligned.m16n16k32.row.col.f32.e4m3.e4m3.f32 "
                 "{x0,x1,x2,x3,x4,x5,x6,x7}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {z,z,z,z,z,z,z,z};\n"
                 "ppu.add.rtte.ftz.f32 %0, %0, x0;\n"
                 "ppu.add.rtte.ftz.f32 %1, %1, x1;\n"
                 "ppu.add.rtte.ftz.f32 %2, %2, x4;\n"
                 "ppu.add.rtte.ftz.f32 %3, %3, x5;\n"
                 "}\n"
                 : "+r"(c(0)), "+r"(c(1)), "+r"(c(4)), "+r"(c(5))
                 : "r"(aa(0) ^ operand_dependency), "r"(aa(1)), "r"(aa(2)), "r"(aa(3)), "r"(bb(0)),
                   "r"(bb(1)), "r"(bb(2)), "r"(bb(3)));
    // C-layout components 2,3,6,7 are rows 8..15; this guarded path has six heads.
    c(2) = 0.f;
    c(3) = 0.f;
    c(6) = 0.f;
    c(7) = 0.f;
#else
    assert(false && "QSA exact cross-tile add requires PPU0015");
#endif
}

} // namespace flash
