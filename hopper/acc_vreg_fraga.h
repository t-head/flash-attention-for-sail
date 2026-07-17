/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

#pragma once
#include "cutlass/array.h"
static CUTLASS_DEVICE int32_t get_lane_id() {
  return threadIdx.x % 32;
}
static CUTLASS_DEVICE int32_t get_warp_id() {
  return threadIdx.y;
}
template <typename T, typename S, int N> struct NumericArrayConverterPPU {
  using ElementCompute = T;
  bool is_source_needed() { return false; }
};

#define CONVERTER_BASE_UNIT_32 32
#define CONVERTER_BASE_UNIT_8 8
#define SHFL_V1

template <typename T> struct NumericArrayConverterPPU<T, float, CONVERTER_BASE_UNIT_32> {
  using result_type = cutlass::Array<T, CONVERTER_BASE_UNIT_32>;
  using source_type = cutlass::Array<float, CONVERTER_BASE_UNIT_32>;
  using ElementCompute = T;
  CUTLASS_DEVICE
  result_type convert(source_type const &src_const) {
    result_type result;
    source_type src;
    // copy src to src_copy
    src = src_const;
#ifdef SHFL_V0
    // reorder data layout in Acc fragment
    // first exchange T1(R0)<->T0(R1), T3(R0)<->T2(R1)
    // second exchange T1(R0)<->T2(R0), T1(R1)<->T2(R1)

    int lane_id = get_lane_id();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i += 2) {
      float data = lane_id & 0x1 ? src[i] : src[i + 1];
      float res = __shfl_xor_sync(0xFFFFFFFF, data, 1, 4);
      if (lane_id & 0x1) {
        src[i] = res;
      } else {
        src[i + 1] = res;
      }
      double* tmp = reinterpret_cast<double*>(src.data() + i);
      double d = __shfl_xor_sync(0xffffffff, tmp[0], 3, 4);
      // double d = __shfl_xor_sync(0x66666666, tmp[0], 3, 4);
      if (lane_id % 4 == 1 || lane_id % 4 == 2) {
        tmp[0] = d;
      }
    }
#endif
    // convert and pack to fp16
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i++) {
      result[i] = (T)src[i];
    }
#ifdef SHFL_V1
    result_type final_result;
    // create weight matrix
    // bool high_phase = threadIdx.x >> 4;
    // uint32_t inner_idx = threadIdx.x & 0xf;
    bool high_phase = get_lane_id() >> 4;
    uint32_t inner_idx = get_lane_id() & 0xf;
    uint32_t row = inner_idx >> 2;
    uint32_t col  = inner_idx & 0x3;
    bool value1 = (!high_phase) && (row==col);
    bool value2 = (high_phase) && (row==col);
    uint32_t vreg0 = 0;
    uint32_t vreg1  = 0;
    uint32_t vreg2 = 0;
    uint32_t vreg3 = 0;
    vreg0 = value1 ? (std::is_same<cutlass::half_t, T>::value ? 0x3c00 : 0x3f80) : vreg0;
    vreg0 = value2 ? (std::is_same<cutlass::half_t, T>::value ? 0x3c000000 : 0x3f800000) : vreg0;
    // vreg0 = value1 ? 0x3c00 : vreg0;
    // vreg0 = value2 ?  0x3c000000 : vreg0;
    vreg3 = vreg0;
    uint32_t weight[4] = {0};
    weight[0] = vreg0;
    weight[1] = vreg1;
    weight[2] = vreg2;
    weight[3] = vreg3;
    uint32_t zero_accum[4] = {0};
    // do matmul

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i += 8) {
      float* d = reinterpret_cast<float*>(final_result.data() + i);
      float* a = reinterpret_cast<float*>(result.data() + i);
      float* b = reinterpret_cast<float*>(&weight[0]);
      float* c = reinterpret_cast<float*>(&zero_accum[0]);

      if (std::is_same<cutlass::half_t, T>::value) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"    
          "{%12,%13,%14,%15};\n"
          : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
          : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
        );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"    
          "{%12,%13,%14,%15};\n"
          : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
          : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
        );
#endif
      } else if(std::is_same<cutlass::bfloat16_t, T>::value) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"    
          "{%12,%13,%14,%15};\n"
          : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
          : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"    
          "{%12,%13,%14,%15};\n"
          : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
          : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          );
#endif
      } else {
        assert("not support layout convert type\n");
      }
    }
    return final_result;
#else
    return result;
#endif
  }


  CUTLASS_DEVICE
  bool is_source_needed() { return false; }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) { return convert(s); }
};

template <typename T> struct NumericArrayConverterPPU<T, float, CONVERTER_BASE_UNIT_8> {
  using result_type = cutlass::Array<T, CONVERTER_BASE_UNIT_8>;
  using source_type = cutlass::Array<float, CONVERTER_BASE_UNIT_8>;
  using ElementCompute = T;
  CUTLASS_DEVICE
  result_type convert(source_type const &src_const) {
    result_type result;
    source_type src;
    // copy src to src_copy
    src = src_const;
#ifdef SHFL_V0
    // reorder data layout in Acc fragment
    // first exchange T1(R0)<->T0(R1), T3(R0)<->T2(R1)
    // second exchange T1(R0)<->T2(R0), T1(R1)<->T2(R1)

    int lane_id = get_lane_id();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i += 2) {
      float data = lane_id & 0x1 ? src[i] : src[i + 1];
      float res = __shfl_xor_sync(0xFFFFFFFF, data, 1, 4);
      if (lane_id & 0x1) {
        src[i] = res;
      } else {
        src[i + 1] = res;
      }
      double* tmp = reinterpret_cast<double*>(src.data() + i);
      double d = __shfl_xor_sync(0xffffffff, tmp[0], 3, 4);
      // double d = __shfl_xor_sync(0x66666666, tmp[0], 3, 4);
      if (lane_id % 4 == 1 || lane_id % 4 == 2) {
        tmp[0] = d;
      }
    }
#endif
    // convert and pack to fp16
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i++) {
      result[i] = (T)src[i];
    }
#ifdef SHFL_V1
    result_type final_result;
    // create weight matrix
    // bool high_phase = threadIdx.x >> 4;
    // uint32_t inner_idx = threadIdx.x & 0xf;
    bool high_phase = get_lane_id() >> 4;
    uint32_t inner_idx = get_lane_id() & 0xf;
    uint32_t row = inner_idx >> 2;
    uint32_t col  = inner_idx & 0x3;
    bool value1 = (!high_phase) && (row==col);
    bool value2 = (high_phase) && (row==col);
    uint32_t vreg0 = 0;
    uint32_t vreg1  = 0;
    uint32_t vreg2 = 0;
    uint32_t vreg3 = 0;
    vreg0 = value1 ? (std::is_same<cutlass::half_t, T>::value ? 0x3c00 : 0x3f80) : vreg0;
    vreg0 = value2 ? (std::is_same<cutlass::half_t, T>::value ? 0x3c000000 : 0x3f800000) : vreg0;
    // vreg0 = value1 ? 0x3c00 : vreg0;
    // vreg0 = value2 ?  0x3c000000 : vreg0;
    vreg3 = vreg0;
    uint32_t weight[4] = {0};
    weight[0] = vreg0;
    weight[1] = vreg1;
    weight[2] = vreg2;
    weight[3] = vreg3;
    uint32_t zero_accum[4] = {0};
    // do matmul

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i += 8) {
      float* d = reinterpret_cast<float*>(final_result.data() + i);
      float* a = reinterpret_cast<float*>(result.data() + i);
      float* b = reinterpret_cast<float*>(&weight[0]);
      float* c = reinterpret_cast<float*>(&zero_accum[0]);

      if (std::is_same<cutlass::half_t, T>::value) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"    
          "{%12,%13,%14,%15};\n"
          : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
          : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
        );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"    
          "{%12,%13,%14,%15};\n"
          : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
          : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
        );
#endif
      } else if(std::is_same<cutlass::bfloat16_t, T>::value) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
asm volatile(
    "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"    
          "{%12,%13,%14,%15};\n"
          : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
          : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          );
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
asm volatile(
    "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"    
          "{%12,%13,%14,%15};\n"
          : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
          : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          );
#endif
      } else {
        assert("not support layout convert type\n");
      }
    }

    return final_result;
#else
    return result;
#endif
  }

  CUTLASS_DEVICE
  bool is_source_needed() { return false; }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) { return convert(s); }
};

template <typename T, int N> struct NumericArrayConverterPPU<T, float, N> {
  using result_type = cutlass::Array<T, N>;
  using source_type = cutlass::Array<float, N>;
  using ElementCompute = T;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    using scalar_result_type = typename result_type::Element;
    using scalar_source_type = typename source_type::Element;

    result_type result;
    if (N % CONVERTER_BASE_UNIT_32 == 0) {
      constexpr int VEC_WIDTH = CONVERTER_BASE_UNIT_32;

      NumericArrayConverterPPU<scalar_result_type, scalar_source_type, VEC_WIDTH> convert_vector_;
      using vec_result = cutlass::Array<scalar_result_type, VEC_WIDTH>;
      using vec_source = cutlass::Array<scalar_source_type, VEC_WIDTH>;
      vec_result *result_ptr = reinterpret_cast<vec_result *>(&result);
      vec_source const *source_ptr = reinterpret_cast<vec_source const *>(&source);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N / VEC_WIDTH; ++i) {
        result_ptr[i] = convert_vector_.convert(source_ptr[i]);
      }
    } else {
      constexpr int VEC_WIDTH = CONVERTER_BASE_UNIT_8;

      NumericArrayConverterPPU<scalar_result_type, scalar_source_type, VEC_WIDTH> convert_vector_;
      using vec_result = cutlass::Array<scalar_result_type, VEC_WIDTH>;
      using vec_source = cutlass::Array<scalar_source_type, VEC_WIDTH>;
      vec_result *result_ptr = reinterpret_cast<vec_result *>(&result);
      vec_source const *source_ptr = reinterpret_cast<vec_source const *>(&source);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N / VEC_WIDTH; ++i) {
        result_ptr[i] = convert_vector_.convert(source_ptr[i]);
      }
    }

    return result;
  }
  CUTLASS_DEVICE
  bool is_source_needed() { return false; }
  CUTLASS_DEVICE
  result_type operator()(source_type const &s) { return convert(s); }
};
