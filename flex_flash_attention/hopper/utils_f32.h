// ISOLATED fp32 kernel-set overlay on top of the frozen 16-bit utils.h.
// Only the fp32-specific helper lives here (renamed to avoid clashing
// with the 16-bit copy); everything else is shared verbatim.
#pragma once

#include "utils.h"

namespace flash {

#ifdef USE_PPU
// ISOLATED fp32 (TF32) kernel set: build the A-operand fragment of the PV
// gemm from the QK accumulator. For the PPU0015 tf32 m16n16k8 atom the C
// thread->tile mapping (row = t1 + 8*v1, col = 2*t0 + v0 + 8*v2, with
// t0 = lane%4, t1 = lane/4) differs from the A mapping (row = t1 + 8*u0,
// col = t0 + 4*u1), so the acc C-regs cannot be re-viewed as A-regs.
// k-slice j (cols 8j..8j+7) lives in atom j/2 (v2 = j&1); the value a lane
// needs for (u0, u1) sits in the lane whose t0 = ((lane&3)>>1) + 2*u1
// (same t1, so same row) at register v0 = lane&1. That register parity
// differs from the source lane's own, so both v0 variants are shuffled and
// selected by lane parity. fp16/bf16 builds never reach this function
// (their A/C layouts are identical, plain reinterpret suffices).
template<typename TiledMma, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_acc_to_tf32_Aregs_f32(Tensor<Engine, Layout> const &acc) {
    static_assert(std::is_same_v<typename Engine::value_type, float>);
    constexpr int mma_n = decltype(size<2>(acc.layout()))::value;  // 16-col atoms across kBlockN
    constexpr int k_iter = 2 * mma_n;                              // k=8 slices
    Tensor a = make_fragment_like<float>(
        make_layout(make_layout(make_shape(_2{}, _2{}), make_stride(_1{}, _2{})),
                    get<1>(acc.layout()), make_layout(make_shape(Int<k_iter>{}))));
    int const lane = threadIdx.x % 32;
    #pragma unroll
    for (int j = 0; j < k_iter; ++j) {
        int const n_atom = j >> 1;   // 16-col atom holding slice j
        int const v2 = j & 1;        // low/high 8-col half inside the atom
        #pragma unroll
        for (int u0 = 0; u0 < 2; ++u0) {
            #pragma unroll
            for (int u1 = 0; u1 < 2; ++u1) {
                int const src_lane = (lane & ~3) | (((lane & 3) >> 1) + 2 * u1);
                float const s0 = __shfl_sync(0xffffffff,
                    acc(make_coord(0, u0, v2), 0, n_atom), src_lane);
                float const s1 = __shfl_sync(0xffffffff,
                    acc(make_coord(1, u0, v2), 0, n_atom), src_lane);
                a(make_coord(u0, u1), 0, j) = (lane & 1) ? s1 : s0;
            }
        }
    }
    return a;
}
#endif

}  // namespace flash
