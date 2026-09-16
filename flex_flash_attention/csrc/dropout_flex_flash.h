/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 ******************************************************************************/

// Dropout for the flex flash attention kernels.
//
// Design (mirrors FA2 semantics, adapted to slice-based dispatch):
//   - Philox4x32 (7 rounds, identical constants to csrc/flash_attn/src/
//     philox.cuh) is keyed on the GLOBAL score coordinates:
//         subsequence = (col << 32) | row,   offset = per-(batch, head) base
//     Because the key is coordinate-based (not iteration-order-based), the
//     mask is identical no matter how slices/zones reorder the K-block
//     traversal, and fwd/bwd replay the same stream as long as they use the
//     same (seed, offset) pair.
//   - Dropout is applied to the fp32 score accumulator AFTER online softmax
//     (row_sum intentionally accumulates the UN-dropped P, matching FA2) and
//     BEFORE the P->Element conversion / PV gemm.  Dropped elements are
//     zeroed; the epilogue output is scaled by rp_dropout = 1 / p_keep.
//   - The rng_state tensor ([0]=seed, [1]=offset) is populated host-side by
//     the API layer and returned to the caller so bwd can replay (eager);
//     under CUDA graph capture it is a DEVICE tensor populated by the fwd
//     kernel itself (see DropoutArbArgs::rng_state_dev below).

#pragma once

#include <cute/tensor.hpp>

#include "../hopper/utils.h"    // flash::convert_layout_acc_rowcol

namespace flash {

using namespace cute;

// ─── Shared dropout kernel arguments (fwd & bwd) ──────────────────────────
// Passed by value (all host-side scalars); rng_offset_base already includes
// the API-layer rng_state[1], the kernel only adds the per-(batch, head)
// term.  Zero-initialized ⇒ Is_dropout=false builds never touch these.
struct DropoutArbArgs {
    unsigned long long rng_seed = 0;
    unsigned long long rng_offset_base = 0;
    float rp_dropout = 1.f;
    uint8_t p_keep_in_uint8_t = 255;
    int num_heads = 1;   // per-(batch, head) offset stride

    // ── CUDA graph capture mode ───────────────────────────────────────────
    // Kernel launches baked into a graph freeze all by-value arguments, so
    // the Philox (seed, offset) pair cannot be drawn host-side at capture
    // time (it would also replay the SAME dropout mask forever).  Instead
    // the API layer hands the kernel device POINTERS into the generator's
    // extragraph tensors (owned by CUDAGeneratorImpl, alive across
    // replays); PyTorch refreshes them before every graph.replay(), so each
    // replay draws a fresh stream exactly like eager.  philox_seed_ptr ==
    // nullptr selects the eager by-value fields above.
    const unsigned long long* philox_seed_ptr = nullptr;
    const unsigned long long* philox_offset_ptr = nullptr;
    // Per-graph counter assigned at capture (philox_cuda_state's
    // offset_intragraph), added to *philox_offset_ptr.
    unsigned long long intragraph_offset = 0;
    // fwd only: device tensor[2] where the fwd kernel publishes the
    // (seed, offset_base) pair it used, so a CAPTURED bwd graph replays the
    // same stream (its params.rng_state points here and is read on device).
    unsigned long long* rng_state_dev = nullptr;
};

// Resolve the effective (seed, offset_base) at kernel start: device-pointer
// indirection in graph mode, baked by-value scalars in eager.  Call once per
// tile/thread and feed DropoutFlexFlash; both fwd and bwd must resolve the
// same way to replay the identical coordinate-keyed stream.
__device__ __forceinline__ void arb_resolve_rng_stream(
    DropoutArbArgs const& d,
    unsigned long long& seed, unsigned long long& offset_base) {
    if (d.philox_seed_ptr != nullptr) {
        seed = *d.philox_seed_ptr;
        offset_base = *d.philox_offset_ptr + d.intragraph_offset;
    } else {
        seed = d.rng_seed;
        offset_base = d.rng_offset_base;
    }
}

// ─── Self-contained Philox (bitwise identical to FA2 philox.cuh) ───────────

struct ArbPhilox_ull2 {
    unsigned long long x;
    unsigned long long y;
};

__forceinline__ __device__ uint2 arb_philox_mulhilo32(unsigned int a, unsigned int b) {
    uint2 *res;
    unsigned long long tmp;
    asm ("mul.wide.u32 %0, %1, %2;\n\t"
          : "=l"(tmp)
          : "r"(a), "r"(b));
    res = (uint2*)(&tmp);
    return *res;
}

__forceinline__ __device__ uint4 arb_philox_single_round(const uint4 ctr, const uint2 key) {
    constexpr unsigned long kPhiloxSA = 0xD2511F53;
    constexpr unsigned long kPhiloxSB = 0xCD9E8D57;
    uint2 res0 = arb_philox_mulhilo32(kPhiloxSA, ctr.x);
    uint2 res1 = arb_philox_mulhilo32(kPhiloxSB, ctr.z);
    uint4 ret = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
    return ret;
}

__forceinline__ __device__ uint4 arb_philox(unsigned long long seed,
                                            unsigned long long subsequence,
                                            unsigned long long offset) {
    constexpr unsigned long kPhilox10A = 0x9E3779B9;
    constexpr unsigned long kPhilox10B = 0xBB67AE85;
    uint2 key = reinterpret_cast<uint2&>(seed);
    uint4 counter;
    ArbPhilox_ull2 *tmp = reinterpret_cast<ArbPhilox_ull2*>(&counter);
    tmp->x = offset;
    tmp->y = subsequence;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        counter = arb_philox_single_round(counter, key);
        key.x += (kPhilox10A);
        key.y += (kPhilox10B);
    }
    return arb_philox_single_round(counter, key);
}

// ─── DropoutApplier ─────────────────────────────────────────────────────────

template <int kBlockM, int kBlockN, typename TiledMma, bool Transposed = false>
struct DropoutFlexFlash {

    unsigned long long seed, offset;
    uint8_t p_keep_in_uint8_t;

    CUTLASS_DEVICE DropoutFlexFlash(unsigned long long seed_, unsigned long long offset_,
                                    uint8_t p_keep_in_uint8_t_)
        : seed(seed_), offset(offset_), p_keep_in_uint8_t(p_keep_in_uint8_t_) {}

    // Zero out dropped elements of the fp32 score accumulator tSrS.
    // m_block/n_block are GLOBAL block indices; the per-element key is the
    // global (row, col) of the score matrix S (row = q position, col = k
    // position regardless of Transposed: convert_layout_acc_rowcol already
    // swaps the transposed bwd fragment back to S-space coordinates).
    //
    // Perf (FA2-style batching): in the rowcol fragment, cols 2j and 2j+1
    // of a thread's row are ALWAYS consecutive (m16n8 acc pairing), so two
    // ADJACENT pairs (n = 2j and n = 2j+1) map to four consecutive global
    // columns {c, c+1, c+2, c+3} and one Philox call covers all four —
    // 4x fewer Philox calls than a per-element stream.  Keying stays
    // COORDINATE-based (not fragment-position based): the drop decision of a
    // cell depends only on its global (row, col), so fwd and bwd replay
    // identical masks even though their TiledMma partitions differ, and the
    // grouping only needs to partition coordinates deterministically (which
    // it does — each thread's rowcol cells are disjoint and every group is
    // keyed on its own first coordinate).
    // Each byte is compared against the keep threshold with an ordinary
    // unsigned compare (branch-free select) — portable across the toolchain
    // and bitwise-deterministic.
    template <typename Engine, typename Layout>
    CUTLASS_DEVICE void apply(
        Tensor<Engine, Layout>& tSrS,
        int m_block, int n_block, int thread_idx) const
    {
        static_assert(Layout::rank == 3, "Only support 3D tensor");
    
        auto thread_mma = TiledMma{}.get_slice(thread_idx);
    
        // Identity tensor → block-local (row, col), same mechanism as
        // MaskFlexFlash (see the NOTE there: per-thread coordinates only).
        Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thread_mma.partition_C(cS);
    
        Tensor tSrS_rowcol = make_tensor(
            tSrS.data(),
            flash::convert_layout_acc_rowcol<Transposed>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(
            tScS.data(),
            flash::convert_layout_acc_rowcol<Transposed>(tScS.layout()));
    
        unsigned const p_keep_u = static_cast<unsigned>(p_keep_in_uint8_t);
        static_assert(decltype(size<1>(tSrS_rowcol))::value % 4 == 0,
                      "dropout quad batching needs cols per row divisible by 4");
    
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row = int(get<0>(tScS_rowcol(m, 0))) + m_block * kBlockM;
            #pragma unroll
            for (int n = 0; n < size<1>(tSrS_rowcol); n += 4) {
                // First col of the quad == first col of pair (m, n): the two
                // pairs cover cols {col, col+1, col+2, col+3} (m16n8 pairing).
                int const col = int(get<1>(tScS_rowcol(m, n))) + n_block * kBlockN;
                unsigned long long const subseq =
                    (static_cast<unsigned long long>(static_cast<unsigned>(col)) << 32)
                    | static_cast<unsigned>(row);
                uint4 const rnd = arb_philox(seed, subseq, offset);
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    unsigned const rb = (rnd.x >> (8 * b)) & 0xFFu;
                    // keep when rb <= p_keep (FA2 threshold semantics);
                    // dropped entries are zeroed on the fp32 accumulator.
                    tSrS_rowcol(m, n + b) =
                        rb <= p_keep_u ? tSrS_rowcol(m, n + b) : 0.f;
                }
            }
        }
    }
};

} // namespace flash
