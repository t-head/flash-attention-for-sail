/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "cute/tensor.hpp"

#include "seqlen.h"
#include "block.h"
#include "mask.h"
#include "softmax.h"
#include "utils.h"

#define DEBUG_TID 0
#define DEBUG_MODE 0
// CVT_LAYOUT comes from utils.h so that this file and epilogue_bwd.hpp can never disagree.

// ============================================================================
// Local override of PPU0015_TSM_LD_SWZL_CVT for rapid iteration without
// modifying the cutlass3 submodule. Placed in namespace cute so that
// SmemCopyOp / Copy_Atom<> can reference it directly.
// Only compiled when PPU AIU headers are available.
// ============================================================================
#if defined(USE_PPU) && USE_AIU
namespace cute {

// Forward declaration (primary template)
template <typename Element, int CUBE_H, int CUBE_W, int BlockH, int BlockW,
          bool Swap, bool Trans, int InstNum, bool Cvt,
          int LBO, int SBO, bool SHUFFLING_GAIT = false>
struct PPU0015_TSM_LD_SWZL_CVT_LOCAL;

// Trans=false specialization
template <typename Element, int CUBE_H, int CUBE_W, int BlockH, int BlockW,
          bool Swap, int InstNum, bool Cvt, int LBO, int SBO, bool SHUFFLING_GAIT>
struct PPU0015_TSM_LD_SWZL_CVT_LOCAL<Element, CUBE_H, CUBE_W, BlockH, BlockW,
                                      Swap, false, InstNum, Cvt, LBO, SBO, SHUFFLING_GAIT> {
  static_assert(sizeof(Element) == 2 && CUBE_H == 8 && CUBE_W == 64);
  static_assert(BlockH % CUBE_H == 0 && BlockW % (2*CUBE_W) == 0);
  static_assert(Cvt);
  static constexpr int swzl_mode = 0;
  CUTE_HOST_DEVICE static void
  copy(void *frag_ptr, void *smem_base, int coord_w, int coord_h, int cube_in_stage = 0, int stage = 0)
  {
#if defined(__CUDA_ARCH__) && ACOMPUTE_VERSION >= 10500
    Element *stage_base = reinterpret_cast<Element*>(smem_base);
    const int lbo = SHUFFLING_GAIT ? (cube_in_stage & 1 ? LBO-1 : LBO+1) : LBO;
    {
      stage_base += CUBE_H * CUBE_W * InstNum * stage;
      stage_base += (cube_in_stage >> 1) * (CUBE_H * CUBE_W * (InstNum >> 1));
      stage_base += (cube_in_stage & 1) << 3;
      stage_base += coord_h * BlockW + coord_w;
    }
    int tsm_add = reinterpret_cast<uintptr_t>(stage_base) / 16;
    int *vreg = reinterpret_cast<int *>(frag_ptr);
#if DEBUG_MODE
    if (cute::thread(DEBUG_TID)) {
      printf("[TSM_LD_SWZL_CVT_LOCAL Trans=0] frag=%p smem_base=%p coord_w=%d coord_h=%d cube_in_stage=%d stage=%d | tsm_add=0x%x lbo=%d SBO=%d swzl_mode=%d | tpl: LBO=%d InstNum=%d BlockH=%d BlockW=%d SHUFFLING_GAIT=%d\n",
             frag_ptr, smem_base, coord_w, coord_h, cube_in_stage, stage,
             (unsigned)tsm_add, lbo, SBO, swzl_mode,
             LBO, InstNum, BlockH, BlockW, (int)SHUFFLING_GAIT);
    }
#endif
    PPU0015_TSM_LD_SWZL_IMPL<Element, false, false>()(vreg, tsm_add, lbo, SBO, swzl_mode);
#else
    CUTE_INVALID_CONTROL_PATH("Support for TSM_LD_SWZL has not been enabled");
#endif
  }
};

// Trans=true specialization
template <typename Element, int CUBE_H, int CUBE_W, int BlockH, int BlockW,
          bool Swap, int InstNum, bool Cvt, int LBO, int SBO, bool SHUFFLING_GAIT>
struct PPU0015_TSM_LD_SWZL_CVT_LOCAL<Element, CUBE_H, CUBE_W, BlockH, BlockW,
                                      Swap, true, InstNum, Cvt, LBO, SBO, SHUFFLING_GAIT> {
  static_assert(sizeof(Element) == 2 && CUBE_H == 8 && CUBE_W == 64);
  static_assert(BlockH % CUBE_H == 0);
  static constexpr int swzl_mode = 0;
  CUTE_HOST_DEVICE static void
  copy(void *frag_ptr, void *smem_base, int coord_h, int coord_w, int cube_in_stage = 0, int stage = 0)
  {
#if defined(__CUDA_ARCH__) && ACOMPUTE_VERSION >= 10500
    Element *stage_base = reinterpret_cast<Element*>(smem_base);
    const int sbo = SHUFFLING_GAIT ? (coord_h >= 64 ? SBO - (CUBE_H * BlockW * (int)sizeof(Element) >> 4) : SBO + (CUBE_H * BlockW * (int)sizeof(Element) >> 4)) : SBO;
    if constexpr (Cvt) {
      {
        if constexpr (BlockH == 128) {
          stage_base += CUBE_H * CUBE_W * (stage * InstNum);
          stage_base += (cube_in_stage >> 1) * (CUBE_H * CUBE_W * (InstNum >> 1));
          stage_base += (cube_in_stage & 1) << 3;
          stage_base += ((coord_h & 63) + ((coord_h >> 3) & 8)) * BlockW + coord_w;
        } else {
          stage_base += CUBE_H * CUBE_W * InstNum * stage;
          stage_base += (cube_in_stage >> 1) * (CUBE_H * CUBE_W * (InstNum >> 1));
          stage_base += (cube_in_stage & 1) << 3;
          stage_base += coord_h * BlockW + coord_w;
        }
      }
    } else {
      stage_base += CUBE_H * CUBE_W * (cube_in_stage + stage * InstNum);
      stage_base += coord_h * BlockW + coord_w;
    }
    int tsm_add = reinterpret_cast<uintptr_t>(stage_base) / 16;
    int *vreg = reinterpret_cast<int *>(frag_ptr);
#if DEBUG_MODE
    if (cute::thread(DEBUG_TID)) {
      printf("[TSM_LD_SWZL_CVT_LOCAL Trans=1] frag=%p smem_base=%p coord_w=%d coord_h=%d cube_in_stage=%d stage=%d | tsm_add=0x%x LBO=%d sbo=%d swzl_mode=%d | tpl: LBO=%d SBO=%d InstNum=%d BlockH=%d BlockW=%d Cvt=%d SHUFFLING_GAIT=%d\n",
             frag_ptr, smem_base, coord_w, coord_h, cube_in_stage, stage,
             (unsigned)tsm_add, LBO, sbo, swzl_mode,
             LBO, SBO, InstNum, BlockH, BlockW, (int)Cvt, (int)SHUFFLING_GAIT);
    }
#endif
    PPU0015_TSM_LD_SWZL_IMPL<Element, true, false>()(vreg, tsm_add, LBO, sbo, swzl_mode);
#else
    CUTE_INVALID_CONTROL_PATH("Support for TSM_LD_SWZL has not been enabled");
#endif
  }
};

// Copy_Traits specialization for PPU0015_TSM_LD_SWZL_CVT_LOCAL
template <typename Element, int CUBE_H, int CUBE_W, int BlockH, int BlockW,
          bool Swap, bool Trans, int InstNum, bool last_blk_cvt,
          int LBO, int SBO, bool SHUFFLING_GAIT>
struct Copy_Traits<PPU0015_TSM_LD_SWZL_CVT_LOCAL<Element, CUBE_H, CUBE_W, BlockH, BlockW,
                                                  Swap, Trans, InstNum, last_blk_cvt, LBO, SBO, SHUFFLING_GAIT>>
{
  using ThrID = Layout<_32>;
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  using DstLayout = Layout<Shape <_32,Shape <_32,   _4>>,
                           Stride<_32,Stride< _1,_1024>>>;
  using RefLayout = DstLayout;

  void *smem_base_;
  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr
  void
  copy_unpack_(void *dst_ptr, void* src_ptr,
               Coord const& src_coord, seq<Is...>) const
  {
    PPU0015_TSM_LD_SWZL_CVT_LOCAL<Element, CUBE_H, CUBE_W, BlockH, BlockW,
                                   Swap, Trans, InstNum, last_blk_cvt, LBO, SBO, SHUFFLING_GAIT>::copy(dst_ptr, src_ptr, get<Is>(src_coord)...);
  }
  template <class TS, class SLayout,
          class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    if constexpr (is_mix_iterator<typename TS::iterator>::value) {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), src.data().ptr_.get(), src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    } else {
      traits.copy_unpack_(cute::raw_pointer_cast(dst.data()), traits.smem_base_, src.data().coord_, tuple_seq<decltype(src.data().coord_)>{});
    }
  }
};

} // namespace cute (local overrides)
#endif // defined(USE_PPU) && USE_AIU
// ============================================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800) && defined(USE_PPU) && USE_AIU
#define PPU1v0_R2S_SLICE_LAYOUT 1
#else
#define PPU1v0_R2S_SLICE_LAYOUT 0
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 890) && defined(USE_PPU) && USE_AIU
#define PPU1v5_R2S_SLICE_LAYOUT 1
#else
#define PPU1v5_R2S_SLICE_LAYOUT 0
#endif

namespace flash {

using namespace cute;

template <int Stages, int Stages_dO, class TileShape_MNK_, class Element_, class ElementAccum_, class ArchTag_,
        bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool Deterministic,
        bool SdP_swapAB_, bool dKV_swapAB_, bool dQ_swapAB_,
        int NumMmaWarpGroups=2, int AtomLayoutMSdP=4, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
        bool V_in_regs=false>
struct CollectiveMainloopBwdSm80 {

    static constexpr int kStages = Stages;
    static constexpr int kStages_dO = Stages_dO;
    static_assert(kStages >= kStages_dO);
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;
    static constexpr int NumMmaWarps = NumMmaWarpGroups * cutlass::NumWarpsPerWarpGroup;

    static constexpr bool SdP_swapAB = SdP_swapAB_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool dQ_swapAB = dQ_swapAB_;

    static constexpr bool Q_dO_same_stages = kStages == kStages_dO;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    using SeqlenInfo_t = flash::SeqlenInfoQK<Varlen, kBlockM>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local>;

    static_assert(ArchTag::kMinComputeCapability >= 80);

    static constexpr bool Has_cp_async = ArchTag::kMinComputeCapability >= 80;

    // The CVT + swizzled smem load fast path (8x64 AIU gmem tiles fed to PPU0015_TSM_LD_SWZL_CVT)
    // is only implemented for kHeadDim == 128 on compute capability >= 89.
#if defined(USE_PPU) && USE_AIU && CVT_LAYOUT
    // The CVT + swizzled path requires kBlockN >= 128 because:
    //  1. PdS copy atoms (Trans=false) need BlockW >= 2*CUBE_W = 128
    //  2. Epilogue register permutation assumes 64 accumulator elements/thread (= kBlockN*kHeadDim/NumThreads with kBlockN=128)
    static constexpr bool Use_CVT_SWZL_LD = ArchTag::kMinComputeCapability >= 89 && (kHeadDim == 128 || kHeadDim == 256) && (kBlockN >= 128);
    // static constexpr bool Use_CVT_SWZL_LD = false;
    static constexpr bool Use_CVT_SWZL_LD_PdS = Use_CVT_SWZL_LD;
#else
    static constexpr bool Use_CVT_SWZL_LD = false;
    static constexpr bool Use_CVT_SWZL_LD_PdS = false;
#endif

    static constexpr int NumMmaThreads = NumMmaWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumProducerThreads = NumMmaThreads;  // For compatibility with TileScheduler

    using MMA_Atom_Arch =
#ifdef USE_PPU
        std::conditional_t<
            ArchTag::kMinComputeCapability >= 89,
            std::conditional_t<
                std::is_same_v<Element, cutlass::half_t>,
                MMA_Atom<PPU0015_16x16x16_F32F16F16F32_TN>,
                MMA_Atom<PPU0015_16x16x16_F32BF16BF16F32_TN>
            >,
            std::conditional_t<
                std::is_same_v<Element, cutlass::half_t>,
                MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
                MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>
            >
#else
        std::conditional_t<
        ArchTag::kMinComputeCapability >= 80,
        std::conditional_t<
            std::is_same_v<Element, cutlass::half_t>,
            MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
            MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
        >,
        MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>
#endif
    >;

    static_assert(NumMmaWarps % AtomLayoutMSdP == 0);
    static_assert(NumMmaWarps % AtomLayoutNdKV == 0);
    static_assert(NumMmaWarps % AtomLayoutMdQ == 0);
    static constexpr bool Mma_dKV_is_RS = AtomLayoutMSdP == 1 && AtomLayoutNdKV == NumMmaWarps && SdP_swapAB && !dKV_swapAB;
    static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == NumMmaWarps && AtomLayoutMdQ == NumMmaWarps && !SdP_swapAB && !dQ_swapAB;  // If dQ_swapAB we can't use RS

    using AtomLayoutSdP = std::conditional_t<
        !SdP_swapAB,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<NumMmaWarps / AtomLayoutMSdP>, _1>>,
        Layout<Shape<Int<NumMmaWarps / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>
    >;
    static constexpr bool MmaSdPEvenN = ((!SdP_swapAB ? kBlockN : kBlockM) / size<1>(AtomLayoutSdP{})) % 16 == 0;
#ifdef USE_PPU
    static_assert(MmaSdPEvenN, "MmaSdPEvenN: MMA N must be a multiple of 16 for PPU.");
#endif
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch,
        AtomLayoutSdP,
        Tile<Int<16 * CUTE_STATIC_V(size<0>(AtomLayoutSdP{}))>, Int<(MmaSdPEvenN ? 16 : 8) * CUTE_STATIC_V(size<1>(AtomLayoutSdP{}))>, _16>>;

    using AtomLayoutdKV = std::conditional_t<
        !dKV_swapAB,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<NumMmaWarps / AtomLayoutNdKV>, _1>>,
        Layout<Shape<Int<NumMmaWarps / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>
    >;
    static constexpr bool MmadKVEvenN = ((!dKV_swapAB ? kHeadDim : kBlockN) / size<1>(AtomLayoutdKV{})) % 16 == 0;
#ifdef USE_PPU
    static_assert(MmadKVEvenN, "MmadKVEvenN: MMA N must be a multiple of 16 for PPU.");
#endif
    using TiledMmadKV = TiledMMA<
        MMA_Atom_Arch,
        AtomLayoutdKV,
        Tile<Int<16 * CUTE_STATIC_V(size<0>(AtomLayoutdKV{}))>, Int<(MmadKVEvenN ? 16 : 8) * CUTE_STATIC_V(size<1>(AtomLayoutdKV{}))>, _16>>;

    using AtomLayoutdQ = std::conditional_t<
        !dQ_swapAB,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<NumMmaWarps / AtomLayoutMdQ>, _1>>,
        Layout<Shape<Int<NumMmaWarps / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>
    >;
    static constexpr bool MmadQEvenN = ((!dQ_swapAB ? kHeadDim : kBlockM) / size<1>(AtomLayoutdQ{})) % 16 == 0;
#ifdef USE_PPU
    static_assert(MmadQEvenN, "MmadQEvenN: MMA N must be a multiple of 16 for PPU.");
#endif
    using TiledMmadQ = TiledMMA<
        MMA_Atom_Arch,
        AtomLayoutdQ,
        Tile<Int<16 * CUTE_STATIC_V(size<0>(AtomLayoutdQ{}))>, Int<(MmadQEvenN ? 16 : 8) * CUTE_STATIC_V(size<1>(AtomLayoutdQ{}))>, _16>>;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g. if hdim=128, we want each
    // thread to have 4 loads in the M direction and 2 vectorized load in the K direction.
    static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int nBytePerRow = kBlockN * sizeof(Element);// 192
#if PPU1v0_R2S_SLICE_LAYOUT
    static constexpr int PPUChannelSliceSize = 16;
    static constexpr int PPUChannelSliceCount = 2;
    static constexpr int PPUChannelSlice2Size = PPUChannelSliceSize * PPUChannelSliceCount;
    static constexpr int kBlockNGmem = PPUChannelSlice2Size;
#else
    static constexpr int kBlockNGmem = (nBytePerRow % 128 == 0 ? 128 : (nBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
#endif
    static constexpr int mBytePerRow = kBlockM * sizeof(Element);
#if PPU1v0_R2S_SLICE_LAYOUT
    static constexpr int kBlockMGmem = PPUChannelSlice2Size;
#else
    static constexpr int kBlockMGmem = (mBytePerRow % 128 == 0 ? 128 : (mBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
#endif

    static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
    static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);

    // We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
    // Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
    // Since this is GMMA::Major::K, the M dimension (kBlockM) doesn't matter for the layout, only the K dimension
    // changes the layout.
#if defined(USE_PPU) && USE_AIU
    using SmemLayoutAtomQdO = Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>;
#else
    using SmemLayoutAtomQdO = decltype(
        composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                    Layout<Shape<_8, Int<kBlockKGmem>>,
                           Stride<Int<kBlockKGmem>, _1>>{}));
#endif
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutQ_AIU_COPY =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{}), LayoutRight{}));
    using SmemLayoutdO =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages_dO>{})));
    using SmemLayoutdO_AIU_COPY =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages_dO>{}), LayoutRight{}));

#if defined(USE_PPU) && USE_AIU
    using SmemLayoutAtomKV = Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>;
#else
    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                    // TODO: FA2 has a slightly different layout, does it matter?
                    Layout<Shape<_8, Int<kBlockKGmem>>,
                           Stride<Int<kBlockKGmem>, _1>>{}));
#endif

    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomKV{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutK_AIU_COPY = decltype(tile_to_shape(SmemLayoutAtomKV{}, select<1, 2>(TileShape_MNK{}), LayoutRight{}));
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomKV{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutV_AIU_COPY = decltype(tile_to_shape(SmemLayoutAtomKV{}, select<1, 2>(TileShape_MNK{}), LayoutRight{}));

    // TD [2023-03-19]: Idk why kPBlockN = 16 and kSwizzlePdS=3 is the fastest.
    static constexpr int kPBlockN = kBlockN % 64 == 0 ? 64 : (kBlockN % 32 == 0 ? 32 : 16);
    static_assert(kPBlockN == 16 || kPBlockN == 32 || kPBlockN == 64);
    // static constexpr int kSwizzlePdS = kPBlockN == 16 ? 1 : (kPBlockN == 32 ? 2 : 3);

#if PPU1v0_R2S_SLICE_LAYOUT
    using SmemLayoutAtomPdS_R2Slice = decltype(
        tile_to_shape(
        composition(Swizzle<1, kSwizzleBase, kSwizzleBase>{},
                    Layout<Shape<Int<PPUChannelSliceSize>, Int<PPUChannelSliceSize>>,
                           Stride<Int<1>, Int<PPUChannelSliceSize>>>{}),
        Layout<Shape<Int<PPUChannelSliceSize>, Int<kBlockN>>>{}
    ));

    // Store Layout of ppu aiu slice format
    using SmemLayoutPdS_R2Slice = decltype(tile_to_shape(
        SmemLayoutAtomPdS_R2Slice{},
        make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                    make_stride(Int<kBlockN>{}, _1{})
        )
    ));
    using SmemLayoutPdSt_R2Slice = decltype(composition(
        SmemLayoutPdS_R2Slice{},
        make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}),
                               make_stride(Int<kBlockM>{}, _1{})))
    );
#endif
#if PPU1v5_R2S_SLICE_LAYOUT
    static constexpr int Swizzle_Slice = kBlockNGmem == 64? 3: (kBlockNGmem == 32? 2:1);
    // CVT path uses a row-major tiled 8×kBlockNGmem atom, otherwise the atom spans kBlockM rows.
    using SmemLayoutAtomPdS_CVT = Layout<Shape<_8, Int<kBlockNGmem>>, Stride<Int<kBlockNGmem>, _1>>;
    using SmemLayoutAtomPdS_Slice = Layout<Shape<Int<kBlockM>, Int<kBlockNGmem>>, Stride<Int<kBlockNGmem>, Int<1>>>;
    using SmemLayoutPdS_R2Slice = std::conditional_t<
        Use_CVT_SWZL_LD_PdS,
        decltype(tile_to_shape(
            composition(Swizzle<Swizzle_Slice, kSwizzleBase, kSwizzleBase>{},
                        SmemLayoutAtomPdS_CVT{}),
            make_shape(Int<kBlockM>{}, Int<kBlockN>{}), LayoutRight{})),
        decltype(tile_to_shape(
            composition(Swizzle<Swizzle_Slice, kSwizzleBase, kSwizzleBase>{},
                        SmemLayoutAtomPdS_Slice{}),
            Layout<Shape<Int<kBlockM>, Int<kBlockN>>>{}))
    >;
    using SmemLayoutPdSt_R2Slice = decltype(composition(
        SmemLayoutPdS_R2Slice{},
        make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}),
                    make_stride(Int<kBlockM>{}, Int<1>{}))));
#endif
    static constexpr int kSwizzlePdS = 3;
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<kSwizzlePdS, kSwizzleBase, kSwizzleBase>{},
                    Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                           Stride<Int<kPBlockN>, _1>>{}));
#if PPU1v0_R2S_SLICE_LAYOUT
    using SmemLayoutPdSt = decltype(tile_to_shape(
        Layout<Shape<Int<kBlockN>, Int<PPUChannelSlice2Size>>, Stride<Int<PPUChannelSlice2Size>, _1>>{},
        make_shape(Int<kBlockN>{}, Int<kBlockM>{})));
    using SmemLayoutPdS =
        decltype(cute::composition(SmemLayoutPdSt{},
                                   make_layout(make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                                               make_stride(Int<kBlockN>{}, _1{}))));
#else
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using SmemLayoutPdSt =
        decltype(cute::composition(SmemLayoutPdS{},
        make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}),
                    make_stride(Int<kBlockM>{}, _1{}))));
#endif

    // We set stride to be multiple of 64 so that if ShuffleLSE, even if threads read from sLSE but out of bounds,
    // it's still a valid smem address.
    using SmemLayoutLSE = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>, cute::Stride<_1, Int<cute::round_up(kBlockM, 64)>>>;
    using SmemLayoutLSEMma = std::conditional_t<
        SdP_swapAB,
        cute::Layout<cute::Shape<Int<kBlockN>, Int<kBlockM>, Int<kStages>>, cute::Stride<_0, _1, Int<cute::round_up(kBlockM, 64)>>>,
        cute::Layout<cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>, cute::Stride<_1, _0, Int<cute::round_up(kBlockM, 64)>>>
    >;

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutQt =
        decltype(cute::composition(SmemLayoutQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutdOt =
        decltype(cute::composition(SmemLayoutdO{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages_dO>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutKt =
        decltype(cute::composition(SmemLayoutK{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(Int<kBlockN>{}, _1{}))));
    // Thread layout, 256 or 384 threads per row
    using R2SLayoutAtomdQaccum = Layout<Shape<Int<NumMmaThreads>>>;
    using R2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2SLayoutAtomdQaccum{},
                                                         Layout<Shape < _1>>{}));  // Val layout, 1 vals per store

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
#if PPU1v0_R2S_SLICE_LAYOUT
    // we purpose to pre-transpose P in share memory and load in non-transpose fashion
    using SmemCopyAtomPdSt_AIU = Copy_Atom<PPU0010_TSM_LD_SWZL<Element, kBlockN, kBlockMGmem, false, false, kBlockM / kBlockMGmem>, Element>;
    using SmemCopyAtomdS_AIU = Copy_Atom<PPU0010_TSM_LD_SWZL<Element, kBlockN, kBlockMGmem, false, true, kBlockM / kBlockMGmem>, Element>;
#endif
#if PPU1v5_R2S_SLICE_LAYOUT
    // Use a safe BlockN (>= 128) for CVT_LOCAL template instantiation to avoid its internal
    // static_assert(BlockW % 128 == 0) firing, even when the CVT branch is never selected.
    static constexpr int kBlockN_PdS_safe = (kBlockN >= 128) ? kBlockN : 128;
    using SmemCopyAtomdS_AIU = Copy_Atom<std::conditional_t<
        Use_CVT_SWZL_LD_PdS,
        PPU0015_TSM_LD_SWZL_CVT_LOCAL<Element, 8, 64, kBlockM, kBlockN_PdS_safe, dQ_swapAB, false, kBlockM / 8 * kBlockN_PdS_safe / 64, true, 64, 128, true>,
        PPU0015_TSM_LD_SWZL<Element, kBlockM, kBlockNGmem, dQ_swapAB, false, kBlockN / kBlockNGmem>>, Element>;
    using SmemCopyAtomPdSt_AIU = Copy_Atom<std::conditional_t<
        Use_CVT_SWZL_LD_PdS,
        PPU0015_TSM_LD_SWZL_CVT_LOCAL<Element, 8, 64, kBlockM, kBlockN_PdS_safe, dKV_swapAB, true, kBlockM / 8 * kBlockN_PdS_safe / 64, true, 128, 64>,
        PPU0015_TSM_LD_SWZL<Element, kBlockM, kBlockNGmem, dKV_swapAB, true, kBlockN / kBlockNGmem>>, Element>;
#endif
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
    // For the case where the N dimension of MmaSdP is divisible by 8 but not by 16
    using SmemCopyAtomHalf = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    // For the case where the N dimension of MmadQ is divisible by 8 but not by 16
    using SmemCopyAtomTransposedHalf = Copy_Atom<SM75_U16x4_LDSM_T, Element>;
#if defined(USE_PPU) && USE_AIU
    // The CVT copy ops always address a headdim tile padded to 128 elements.
    static constexpr int kHeadDimPadding = 128;
    using SmemCopyOpQ = std::conditional_t<
        Use_CVT_SWZL_LD,
        PPU0015_TSM_LD_SWZL_CVT_LOCAL<Element, 8, 64, kBlockM, kHeadDimPadding, SdP_swapAB, false, kBlockM / 8 * kHeadDim / 64, true, 128, 64>,
        std::conditional_t<
            ArchTag::kMinComputeCapability >= 89,
            PPU0015_TSM_LD_SWZL<Element, kBlockM, kBlockKGmem, SdP_swapAB, false, kHeadDim / kBlockKGmem>,
            PPU0010_TSM_LD_SWZL<Element, kBlockM, kBlockKGmem, false, false, kHeadDim / kBlockKGmem>>>;
    using SmemCopyOpQt = std::conditional_t<
        Use_CVT_SWZL_LD,
        PPU0015_TSM_LD_SWZL_CVT_LOCAL<Element, 8, 64, kBlockM, kHeadDimPadding, !dKV_swapAB, true, kBlockM / 8 * kHeadDim / 64, true, 128, 64>,
        std::conditional_t<
            ArchTag::kMinComputeCapability >= 89,
            PPU0015_TSM_LD_SWZL<Element, kBlockM, kBlockKGmem, !dKV_swapAB, true, kHeadDim / kBlockKGmem>,
            PPU0010_TSM_LD_SWZL<Element, kBlockM, kBlockKGmem, false, true, kHeadDim / kBlockKGmem>>>;
    using SmemCopyOpK = std::conditional_t<
        Use_CVT_SWZL_LD,
        PPU0015_TSM_LD_SWZL_CVT_LOCAL<Element, 8, 64, kBlockN, kHeadDimPadding, !SdP_swapAB, false, kBlockN / 8 * kHeadDim / 64, true, 128, 64>,
        std::conditional_t<
            ArchTag::kMinComputeCapability >= 89,
            PPU0015_TSM_LD_SWZL<Element, kBlockN, kBlockKGmem, !SdP_swapAB, false, kHeadDim / kBlockKGmem>,
            PPU0010_TSM_LD_SWZL<Element, kBlockN, kBlockKGmem, false, false, kHeadDim / kBlockKGmem>>>;
    using SmemCopyOpKVt = std::conditional_t<
        Use_CVT_SWZL_LD,
        PPU0015_TSM_LD_SWZL_CVT_LOCAL<Element, 8, 64, kBlockN, kHeadDimPadding, !dQ_swapAB, true,  kBlockN / 8 * kHeadDim / 64, true, 64, 1024, true>,
        std::conditional_t<
            ArchTag::kMinComputeCapability >= 89,
            PPU0015_TSM_LD_SWZL<Element, kBlockN, kBlockKGmem, !dQ_swapAB, true, kHeadDim / kBlockKGmem>,
            PPU0010_TSM_LD_SWZL<Element, kBlockN, kBlockKGmem, false, true, kHeadDim / kBlockKGmem>>>;
    using SmemCopyAtomQ = Copy_Atom<SmemCopyOpQ, Element>;
    using SmemCopyAtomQt = Copy_Atom<SmemCopyOpQt, Element>;
    using SmemCopyAtomK = Copy_Atom<SmemCopyOpK, Element>;
    using SmemCopyAtomKVt = Copy_Atom<SmemCopyOpKVt, Element>;
#else
    using SmemCopyAtomQ = SmemCopyAtom;
    using SmemCopyAtomQt = SmemCopyAtomTransposed;
    using SmemCopyAtomK = SmemCopyAtom;
    using SmemCopyAtomKVt = SmemCopyAtomTransposed;
#endif
    // If !SdP_swapAB, the accum registers hold P / dS, otherwise they hold Pt / dSt.
    // If PdS_major is MN, then we need to "transpose" the write.
    // TODO: check this write
    using R2SCopyAtomPdS = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using GmemCopyStruct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>,
        AutoVectorizingCopyWithAssumedAlignment<128>
    >;
    using GmemCopyAtom = Copy_Atom<GmemCopyStruct, Element>;

    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(GmemCopyAtom{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per read
#if defined(USE_PPU) && USE_AIU
    static constexpr int bits_per_aiu_Q = kBlockM * kBlockKGmem * sizeof(Element) * 8;
    static constexpr int bits_per_aiu_KV = kBlockN * kBlockKGmem * sizeof(Element) * 8;
    static constexpr int bits_per_aiu_Q_Cvt = 8 * kBlockKGmem * sizeof(Element) * 8;
    static constexpr int bits_per_aiu_KV_Cvt = 8 * kBlockKGmem * sizeof(Element) * 8;
    using Gmem_copy_struct_Q = std::conditional_t<
        Use_CVT_SWZL_LD,
        PPU0015_AIU_LOAD<cute::C<bits_per_aiu_Q_Cvt>, Element, false, 8, kBlockKGmem>,
        std::conditional_t<
            ArchTag::kMinComputeCapability >= 89,
            PPU0015_AIU_LOAD<cute::C<bits_per_aiu_Q>, Element, false, kBlockM, kBlockKGmem>,
            PPU0010_AIU_LOAD<cute::C<bits_per_aiu_Q>, Element, false>>>;
    using Gmem_copy_struct_KV = std::conditional_t<
        Use_CVT_SWZL_LD,
        PPU0015_AIU_LOAD<cute::C<bits_per_aiu_KV_Cvt>, Element, false, 8, kBlockKGmem>,
        std::conditional_t<
            ArchTag::kMinComputeCapability >= 89,
            PPU0015_AIU_LOAD<cute::C<bits_per_aiu_KV>, Element, false, kBlockN, kBlockKGmem>,
            PPU0010_AIU_LOAD<cute::C<bits_per_aiu_KV>, Element, false>>>;
    // CVT path loads 8 x kBlockKGmem gmem tiles, the other paths load a full kBlockM/kBlockN block.
    using GmemTiledCopyQ = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_Q, Element>{},
                        Layout<Shape <_1,_1>,
                               Stride<_1,_1>>{},
                        std::conditional_t<Use_CVT_SWZL_LD,
                            Layout<Shape <Int<8>, Int<kBlockKGmem>>>,
                            Layout<Shape <Int<kBlockM>, Int<kBlockKGmem>>>>{}));
    using GmemTiledCopyKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_KV, Element>{},
                        Layout<Shape <_1,_1>,
                               Stride<_1,_1>>{},
                        std::conditional_t<Use_CVT_SWZL_LD,
                            Layout<Shape <Int<8>, Int<kBlockKGmem>>>,
                            Layout<Shape <Int<kBlockN>, Int<kBlockKGmem>>>>{}));
#else
    using GmemTiledCopyQ = GmemTiledCopyQKV;
    using GmemTiledCopyKV = GmemTiledCopyQKV;
#endif
    using GmemCopyAtomLSE = Copy_Atom<GmemCopyStruct, float>;
    using GmemLayoutAtomLSE = Layout<Shape<Int<NumMmaThreads>>>;
    using GmemTiledCopyLSE = decltype(make_tiled_copy(GmemCopyAtomLSE{}, GmemLayoutAtomLSE{},
                                                      Layout<Shape<_4>>{}));  // Val layout, 4 vals per store
    // So that we don't have to check if we overshot kBlockM when we load Q
    // static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, head, batch)
    using StrideLSE = cute::Stride<_1, int64_t, int64_t>;  // (seqlen, head, batch)
    using ShapedQaccum = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_q * d, head, batch)
    using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

    // These are tuned for speed. They don't affect correctness.
    // We have separate iterations with causal masking. Not necessary for hdim 128 but for hdim 64
    // this helps quite a bit to not have to do causal masking for most of the iterations.
    // For hdim 192, separating masking iterations results in register spills.
    // static constexpr bool SeparateMaskingIterations = kHeadDim <= 64;
    static constexpr bool SeparateMaskingIterations = false;
    // Do we keep the LSE and dPsum in each thread, or split them across 8 threads that share them and then
    // shuffle to get the value whenever we need? This can reduce register pressure when SdP_swapAB, where each
    // thread needs to keep statistics for (kBlockM / 4) rows. If !SdP_swapAB, each thread only needs to keep
    // statistic for 2 rows.
    // static constexpr bool ShuffleLSE = SdP_swapAB && kHeadDim <= 64;
    // static constexpr bool ShuffledPsum = SdP_swapAB && kHeadDim <= 64;
    static constexpr bool ShuffleLSE = SdP_swapAB && false;
    static constexpr bool ShuffledPsum = SdP_swapAB && false;

    static constexpr bool Share_QV_Smem = V_in_regs;
    using SmemP_t = std::conditional_t<Mma_dKV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>>>;

    struct CUTE_ALIGNAS(128) TensorStorageSharedQV {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        union {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>> smem_ds;
        SmemP_t smem_p;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>> smem_lse;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>> smem_dpsum;
    };

    struct CUTE_ALIGNAS(128) TensorStorageSeparateQV {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>> smem_ds;
        SmemP_t smem_p;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>> smem_lse;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>> smem_dpsum;
    };

    using TensorStorage = std::conditional_t<Share_QV_Smem, TensorStorageSharedQV, TensorStorageSeparateQV>;

    // Host side kernel arguments
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* const ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        Element const* const ptr_V;
        ShapeQKV const shape_V;
        StrideQKV const stride_V;
        Element const* const ptr_dO;
        ShapeQKV const shape_dO;
        StrideQKV const stride_dO;
        ElementAccum* const ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum const stride_dQaccum;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale;
        int const window_size_left, window_size_right, attention_chunk;
        float const softcap_val;
        int const num_batch;
        int* const dq_semaphore;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
    };

    // Device side kernel params
    struct Params {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* const ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        Element const* const ptr_V;
        ShapeQKV const shape_V;
        StrideQKV const stride_V;
        Element const* const ptr_dO;
        ShapeQKV const shape_dO;
        StrideQKV const stride_dO;
        ElementAccum* const ptr_dQaccum;
        ShapedQaccum const shape_dQaccum;
        StridedQaccum stride_dQaccum;
        cutlass::FastDivmod qhead_per_khead_divmod;
        float const* const ptr_LSE_log2;
        ShapeLSE const shape_LSE;
        StrideLSE const stride_LSE_log2;
        float const* const ptr_dPsum;
        StrideLSE const stride_dPsum;
        float const softmax_scale, softmax_scale_log2;
        int const window_size_left, window_size_right;
        cutlass::FastDivmod attention_chunk_divmod;
        float const softcap_val;
        int const num_batch;
        int *const dq_semaphore;
        int const *const cu_seqlens_q = nullptr;
        int const *const cu_seqlens_k = nullptr;
        int const *const seqused_q = nullptr;
        int const *const seqused_k = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        if constexpr (Deterministic) { assert(args.dq_semaphore != nullptr); }
        // Avoid dividing by zero
        cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
        attention_chunk_divmod.divisor = args.attention_chunk;
        // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        // Right after this, we multiply by log2(e) before applying exp2.
        // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
        // (assigning it to params.softmax_scale_log2).
        // In the backward, we need to multiply by
        // (1 - tanh^2) * softmax_scale / softcap_val * softcap_val = (1 - tanh^2) * softmax_scale.
        // Instead we multiply by (1 - tanh^2) and multiply dK and dV by params.softmax_scale
        // (the original softmax_scale) at the end.
        return {args.ptr_Q, args.shape_Q, args.stride_Q,
                args.ptr_K, args.shape_K, args.stride_K,
                args.ptr_V, args.shape_V, args.stride_V,
                args.ptr_dO, args.shape_dO, args.stride_dO,
                args.ptr_dQaccum, args.shape_dQaccum, args.stride_dQaccum,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                args.ptr_LSE_log2, args.shape_LSE, args.stride_LSE_log2, args.ptr_dPsum, args.stride_dPsum,
                args.softmax_scale,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                args.window_size_left, args.window_size_right, attention_chunk_divmod,
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                args.num_batch, args.dq_semaphore,
                args.cu_seqlens_q, args.cu_seqlens_k, args.seqused_q, args.seqused_k};
    }

    template <typename SharedStorage, typename FrgTensordKV>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        FrgTensordKV& tdKrdK,
        FrgTensordKV& tdVrdV,
        int thread_idx,
        cute::tuple<int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

        int n_block = get<0>(block_coord);
        int bidh = get<1>(block_coord);
        int bidb = get<2>(block_coord);
        SeqlenInfo_t seqlen_info{
            bidb, get<0>(params.shape_Q), size<0>(params.shape_K),
            params.cu_seqlens_q, params.cu_seqlens_k, params.seqused_q, params.seqused_k
        };
        auto m_block_min_max = BlockMN_t::get_m_block_min_max(
            seqlen_info, n_block, bidb,
            params.window_size_left, params.window_size_right, 0 /*sink_token_length*/);
        int const m_block_min = get<0>(m_block_min_max);
        int const m_block_max = get<1>(m_block_min_max);
        // It's possible to have m_block_max <= m_block_min. Exit early
        if constexpr (Is_causal || Is_local || Varlen) {
            if (m_block_max <= m_block_min) { return false; }
        }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        // R2S copy view: AIU_LOAD writes 8x64 tiles in row-major (LayoutRight) order.
        // The compute path still uses sQ/sdO/sK/sV (LayoutLeft) for partition_S.
        // PPU copy atoms compute smem addresses internally, so both views share the
        // same physical buffer as long as swzl_mode is consistent.
        // Only referenced on the Use_CVT_SWZL_LD path.
        Tensor sQ_AIU_COPY = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ_AIU_COPY{});
        Tensor sdO_AIU_COPY = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO_AIU_COPY{});
        Tensor sK_AIU_COPY = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK_AIU_COPY{});
        Tensor sV_AIU_COPY = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV_AIU_COPY{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdOt{});
        Tensor sKt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutKt{});
        Tensor sP = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdS{});
        Tensor sPt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdSt{});
        Tensor sdS = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdS{});

#if PPU1v0_R2S_SLICE_LAYOUT || PPU1v5_R2S_SLICE_LAYOUT
        Tensor sdS_slice = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdS_R2Slice{});
        Tensor sdSt_slice = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdSt_R2Slice{});
        Tensor sP_slice = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdS_R2Slice{});
        Tensor sPt_slice = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdSt_R2Slice{});
#endif
        Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdSt{});
        Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSE{});
        Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSE{});
        Tensor sLSEMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
        Tensor sdPsumMma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});

        bool const is_varlen_q = Varlen && params.cu_seqlens_q;
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);
        Tensor mQ = make_tensor(make_gmem_ptr(params.ptr_Q), params.shape_Q, params.stride_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdO = make_tensor(make_gmem_ptr(params.ptr_dO), params.shape_dO, params.stride_dO)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V), params.shape_V, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE, params.stride_LSE_log2)(_, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSE, params.stride_dPsum)(_, bidh, !is_varlen_q ? bidb : 0);
        Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQaccum)),
                                      params.shape_dQaccum, params.stride_dQaccum)(_, bidh, !is_varlen_q ? bidb : 0);

        Tensor gQ = local_tile(make_mix_tensor_like(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ)), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gdO = local_tile(make_mix_tensor_like(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdO)), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gK = local_tile(make_mix_tensor_like(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK)), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gV = local_tile(make_mix_tensor_like(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV)), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gLSE = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mLSE), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdPsum = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded), mdPsum), select<0>(TileShape_MNK{}), make_coord(_));  // (M, _)
        Tensor gdQaccum = local_tile(domain_offset(make_coord(seqlen_info.offset_q_padded * kHeadDim), mdQaccum), Shape<Int<kBlockM * kHeadDim>>{}, make_coord(_));  // (M * K, _)

        GmemTiledCopyQ gmem_tiled_copy_Q;
        GmemTiledCopyKV gmem_tiled_copy_K;
        GmemTiledCopyKV gmem_tiled_copy_V;
        GmemTiledCopyQ gmem_tiled_copy_dO;
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(thread_idx);
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(thread_idx);
        auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(thread_idx);
        auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_Q = gmem_tiled_copy_Q.get_thread_slice(_0{});  // For index calculation
        auto gmem_thr0_copy_KV = gmem_tiled_copy_K.get_thread_slice(_0{});  // For index calculation
        GmemTiledCopyLSE gmem_tiled_copy_lse;
        auto gmem_thr_copy_lse = gmem_tiled_copy_lse.get_thread_slice(thread_idx);
        R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
        auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);

        Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
        Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
        Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
        Tensor tLSEgLSE = gmem_thr_copy_lse.partition_S(gLSE);
        Tensor tLSEsLSE = gmem_thr_copy_lse.partition_D(sLSE);
        Tensor tLSEgdPsum = gmem_thr_copy_lse.partition_S(gdPsum);
        Tensor tLSEsdPsum = gmem_thr_copy_lse.partition_D(sdPsum);
        // We can reuse r2s_thr_copy_dQaccum for this partitioning
        Tensor tdQgdQaccum = r2s_thr_copy_dQaccum.partition_D(gdQaccum);
        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(mdQaccum); printf("\n"); print(gdQaccum_); printf("\n"); print(gdQaccum); printf("\n"); print(tdQgdQaccum); printf("\n"); }

        TiledMmaSdP tiled_mma_SdP;
        TiledMmadKV tiled_mma_dKV;
        TiledMmadQ tiled_mma_dQ;

        auto thr_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto thr_mma_dKV = tiled_mma_dKV.get_thread_slice(thread_idx);
        auto thr_mma_dQ = tiled_mma_dQ.get_thread_slice(thread_idx);

        // Allocate "fragments/descriptors"
        // We have to use the templated mma_partition_fragment_AB instead of cute::conditional_return or lambda,
        // because some partition_fragment_A/B don't compile.
        // https://stackoverflow.com/questions/50051473/if-constexpr-in-c17-does-not-work-in-a-non-templated-function
        Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(thr_mma_SdP, sV);

#if defined(USE_PPU) && USE_AIU
        if constexpr (ArchTag::kMinComputeCapability >= 89) {
            if constexpr (Use_CVT_SWZL_LD) {
                // CVT path: 8x64 tile desc for direct AIU copy loops.
                // gmem block offset is handled by the direct AIU copy loop.
                static constexpr int kBlockNPerAiuLoad = 8;
                gmem_tiled_copy_Q.desc_.init(nullptr, kBlockNPerAiuLoad, kBlockKGmem, get<0>(params.stride_Q));
                gmem_tiled_copy_K.desc_.init(nullptr, kBlockNPerAiuLoad, kBlockNGmem, get<0>(params.stride_K));
                gmem_tiled_copy_V.desc_.init(nullptr, kBlockNPerAiuLoad, kBlockNGmem, get<0>(params.stride_V));
                gmem_tiled_copy_dO.desc_.init(nullptr, kBlockNPerAiuLoad, kBlockKGmem, get<0>(params.stride_dO));
            } else {
                gmem_tiled_copy_Q.desc_.init(nullptr, seqlen_info.seqlen_q, get<1>(params.shape_Q), get<0>(params.stride_Q));
                gmem_tiled_copy_K.desc_.init(nullptr, seqlen_info.seqlen_k, get<1>(params.shape_Q), get<0>(params.stride_K));
                gmem_tiled_copy_V.desc_.init(nullptr, seqlen_info.seqlen_k, get<1>(params.shape_V), get<0>(params.stride_V));
                gmem_tiled_copy_dO.desc_.init(nullptr, seqlen_info.seqlen_q, get<1>(params.shape_V), get<0>(params.stride_dO));
            }
        } else {
            int aiu_offset_q = get<1>(params.shape_Q) == kHeadDim ? 0 : (get<0>(params.stride_Q) - get<1>(params.shape_Q));
            int aiu_offset_k = get<1>(params.shape_Q) == kHeadDim ? 0 : (get<0>(params.stride_K) - get<1>(params.shape_Q));
            int aiu_offset_v = get<1>(params.shape_V) == kHeadDim ? 0 : (get<0>(params.stride_V) - get<1>(params.shape_V));
            int aiu_offset_do = get<1>(params.shape_V) == kHeadDim ? 0 : (get<0>(params.stride_dO) - get<1>(params.shape_V));
            gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, seqlen_info.seqlen_q, get<0>(params.stride_Q), kBlockM, kBlockKGmem, aiu_offset_q};
            gmem_tiled_copy_K.desc_ = AiuDesc{nullptr, seqlen_info.seqlen_k, get<0>(params.stride_K), kBlockN, kBlockKGmem, aiu_offset_k};
            gmem_tiled_copy_V.desc_ = AiuDesc{nullptr, seqlen_info.seqlen_k, get<0>(params.stride_V), kBlockN, kBlockKGmem, aiu_offset_v};
            gmem_tiled_copy_dO.desc_ = AiuDesc{nullptr, seqlen_info.seqlen_q, get<0>(params.stride_dO), kBlockM, kBlockKGmem, aiu_offset_do};
        }
        const int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
        const int tid_thread_slice = warp_idx * 32;
#if PPU1v0_R2S_SLICE_LAYOUT
        auto r2S_thread_idx_PdS = thread_idx;
        if(warp_idx % 2){ // keep slice0 layout
            r2S_thread_idx_PdS = __shfl_sync(0xffffffff, r2S_thread_idx_PdS, (r2S_thread_idx_PdS & 31) ^ PPUChannelSliceCount); // simulate slice1 layout: exchage t0~3 with t4~7, t8~11 with t12~15, t16~19 with t20~23, t24~27 with t28~t31
        }
#elif PPU1v5_R2S_SLICE_LAYOUT
        auto r2S_thread_idx_PdS = thread_idx;
#endif
#else
        const int tid_thread_slice = thread_idx;
#endif
        // Copy Atom retiling
        auto smem_copy_atom_SdP_B = cute::conditional_return<MmaSdPEvenN>(SmemCopyAtom{}, SmemCopyAtomHalf{});
#if defined(USE_PPU) && USE_AIU
        auto smem_tiled_copy_QdO = cute::conditional_return<!SdP_swapAB>(make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_SdP), make_tiled_copy_B(SmemCopyAtomQ{}, tiled_mma_SdP));
#else
        auto smem_tiled_copy_QdO = cute::conditional_return<!SdP_swapAB>(make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_SdP), make_tiled_copy_B(smem_copy_atom_SdP_B, tiled_mma_SdP));
#endif
        auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tid_thread_slice);
        Tensor tSsQ = smem_thr_copy_QdO.partition_S(make_mix_tensor_like(sQ));
        Tensor tdPsdO = smem_thr_copy_QdO.partition_S(make_mix_tensor_like(sdO));

#if defined(USE_PPU) && USE_AIU
        auto smem_tiled_copy_KV = cute::conditional_return<!SdP_swapAB>(make_tiled_copy_B(SmemCopyAtomK{}, tiled_mma_SdP), make_tiled_copy_A(SmemCopyAtomK{}, tiled_mma_SdP));
#else
        auto smem_tiled_copy_KV = cute::conditional_return<!SdP_swapAB>(make_tiled_copy_B(smem_copy_atom_SdP_B, tiled_mma_SdP), make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_SdP));
#endif
        auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tid_thread_slice);
        Tensor tSsK = smem_thr_copy_KV.partition_S(make_mix_tensor_like(sK));
        Tensor tdPsV = smem_thr_copy_KV.partition_S(make_mix_tensor_like(sV));

        auto r2s_tiled_copy_PdS = make_tiled_copy_C(R2SCopyAtomPdS{}, tiled_mma_SdP);
#if PPU1v0_R2S_SLICE_LAYOUT || PPU1v5_R2S_SLICE_LAYOUT
        auto r2s_thr_copy_PdS = r2s_tiled_copy_PdS.get_thread_slice(r2S_thread_idx_PdS);
        Tensor tdSsdS = r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS_slice, sdSt_slice));      // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor tPsP = r2s_thr_copy_PdS.partition_D((cute::conditional_return<!SdP_swapAB>(sP_slice, sPt_slice)));      // ((Atom,AtomNum),PIPE_M,PIPE_N)
#else
        auto r2s_thr_copy_PdS = r2s_tiled_copy_PdS.get_thread_slice(thread_idx);
        Tensor tdSsdS = r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS, sdSt));      // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor tPsP = r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP, sPt));      // ((Atom,AtomNum),PIPE_M,PIPE_N)
#endif
        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(r2s_thr_copy_PdS); print(sP); printf("\n"); print(sPt); printf("\n"); print(tPsP); printf("\n"); print(tdSsdS); printf("\n"); }

#if PPU1v0_R2S_SLICE_LAYOUT|| PPU1v5_R2S_SLICE_LAYOUT
        // TODO: Adapt SmemCopyAtomTransposedHalf
        auto smem_copy_atom_dKV_B = cute::conditional_return<MmadKVEvenN>(SmemCopyAtomPdSt_AIU{}, SmemCopyAtomTransposedHalf{});
        auto smem_tiled_copy_PdSt = cute::conditional_return<!dKV_swapAB>(make_tiled_copy_A(SmemCopyAtomPdSt_AIU{}, tiled_mma_dKV), make_tiled_copy_B(smem_copy_atom_dKV_B, tiled_mma_dKV));
        auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tid_thread_slice);
        Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(make_mix_tensor_like(sPt));
        Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(make_mix_tensor_like(sdSt));
#else
        auto smem_copy_atom_dKV_B = cute::conditional_return<MmadKVEvenN>(SmemCopyAtomTransposed{}, SmemCopyAtomTransposedHalf{});
        auto smem_tiled_copy_PdSt = cute::conditional_return<!dKV_swapAB>(make_tiled_copy_A(SmemCopyAtomTransposed{}, tiled_mma_dKV), make_tiled_copy_B(smem_copy_atom_dKV_B, tiled_mma_dKV));
        auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(thread_idx);
        Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
        Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);
#endif

#if defined(USE_PPU) && USE_AIU
        auto smem_tiled_copy_QdOt = cute::conditional_return<!dKV_swapAB>(make_tiled_copy_B(SmemCopyAtomQt{}, tiled_mma_dKV), make_tiled_copy_A(SmemCopyAtomQt{}, tiled_mma_dKV));
#else
        auto smem_tiled_copy_QdOt = cute::conditional_return<!dKV_swapAB>(make_tiled_copy_B(smem_copy_atom_dKV_B, tiled_mma_dKV), make_tiled_copy_A(SmemCopyAtomTransposed{}, tiled_mma_dKV));
#endif
        auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tid_thread_slice);
        Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(make_mix_tensor_like(sdOt));
        Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(make_mix_tensor_like(sQt));

#if PPU1v0_R2S_SLICE_LAYOUT|| PPU1v5_R2S_SLICE_LAYOUT
        auto smem_tiled_copy_dS = cute::conditional_return<!dQ_swapAB>(
            make_tiled_copy_A(SmemCopyAtomdS_AIU{}, tiled_mma_dQ),
            make_tiled_copy_B(cute::conditional_return<MmadQEvenN>(SmemCopyAtom{}, SmemCopyAtomHalf{}), tiled_mma_dQ));
        auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tid_thread_slice);
        Tensor tdQsdS = smem_thr_copy_dS.partition_S(make_mix_tensor_like(sdS));
#else
        auto smem_tiled_copy_dS = cute::conditional_return<!dQ_swapAB>(
            make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_dQ),
            make_tiled_copy_B(cute::conditional_return<MmadQEvenN>(SmemCopyAtom{}, SmemCopyAtomHalf{}), tiled_mma_dQ));
        auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(thread_idx);
        Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);
#endif

        auto smem_tiled_copy_Kt = cute::conditional_return<!dQ_swapAB>(
#if defined(USE_PPU) && USE_AIU
            make_tiled_copy_B(SmemCopyAtomKVt{}, tiled_mma_dQ),
            make_tiled_copy_A(SmemCopyAtomKVt{}, tiled_mma_dQ));
#else
            make_tiled_copy_B(cute::conditional_return<MmadQEvenN>(SmemCopyAtomTransposed{}, SmemCopyAtomTransposedHalf{}), tiled_mma_dQ),
            make_tiled_copy_A(SmemCopyAtomTransposed{}, tiled_mma_dQ));
#endif
        auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tid_thread_slice);
        Tensor tdQsKt = smem_thr_copy_Kt.partition_S(make_mix_tensor_like(sKt));

        // thr_mma_SdP.partition_C(sLSEMma) has shape (MMA=4, MMA_M, MMA_N, PIPE), we only take the col indices
        // or row indices, depending on whether SdP_swapAB.
#ifdef USE_PPU
        Tensor tSsLSEMma = [&]() -> auto {
            if constexpr (ArchTag::kMinComputeCapability >= 89) {
                return group_modes<0, 2>(logical_divide(thr_mma_SdP.partition_C(sLSEMma), Shape<_2>{})(make_coord(_, make_coord(_, _0{})), _, _, _));
            } else {
                return logical_divide(thr_mma_SdP.partition_C(sLSEMma), Shape<_4>{});  // (4, 2, MMA_M, MMA_N, PIPE)
            }
        }();
#else
        Tensor tSsLSEMma = logical_divide(thr_mma_SdP.partition_C(sLSEMma), Shape<_2>{});  // (2, 2, MMA_M, MMA_N, PIPE)
#endif
        Tensor tSsLSE = group_modes<0, 2>(cute::conditional_return<!SdP_swapAB>(
            tSsLSEMma(make_coord(_0{}, _), _, _0{}, _),  // (2, MMA_M, PIPE)
            tSsLSEMma(make_coord(_, _0{}), _0{}, _, _)));  // (2, MMA_N, PIPE)
#ifdef USE_PPU
        Tensor tSsdPsumMma = [&]() -> auto {
            if constexpr (ArchTag::kMinComputeCapability >= 89) {
                return group_modes<0, 2>(logical_divide(thr_mma_SdP.partition_C(sdPsumMma), Shape<_2>{})(make_coord(_, make_coord(_, _0{})), _, _, _));
            } else {
                return logical_divide(thr_mma_SdP.partition_C(sdPsumMma), Shape<_4>{});
            }
        }();
#else
        Tensor tSsdPsumMma = logical_divide(thr_mma_SdP.partition_C(sdPsumMma), Shape<_2>{});
#endif
        Tensor tSsdPsum = group_modes<0, 2>(cute::conditional_return<!SdP_swapAB>(
            tSsdPsumMma(make_coord(_0{}, _), _, _0{}, _),  // (2, MMA_M, PIPE)
            tSsdPsumMma(make_coord(_, _0{}), _0{}, _, _)));  // (2, MMA_N, PIPE)
        // if (blockIdx.x == 0 && threadIdx.x == 128) { print(sLSEMma); printf("\n"); print(tLSEsLSE); printf("\n"); }
        // If we want to split the stats among the 8 threads that share the same rows.
        static constexpr int kStatsPerThread = cute::ceil_div(decltype(size(tSsLSE))::value, 8);

        // Predicates
        Tensor cQ = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
        Tensor t0QcQ = gmem_thr0_copy_Q.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.shape_Q); }
        Tensor cLSE = cute::make_identity_tensor(select<0>(TileShape_MNK{}));
        Tensor tLSEcLSE = gmem_thr_copy_lse.partition_S(cLSE);
        Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdOsdO)));
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.shape_dO); }

        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;

        flash::Mask<kBlockM, kBlockN, false /*PackGQA*/, TiledMmaSdP, SdP_swapAB> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod
        );

        {
            Tensor tKsK = [&]() -> auto {
                if constexpr (Use_CVT_SWZL_LD) {
                    return gmem_thr_copy_K.partition_D(sK_AIU_COPY);
                } else {
                    return gmem_thr_copy_K.partition_D(sK);
                }
            }();
            Tensor tKgK = gmem_thr_copy_K.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
            Tensor tVgV = gmem_thr_copy_V.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
            Tensor tVsV = gmem_thr_copy_V.partition_D(sV);
            // Predicates
            Tensor cKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
            Tensor tKVcKV = gmem_thr_copy_K.partition_S(cKV);
            Tensor t0KVcKV = gmem_thr0_copy_KV.partition_S(cKV);
            Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));
            Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tVsV)));
            #pragma unroll
            for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(params.shape_K); }
            #pragma unroll
            for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(params.shape_V); }
            // Do we need bound check to make sure the row doesn't go above kBlockN
            static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
            // static_assert(EvenN);  // It simplifies the loading of K and V
            // Instead of passing in tKVcKV, we pass in t0KVcKV and subtract the offset from the limit
            // (seqlen_k - n_block * kBlockN). This is because the entries of t0KVcKV are known at compile time.
            // int const seqlenk_row_limit = -int(get<0>(tKVcKV(_0{}, _0{}, _0{}))) + (EvenN
            //     ? seqlen_info.seqlen_k - n_block * kBlockN
            //     : std::min(seqlen_info.seqlen_k - n_block * kBlockN, kBlockN));
            // // Need Clear_OOB_MN to be true here since the gemm will sum over the kBlockN dimension
            // flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clear_OOB_K=*/true>(
            //     gmem_tiled_copy_QKV, tVgV, tVsV, t0KVcKV, tKVpKV, seqlenk_row_limit);
            int const seqlenk_row_limit = seqlen_k - n_block * kBlockN - get<0>(tKVcKV(_0{}, _0{}, _0{}));
#if defined(USE_PPU) && USE_AIU
            if constexpr (Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for V.
                // Hierarchical tiling: 8x64 -> 8x128 (horizontal x2) -> blockN x128 (vertical)
                //                      -> blockN x headdim (horizontal)
                // offset(n,k) = (k/2)*kHalfSize + n*kTile8x128 + (k%2)*kAtomSize
                const Element* block_v_ptr = mV.data().get()
                    + (seqlen_info.offset_k + n_block * kBlockN) * get<0>(params.stride_V);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumNTiles = kBlockN / kAiuTileH;                   // 128/8=16
                static constexpr int kTotalTiles = kNumNTiles * kNumKTilesTotal;          // 64 or 32
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile8x128 = kAiuTileH * 2 * kAiuTileW;             // 8x128 = 1024
                static constexpr int kHalfSize = kBlockN * 2 * kAiuTileW;                // blockN x 128
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += NumMmaWarps) {
                    int const n = tile_linear / kNumKTilesTotal;
                    int const k = tile_linear % kNumKTilesTotal;
                    const Element* tile_gmem = block_v_ptr
                        + n * kAiuTileH * get<0>(params.stride_V)
                        + k * kBlockKGmem;
                    auto tile_smem = sV_AIU_COPY.data()
                        + (k/2)*kHalfSize + n*kTile8x128 + (k%2)*kAtomSize;
                    Gmem_copy_struct_KV::copy(
                        cute::raw_pointer_cast(tile_smem),
                        tile_gmem,
                        gmem_tiled_copy_V.desc_,
                        0, 0);
                }
#if 0
                // Debug: compare V data in gmem vs smem after AIU copy
                flash::cp_async_fence();
                flash::cp_async_wait<0>();
                __syncthreads();
                if (cute::thread(DEBUG_TID)) {
                    const Element* gmem_base = mV.data().get()
                        + (seqlen_info.offset_k + n_block * kBlockN) * get<0>(params.stride_V);
                    const Element* smem_base = reinterpret_cast<const Element*>(
                        cute::raw_pointer_cast(sV_AIU_COPY.data()));
                    int stride_v = get<0>(params.stride_V);

                    printf("=== V GMEM vs SMEM (n_block=%d) ===\n", (int)n_block);
                    // GMEM V: logical row-major via raw pointer
                    printf("GMEM V [%dx%d]:\n", kBlockN, kHeadDim);
                    for (int r = 0; r < kBlockN; ++r) {
                        printf("row %d: ", r);
                        for (int c = 0; c < kHeadDim; ++c)
                            printf("%.3f ", (float)gmem_base[r * stride_v + c]);
                        printf("\n");
                    }
                    // SMEM V: print as (n,k) 8x64 tiles using hierarchical tiling
                    printf("SMEM V [%dx%d] (%dx%d tiles, hierarchical):\n", kBlockN, kHeadDim, kNumNTiles, kNumKTilesTotal);
                    for (int nt = 0; nt < kNumNTiles; ++nt) {
                        for (int kt = 0; kt < kNumKTilesTotal; ++kt) {
                            int const off_base = (kt/2)*kHalfSize + nt*kTile8x128 + (kt%2)*kAtomSize;
                            printf("SMEM V tile (n=%d,k=%d) off=%d [%dx%d]:\n", nt, kt, off_base, kAiuTileH, kBlockKGmem);
                            for (int r = 0; r < kAiuTileH; ++r) {
                                for (int c = 0; c < kBlockKGmem; ++c) {
                                    printf("%.3f ", (float)smem_base[off_base + r * kBlockKGmem + c]);
                                }
                                printf("\n");
                            }
                        }
                    }
                }
                __syncthreads();
#endif
            } else {
                flash::copy</*Is_even_MN=*/true>(
                    gmem_tiled_copy_V, tVgV, tVsV, t0KVcKV, tVpV);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tVsV); ++m) {
                // If kBlockN doesn't evenly divide the tiled copy, only the last `m` needs to be checked
                if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
                    bool const predicate_n = get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tVsV); ++k) {
                        cute::copy(gmem_tiled_copy_V.with(tVpV(k) && predicate_n), tVgV(_, m, k), tVsV(_, m, k));
                    }
                }
            }
#endif
            if constexpr (V_in_regs) { flash::cp_async_fence(); }
            // flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clear_OOB_K=*/true>(
            //     gmem_tiled_copy_QKV, tKgK, tKsK, t0KVcKV, tKVpKV, seqlenk_row_limit);
#if defined(USE_PPU) && USE_AIU
            if constexpr (Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for K.
                // Hierarchical tiling: 8x64 -> 8x128 (horizontal x2) -> blockN x128 (vertical)
                //                      -> blockN x headdim (horizontal)
                // offset(n,k) = (k/2)*kHalfSize + n*kTile8x128 + (k%2)*kAtomSize
                const Element* block_k_ptr = mK.data().get()
                    + (seqlen_info.offset_k + n_block * kBlockN) * get<0>(params.stride_K);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumNTiles = kBlockN / kAiuTileH;                   // 128/8=16
                static constexpr int kTotalTiles = kNumNTiles * kNumKTilesTotal;          // 64 or 32
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile8x128 = kAiuTileH * 2 * kAiuTileW;             // 8x128 = 1024
                static constexpr int kHalfSize = kBlockN * 2 * kAiuTileW;                // blockN x 128
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += NumMmaWarps) {
                    int const n = tile_linear / kNumKTilesTotal;
                    int const k = tile_linear % kNumKTilesTotal;
                    const Element* tile_gmem = block_k_ptr
                        + n * kAiuTileH * get<0>(params.stride_K)
                        + k * kBlockKGmem;
                    auto tile_smem = sK_AIU_COPY.data()
                        + (k/2)*kHalfSize + n*kTile8x128 + (k%2)*kAtomSize;
                    Gmem_copy_struct_KV::copy(
                        cute::raw_pointer_cast(tile_smem),
                        tile_gmem,
                        gmem_tiled_copy_K.desc_,
                        0, 0);
                }
#if 0
                // Debug: compare K data in gmem vs smem after AIU copy
                flash::cp_async_fence();
                flash::cp_async_wait<0>();
                __syncthreads();
                if (cute::thread(DEBUG_TID)) {
                    const Element* gmem_base = mK.data().get()
                        + (seqlen_info.offset_k + n_block * kBlockN) * get<0>(params.stride_K);
                    const Element* smem_base = reinterpret_cast<const Element*>(
                        cute::raw_pointer_cast(sK_AIU_COPY.data()));
                    int stride_k = get<0>(params.stride_K);

                    printf("=== K GMEM vs SMEM (n_block=%d) ===\n", (int)n_block);
                    // GMEM K: logical row-major via raw pointer
                    printf("GMEM K [%dx%d]:\n", kBlockN, kHeadDim);
                    for (int r = 0; r < 32; ++r) {
                        printf("row %d: ", r);
                        for (int c = 0; c < kHeadDim; ++c)
                            printf("%.3f ", (float)gmem_base[r * stride_k + c]);
                        printf("\n");
                    }
                    // // SMEM K: print as (n,k) 8x64 tiles using hierarchical tiling
                    // printf("SMEM K [%dx%d] (%dx%d tiles, hierarchical):\n", kBlockN, kHeadDim, kNumNTiles, kNumKTilesTotal);
                    // for (int nt = 0; nt < 8; ++nt) {
                    //     for (int kt = 0; kt < kNumKTilesTotal; ++kt) {
                    //         int const off_base = (kt/2)*kHalfSize + nt*kTile8x128 + (kt%2)*kAtomSize;
                    //         printf("SMEM K tile (n=%d,k=%d) off=%d [%dx%d]:\n", nt, kt, off_base, kAiuTileH, kBlockKGmem);
                    //         for (int r = 0; r < kAiuTileH; ++r) {
                    //             for (int c = 0; c < kBlockKGmem; ++c) {
                    //                 printf("%.3f ", (float)smem_base[off_base + r * kBlockKGmem + c]);
                    //             }
                    //             printf("\n");
                    //         }
                    //     }
                    // }
                }
                __syncthreads();
#endif
            } else {
                flash::copy</*Is_even_MN=*/true>(
                    gmem_tiled_copy_K, tKgK, tKsK, t0KVcKV, tKpK);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tKsK); ++m) {
                if (EvenN || m < size<1>(tKsK) - 1 || get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
                    bool const predicate_n = get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tKsK); ++k) {
                        cute::copy(gmem_tiled_copy_K.with(tKpK(k) && predicate_n), tKgK(_, m, k), tKsK(_, m, k));
                    }
                }
            }
#endif
            flash::cp_async_fence();
        }

        if constexpr (V_in_regs) {
            flash::cp_async_wait<1>();
            __syncthreads();
            Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
            Tensor tdPsV_copy_view = smem_thr_copy_KV.partition_S(make_mix_tensor_like(sV));
            cute::copy(smem_tiled_copy_KV, tdPsV_copy_view, tdPrV_copy_view);
            __syncthreads();  // Sync to avoid loading Q to smem_q, which overlaps with smem_v
        }

        // Do we need bound check to make sure the row doesn't go above kBlockM
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr bool EvenM = kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;

        auto load_Q_LSE = [&] (int const m_block, int const smem_pipe_write) {
            // if (cute::thread0()) { printf("Inside load_Q_LSE, m_block = %d, smem_pipe_write = %d\n", m_block, smem_pipe_write); }
            Tensor tQsQ_cur = tQsQ(_, _, _, smem_pipe_write);
            Tensor tQgQ_cur = tQgQ(_, _, _, m_block);
            // Instead of passing in tQcQ, we pass in t0QcQ and subtract the offset from the limit
            // (seqlen_q - m_block * kBlockM). This is because the entries of t0QcQ are known at compile time.
            // int const seqlenq_row_limit = -int(get<0>(tQcQ(_0{}, _0{}, _0{}))) + (EvenM
            //     ? seqlen_info.seqlen_q - m_block * kBlockM
            //     : std::min(seqlen_info.seqlen_q - m_block * kBlockM, kBlockM));
            // Need Clear_OOB_MN to be true here since the gemm will sum over the kBlockM dimension
            // flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clear_OOB_K=*/true>(
            //     gmem_tiled_copy_QKV, tQgQ(_, _, _, m_block), tQsQ_cur, t0QcQ, tQpQ, seqlenq_row_limit);
            int const seqlenq_row_limit = seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}));
#if defined(USE_PPU) && USE_AIU
            if constexpr (Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for Q.
                // Hierarchical tiling: 8x64 -> 16x64 (vertical x2) -> 16x128 (horizontal x2)
                //                      -> blockM x128 (vertical) -> blockM x headdim (horizontal)
                // offset(m,k) = (k/2)*kHalfSize + (m/2)*kTile16x128 + (k%2)*kTile16x64 + (m%2)*kAtomSize
                static_assert(kBlockKGmem == 64, "AIU hierarchical layout assumes 8x64 atom");
                const Element* block_q_ptr = mQ.data().get()
                    + (seqlen_info.offset_q + m_block * kBlockM) * get<0>(params.stride_Q);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumMTiles = kBlockM / kAiuTileH;                   // 64/8=8
                static constexpr int kTotalTiles = kNumMTiles * kNumKTilesTotal;          // 32 or 16
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile16x64 = 2 * kAtomSize;                         // 16x64 = 1024
                static constexpr int kTile16x128 = 2 * kTile16x64;                       // 16x128 = 2048
                static constexpr int kHalfSize = kBlockM * 2 * kAiuTileW;                // blockM x 128
                static constexpr int kLog2NumKTiles = __builtin_ctz(kNumKTilesTotal);
                static constexpr int kLog2BlockKGmem = __builtin_ctz(kBlockKGmem);
                static constexpr int kLog2AtomSize = __builtin_ctz(kAtomSize);
                static constexpr int kLog2Tile16x64 = __builtin_ctz(kTile16x64);
                static constexpr int kLog2Tile16x128 = __builtin_ctz(kTile16x128);
                // NumMmaWarps is always a multiple of kNumKTilesTotal, so k is loop-invariant
                // and m increments by kMStride (always even) => m&1 is also loop-invariant.
                static constexpr int kMStride = NumMmaWarps >> kLog2NumKTiles;
                static_assert(kMStride % 2 == 0, "kMStride must be even for m&1 loop-invariance");
                static constexpr int kSmemMHalfDelta = (kMStride >> 1) << kLog2Tile16x128;

                auto const gmem_m_stride_Q = kAiuTileH * get<0>(params.stride_Q);

                // Precompute loop-invariant offsets; scope k/m_init to free registers
                auto smem_loop_base = sQ_AIU_COPY.data() + smem_pipe_write * kBlockM * kHeadDim;
                int smem_m_half_offset;
                const Element* tile_gmem;
                {
                    int const k = warp_idx & (kNumKTilesTotal - 1);
                    int const m_init = warp_idx >> kLog2NumKTiles;
                    smem_loop_base = smem_loop_base + ((k >> 1) * kHalfSize + ((k & 1) << kLog2Tile16x64) + ((m_init & 1) << kLog2AtomSize));
                    smem_m_half_offset = (m_init >> 1) << kLog2Tile16x128;
                    tile_gmem = block_q_ptr
                        + (k << kLog2BlockKGmem)
                        + static_cast<int64_t>(m_init) * gmem_m_stride_Q;
                }

                // Loop body: only 1 smem add + copy + 2 constant increments
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += NumMmaWarps) {
                    auto tile_smem = smem_loop_base + smem_m_half_offset;
                    Gmem_copy_struct_Q::copy(
                        cute::raw_pointer_cast(tile_smem),
                        tile_gmem,
                        gmem_tiled_copy_Q.desc_,
                        0, 0);
                    tile_gmem += static_cast<int64_t>(kMStride) * gmem_m_stride_Q;
                    smem_m_half_offset += kSmemMHalfDelta;
                }
#if 0
                flash::cp_async_fence();
                flash::cp_async_wait<0>();
                __syncthreads();
                if (cute::thread(DEBUG_TID)) {
                    // ---- Q AIU COPY address overlap check with smem_lse (0x19000) ----
                    printf("==== [Q AIU COPY address] (thread%d) ====\n", threadIdx.x);
                    printf("sQ_AIU_COPY base = %p\n", (void*)sQ_AIU_COPY.data().get());
                    printf("smem_pipe_write = %d\n", smem_pipe_write);
                    auto debug_q_aiu_stage_base = sQ_AIU_COPY.data().get() + smem_pipe_write * kBlockM * kHeadDim;
                    printf("Q AIU stage_base = %p\n", (void*)debug_q_aiu_stage_base);
                    printf("Q AIU write range: [%p, %p)\n",
                        (void*)debug_q_aiu_stage_base,
                        (void*)(debug_q_aiu_stage_base + kBlockM * kHeadDim));
                    printf("Q AIU write size (bytes) = %d\n", (int)(kBlockM * kHeadDim * sizeof(Element)));
                    printf("sLSE base = %p\n", (void*)sLSE.data().get());
                    printf("========================================\n");
                    // ---- end address overlap check ----
                    printf("=== Q GMEM vs SMEM (m_block=%d, smem_pipe_write=%d) ===\n", m_block, smem_pipe_write);
                    printf("sQ_AIU_COPY layout: ");
                    print(sQ_AIU_COPY.layout());
                    printf("\n");

                    // GMEM Q: logical row-major via raw pointer
                    const Element* gmem_q_base = mQ.data().get()
                        + (seqlen_info.offset_q + m_block * kBlockM) * get<0>(params.stride_Q);
                    int stride_q = get<0>(params.stride_Q);
                    printf("GMEM Q [%dx%d]:\n", kBlockM, kHeadDim);
                    for (int r = 0; r < kBlockM; ++r) {
                        printf("row %d: ", r);
                        for (int c = 0; c < kHeadDim; ++c)
                            printf("%.3f ", (float)gmem_q_base[r * stride_q + c]);
                        printf("\n");
                    }

                    // // SMEM Q: full tile via sQ (NOT tQsQ_cur which is only thread0's partition)
                    // // CuTe maps logical (r,c) to the correct physical address via SmemLayoutQ
                    // Tensor sQ_cur = sQ(_, _, smem_pipe_write);
                    // print(sQ_cur);
                    // print("\n");
                    // printf("SMEM Q [%dx%d]:\n", kBlockM, kHeadDim);

                    // // SMEM Q: physical 8x64 blocks via raw pointer (like K debug)
                    // const Element* smem_q_base = reinterpret_cast<const Element*>(
                    //     cute::raw_pointer_cast(sQ.data())) + smem_pipe_write * kBlockM * kHeadDim;
                    // for (int mt = 0; mt < kNumMTiles; ++mt) {
                    //     for (int kt = 0; kt < kNumKTilesTotal; ++kt) {
                    //         int const off_base = (kt/2)*kHalfSize + (mt/2)*kTile16x128 + (kt%2)*kTile16x64 + (mt%2)*kAtomSize;
                    //         printf("SMEM Q tile (m=%d,k=%d) off=%d [%dx%d]:\n", mt, kt, off_base, kAiuTileH, kAiuTileW);
                    //         for (int r = 0; r < kAiuTileH; ++r) {
                    //             for (int c = 0; c < kBlockKGmem; ++c) {
                    //                 int off = off_base + r * kBlockKGmem + c;
                    //                 printf("%.3f ", (float)smem_q_base[off]);
                    //             }
                    //             printf("\n");
                    //         }
                    //     }
                    // }
                }
                __syncthreads();
#endif
            } else {
                flash::copy</*Is_even_MN=*/true>(
                    gmem_tiled_copy_Q, tQgQ_cur, tQsQ_cur, t0QcQ, tQpQ);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tQsQ); ++m) {
                // If kBlockM doesn't evenly divide the tiled copy, only the last `m` needs to be checked
                if (EvenM || m < size<1>(tQsQ) - 1 || get<0>(tQcQ(_0{}, m, _0{})) < kBlockM) {
                    bool const predicate_m = get<0>(t0QcQ(_0{}, m, _0{})) < seqlenq_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tQsQ); ++k) {
                        cute::copy(gmem_tiled_copy_Q.with(tQpQ(k) && predicate_m), tQgQ_cur(_, m, k), tQsQ_cur(_, m, k));
                    }
                }
            }
#endif
            Tensor tLSEgLSE_cur = tLSEgLSE(_, _, m_block);
            Tensor tLSEsLSE_cur = tLSEsLSE(_, _, smem_pipe_write);
            // We made sure LSE length is padded so we read `kBlockM` elements so that all
            // elements in sLSE are filled. Without this we might have uninitialized sLSE values.
#if 0
            if (cute::thread(DEBUG_TID)) {
                printf("==== [gLSE before copy] (thread%d) ====\n", threadIdx.x);
                print(tLSEgLSE_cur.layout()); printf("\n");
                const int total_g = int(size(tLSEgLSE_cur));
                for (int i = 0; i < total_g; ++i) {
                    printf("thread%d[gLSE][%d] = %f\n", threadIdx.x, i, static_cast<float>(tLSEgLSE_cur(i)));
                }
            }
#endif
            #pragma unroll
            for (int m = 0; m < size<1>(tLSEsLSE); ++m) {
                if (get<0>(tLSEcLSE(_0{}, m)) < kBlockM) {
                    cute::copy(gmem_tiled_copy_lse, tLSEgLSE_cur(_, m), tLSEsLSE_cur(_, m));
                }
            }
#if 0
            cute::cp_async_fence();
            flash::cp_async_wait<0>();
            __syncthreads();
            if (cute::thread(DEBUG_TID)) {
                printf("==== [sLSE after copy] (thread%d) ====\n", threadIdx.x);
                print(tLSEsLSE_cur.layout()); printf("\n");
                const int total_s = int(size(tLSEsLSE_cur));
                for (int i = 0; i < total_s; ++i) {
                    printf("thread%d[sLSE][%d] = %f\n", threadIdx.x, i, static_cast<float>(tLSEsLSE_cur(i)));
                }
            }
#endif
        };

        auto load_dO_dPsum = [&] (int const m_block, int const smem_pipe_write) {
            // if (cute::thread0()) { printf("Inside load_dO_dPsum, m_block = %d, smem_pipe_write = %d\n", m_block, smem_pipe_write); }
            Tensor tdOsdO_cur = tdOsdO(_, _, _, smem_pipe_write);
            Tensor tdOgdO_cur = tdOgdO(_, _, _, m_block);
            // int const seqlenq_row_limit = -int(get<0>(tQcQ(_0{}, _0{}, _0{}))) + (EvenM
            //     ? seqlen_info.seqlen_q - m_block * kBlockM
            //     : std::min(seqlen_info.seqlen_q - m_block * kBlockM, kBlockM));
            // flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clear_OOB_K=*/true>(
            //     gmem_tiled_copy_QKV, tdOgdO(_, _, _, m_block), tdOsdO_cur, t0QcQ, tQpQ, seqlenq_row_limit);
            int const seqlenq_row_limit = seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}));
#if defined(USE_PPU) && USE_AIU
            if constexpr (Use_CVT_SWZL_LD) {
                // Direct 8x64 tile-granularity AIU copy for dO.
                // Hierarchical tiling: 8x64 -> 16x64 (vertical x2) -> 16x128 (horizontal x2)
                //                      -> blockM x128 (vertical) -> blockM x headdim (horizontal)
                // offset(m,k) = (k/2)*kHalfSize + (m/2)*kTile16x128 + (k%2)*kTile16x64 + (m%2)*kAtomSize
                const Element* block_do_ptr = mdO.data().get()
                    + (seqlen_info.offset_q + m_block * kBlockM) * get<0>(params.stride_dO);
                static constexpr int kAiuTileH = 8;
                static constexpr int kAiuTileW = 64;                                     // AIU atom width
                static constexpr int kNumKTilesTotal = kHeadDim / kAiuTileW;             // 4 for hdim256, 2 for hdim128
                static constexpr int kNumMTiles = kBlockM / kAiuTileH;                   // 64/8=8
                static constexpr int kTotalTiles = kNumMTiles * kNumKTilesTotal;          // 32 or 16
                static constexpr int kAtomSize = kAiuTileH * kAiuTileW;                  // 8x64 = 512
                static constexpr int kTile16x64 = 2 * kAtomSize;                         // 16x64 = 1024
                static constexpr int kTile16x128 = 2 * kTile16x64;                       // 16x128 = 2048
                static constexpr int kHalfSize = kBlockM * 2 * kAiuTileW;                // blockM x 128
                static constexpr int kLog2NumKTiles = __builtin_ctz(kNumKTilesTotal);
                static constexpr int kLog2BlockKGmem = __builtin_ctz(kBlockKGmem);
                static constexpr int kLog2AtomSize = __builtin_ctz(kAtomSize);
                static constexpr int kLog2Tile16x64 = __builtin_ctz(kTile16x64);
                static constexpr int kLog2Tile16x128 = __builtin_ctz(kTile16x128);
                // NumMmaWarps is always a multiple of kNumKTilesTotal, so k is loop-invariant
                // and m increments by kMStride (always even) => m&1 is also loop-invariant.
                static constexpr int kMStride = NumMmaWarps >> kLog2NumKTiles;
                static_assert(kMStride % 2 == 0, "kMStride must be even for m&1 loop-invariance");
                static constexpr int kSmemMHalfDelta = (kMStride >> 1) << kLog2Tile16x128;

                auto const gmem_m_stride_dO = kAiuTileH * get<0>(params.stride_dO);

                // Precompute loop-invariant offsets; scope k/m_init to free registers
                auto smem_loop_base = sdO_AIU_COPY.data() + smem_pipe_write * kBlockM * kHeadDim;
                int smem_m_half_offset;
                const Element* tile_gmem;
                {
                    int const k = warp_idx & (kNumKTilesTotal - 1);
                    int const m_init = warp_idx >> kLog2NumKTiles;
                    smem_loop_base = smem_loop_base + ((k >> 1) * kHalfSize + ((k & 1) << kLog2Tile16x64) + ((m_init & 1) << kLog2AtomSize));
                    smem_m_half_offset = (m_init >> 1) << kLog2Tile16x128;
                    tile_gmem = block_do_ptr
                        + (k << kLog2BlockKGmem)
                        + static_cast<int64_t>(m_init) * gmem_m_stride_dO;
                }

                // Loop body: only 1 smem add + copy + 2 constant increments
                #pragma unroll
                for (int tile_linear = warp_idx; tile_linear < kTotalTiles; tile_linear += NumMmaWarps) {
                    auto tile_smem = smem_loop_base + smem_m_half_offset;
                    Gmem_copy_struct_Q::copy(
                        cute::raw_pointer_cast(tile_smem),
                        tile_gmem,
                        gmem_tiled_copy_dO.desc_,
                        0, 0);
                    tile_gmem += static_cast<int64_t>(kMStride) * gmem_m_stride_dO;
                    smem_m_half_offset += kSmemMHalfDelta;
                }
#if 0
                flash::cp_async_fence();
                flash::cp_async_wait<0>();
                __syncthreads();
                if (cute::thread(DEBUG_TID)) {
                    printf("=== dO GMEM vs SMEM (m_block=%d, smem_pipe_write=%d) ===\n", m_block, smem_pipe_write);
                    // GMEM dO
                    const Element* gmem_do_base = mdO.data().get()
                        + (seqlen_info.offset_q + m_block * kBlockM) * get<0>(params.stride_dO);
                    int stride_do = get<0>(params.stride_dO);
                    printf("GMEM dO [%dx%d]:\n", kBlockM, kHeadDim);
                    for (int r = 0; r < kBlockM; ++r) {
                        printf("row %d: ", r);
                        for (int c = 0; c < kHeadDim; ++c)
                            printf("%.3f ", (float)gmem_do_base[r * stride_do + c]);
                        printf("\n");
                    }
                    // SMEM dO: physical tiles via raw pointer (hierarchical tiling)
                    const Element* smem_do_base = reinterpret_cast<const Element*>(
                        cute::raw_pointer_cast(sdO_AIU_COPY.data())) + smem_pipe_write * kBlockM * kHeadDim;
                    printf("SMEM dO [%dx%d] (%dx%d tiles, hierarchical):\n", kBlockM, kHeadDim, kNumMTiles, kNumKTilesTotal);
                    for (int mt = 0; mt < kNumMTiles; ++mt) {
                        for (int kt = 0; kt < kNumKTilesTotal; ++kt) {
                            int const off_base = (kt/2)*kHalfSize + (mt/2)*kTile16x128 + (kt%2)*kTile16x64 + (mt%2)*kAtomSize;
                            printf("SMEM dO tile (m=%d,k=%d) off=%d [%dx%d]:\n", mt, kt, off_base, kAiuTileH, kAiuTileW);
                            for (int r = 0; r < kAiuTileH; ++r) {
                                for (int c = 0; c < kBlockKGmem; ++c) {
                                    int off = off_base + r * kBlockKGmem + c;
                                    printf("%.3f ", (float)smem_do_base[off]);
                                }
                                printf("\n");
                            }
                        }
                    }
                }
                __syncthreads();
#endif
            } else {
                flash::copy</*Is_even_MN=*/true>(
                    gmem_tiled_copy_dO, tdOgdO_cur, tdOsdO_cur, t0QcQ, tQpQ);
            }
#else
            #pragma unroll
            for (int m = 0; m < size<1>(tdOsdO); ++m) {
                // If kBlockM doesn't evenly divide the tiled copy, only the last `m` needs to be checked
                if (EvenM || m < size<1>(tdOsdO) - 1 || get<0>(tQcQ(_0{}, m, _0{})) < kBlockM) {
                    bool const predicate_m = get<0>(t0QcQ(_0{}, m, _0{})) < seqlenq_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tdOsdO); ++k) {
                        cute::copy(gmem_tiled_copy_dO.with(tdOpdO(k) && predicate_m), tdOgdO_cur(_, m, k), tdOsdO_cur(_, m, k));
                    }
                }
            }
#endif
            Tensor tLSEgdPsum_cur = tLSEgdPsum(_, _, m_block);
            Tensor tLSEsdPsum_cur = tLSEsdPsum(_, _, smem_pipe_write);
            #pragma unroll
            for (int m = 0; m < size<1>(tLSEsdPsum); ++m) {
                if (get<0>(tLSEcLSE(_0{}, m)) < kBlockM) {
                    cute::copy(gmem_tiled_copy_lse, tLSEgdPsum_cur(_, m), tLSEsdPsum_cur(_, m));
                }
            }
        };

        int m_block = m_block_min;

        // Note, using the for_each() function here to ensure `stage` is of type Int<x>.
        for_each(make_int_sequence<kStages>{}, [&] (auto stage) {
            static constexpr bool Is_first_stage = CUTE_STATIC_V(stage) == 0;
            static constexpr bool Is_last_stage = CUTE_STATIC_V(stage) == kStages - 1;
            if constexpr (!Is_last_stage || kStages == 1) {
                if (Is_first_stage || m_block + stage < m_block_max) {
                    load_Q_LSE(m_block + stage, stage);
                }
            }
            // We want the fence outside the if statement to have a fixed number of cp.async commits.
            // so that we can wait with the correct number of outstanding commits.
            cute::cp_async_fence();
            if constexpr (stage < kStages_dO) {
                if (Is_first_stage || m_block + stage < m_block_max) {
                    load_dO_dPsum(m_block + stage, stage);
                }
                cute::cp_async_fence();
            }
        });

        int smem_pipe_read = 0, smem_pipe_read_do = 0, smem_pipe_write = kStages - 1, smem_pipe_write_do = 0;

        auto load_Q_next = [&] {
            // if (cute::thread0()) { printf("m_block = %d, m_block_max = %d, smem_pipe_write = %d\n", m_block, m_block_max, smem_pipe_write); }
            if (m_block + (kStages > 1 ? kStages - 1 : 1) < m_block_max) {
                load_Q_LSE(m_block + (kStages > 1 ? kStages - 1 : 1), kStages > 1 ? smem_pipe_write : 0);
            }
            cute::cp_async_fence();
        };

        auto load_dO_next = [&] {
            // int smem_pipe_write_do_cur = Q_dO_same_stages ? smem_pipe_write : smem_pipe_write_do;
            if (m_block + kStages_dO < m_block_max) {
                // load_dO_dPsum(m_block + kStages_dO, kStages_dO > 1 ? smem_pipe_write_do_cur : 0);
                load_dO_dPsum(m_block + kStages_dO, kStages_dO > 1 ? smem_pipe_write_do : 0);
            }
            cute::cp_async_fence();
        };

        clear(tdKrdK);
        clear(tdVrdV);

        auto bwd_step = [&](int m_block, auto mask_fn) {
            Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            clear(tSrS);
            flash::cp_async_wait<(kStages > 1) ? 1 : 0>();
            __syncthreads();
#if 0
            // Debug: verify Q SMEM and LSE data at each m_block iteration
            if (cute::thread0() && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                printf("\n==== [bwd_step] m_block=%d, smem_pipe_read=%d, smem_pipe_write=%d ====\n",
                       m_block, smem_pipe_read, smem_pipe_write);
                // Print first 8 values of Q from SMEM (physical layout)
                const Element* smem_q_base = reinterpret_cast<const Element*>([&]() -> auto {
                    if constexpr (Use_CVT_SWZL_LD) {
                        return cute::raw_pointer_cast(sQ_AIU_COPY.data());
                    } else {
                        return cute::raw_pointer_cast(sQ.data());
                    }
                }()) + (kStages > 1 ? smem_pipe_read : 0) * kBlockM * kHeadDim;
                printf("Q SMEM first 8 vals: ");
                for (int i = 0; i < 8; ++i) {
                    printf("%.4f ", (float)smem_q_base[i]);
                }
                printf("\n");
                // Print LSE first 4 values
                printf("LSE SMEM first 4 vals: ");
                for (int i = 0; i < 4; ++i) {
                    printf("%f ", (float)tSsLSE(i, kStages > 1 ? smem_pipe_read : 0));
                }
                printf("\n");
            }
            __syncthreads();
#endif
            Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(thr_mma_SdP, sQ(_, _, _0{}));
            Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(thr_mma_SdP, sK);
            // if (cute::thread0()) { print(tiled_mma_SdP); print(tSrS); printf("\n"); print(tSrQ); printf("\n"); print(tSrK); printf("\n"); print(tSsQ); printf("\n"); print(tSsK); printf("\n"); }
            /** recompute s=Q*K^T */
            flash::gemm_sm80<false /*A_in_regs*/, false /*B_in_regs*/, SdP_swapAB>(
                tSrS, tSrQ, tSrK, tSsQ(_, _, _, kStages > 1 ? smem_pipe_read : 0), tSsK,
                tiled_mma_SdP, smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV, nullptr /*hook*/);
#if 0
                if (cute::thread(DEBUG_TID)) {
                    printf("==== [tSrS after GEMM Q*K^T] (thread%d) ====\n", threadIdx.x);
                    print(tSrS.layout()); printf("\n");
                    const int total = int(size(tSrS));
                    for (int i = 0; i < total; ++i) {
                        printf("thread%d[tSrS_afterGEMM][%d] = %f\n", threadIdx.x, i, static_cast<float>(tSrS(i)));
                    }
                }
#endif
            Tensor tLSErLSE = cute::conditional_return<!ShuffleLSE>(make_fragment_like(tSsLSE(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
            if constexpr (!ShuffleLSE) {
                cute::copy(tSsLSE(_, kStages > 1 ? smem_pipe_read : 0), tLSErLSE);
            } else {
                #pragma unroll
                for (int i = 0; i < kStatsPerThread; ++i) {
                    // It's ok to read OOB, since we made sure sLSE is large enough and we won't use the OOB values
                    tLSErLSE(i) = tSsLSE((thread_idx % 32) / 4 + i * 8, kStages > 1 ? smem_pipe_read : 0);
                }
            }
#if 0
                if (cute::thread(DEBUG_TID)) {
                    printf("==== [tSsLSE raw values] (thread%d) ====\n", threadIdx.x);
                    print(tSsLSE.layout()); printf("\n");
                    printf("tSsLSE data ptr = %p\n", (void*)tSsLSE.data().get());
                    const int total_lse = int(size(tSsLSE));
                    for (int i = 0; i < min(total_lse, 20); ++i) {
                        printf("thread%d[tSsLSE][%d] = %f\n", threadIdx.x, i, static_cast<float>(tSsLSE(i)));
                    }
                    printf("==== [tLSErLSE after load] (thread%d) ====\n", threadIdx.x);
                    print(tLSErLSE.layout()); printf("\n");
                    const int total = int(size(tLSErLSE));
                    for (int i = 0; i < total; ++i) {
                        printf("thread%d[tLSErLSE][%d] = %f\n", threadIdx.x, i, static_cast<float>(tLSErLSE(i)));
                    }
                }
#endif
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, params.softcap_val); }

            // Reshape tSrS from (4, MMA_N, MMA_M) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));
            // dtanh needs to happen before masking, otherwise we get 1 - (-inf)^2 = NaN in the dtanh
            // if (cute::thread0()) { print_tensor(scores); }
            auto dtanh = [&] { if constexpr (Has_softcap) return flash::calculate_dtanh(scores); else return nullptr; }();
#if 0
                if (cute::thread(DEBUG_TID)) {
                    printf("==== [scores after reshape] (thread%d) ====\n", threadIdx.x);
                    print(scores.layout()); printf("\n");
                    const int total = int(size(scores));
                    for (int i = 0; i < total; ++i) {
                        printf("thread%d[scores][%d] = %f\n", threadIdx.x, i, static_cast<float>(scores(i)));
                    }
                }
#endif
            mask_fn(tSrS, m_block);
#if 0
                if (cute::thread(DEBUG_TID)) {
                    printf("==== [tSrS after mask] (thread%d) ====\n", threadIdx.x);
                    print(tSrS.layout()); printf("\n");
                    const int total = int(size(tSrS));
                    for (int i = 0; i < total; ++i) {
                        printf("thread%d[tSrS_afterMask][%d] = %f\n", threadIdx.x, i, static_cast<float>(tSrS(i)));
                    }
                }
#endif

            /** recompute P = softmax(S)*/
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); ++mi) {
                float const lse_scaled = [&] {
                    if constexpr (!ShuffleLSE) return tLSErLSE(mi);
                    else return __shfl_sync(0xffffffff, tLSErLSE(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                }();
                #pragma unroll
                for (int ni = 0; ni < size<1>(scores); ++ni) {
                    scores(mi, ni) = exp2f(scores(mi, ni) * params.softmax_scale_log2 - lse_scaled);
                }
            }

            /** dP = dO*V^T  */
            Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            clear(tdPrdP);
            int smem_pipe_read_do_cur = Q_dO_same_stages ? smem_pipe_read : smem_pipe_read_do;
            flash::cp_async_wait<(kStages_dO > 1) ? 1 : 0>();
            __syncthreads();
            auto hook = cute::conditional_return<(kStages > 1)>(load_Q_next, nullptr);
            Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(thr_mma_SdP, sdO(_, _, _0{}));
            Tensor tdPrV_cur = cute::conditional_return<V_in_regs>(tdPrV, mma_partition_fragment_AB</*A=*/SdP_swapAB>(thr_mma_SdP, sV));
            flash::gemm_sm80<false /*A_in_regs*/, V_in_regs, SdP_swapAB>(
                tdPrdP, tdPrdO, tdPrV_cur, tdPsdO(_, _, _, kStages_dO > 1 ? smem_pipe_read_do_cur : 0), tdPsV,
                tiled_mma_SdP, smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV, hook);
            Tensor tLSErdPsum = cute::conditional_return<!ShuffledPsum>(make_fragment_like(tSsdPsum(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
            /** D=rowsum(dO) step1: reduce row data inside a warp range*/
            if constexpr (!ShuffledPsum) {
                cute::copy(tSsdPsum(_, kStages_dO > 1 ? smem_pipe_read_do_cur : 0), tLSErdPsum);
            } else {
                #pragma unroll
                for (int i = 0; i < kStatsPerThread; ++i) {
                    tLSErdPsum(i) = tSsdPsum((thread_idx % 32) / 4 + i * 8, kStages_dO > 1 ? smem_pipe_read_do_cur : 0);
                }
            }

            // Reshape tdPrdP from (4, MMA_N, MMA_M) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
             /** D=rowsum(dO) step2: __shfl_sync warp dreduce*/
            Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                float const dP_sum_cur = [&] {
                    if constexpr (!ShuffledPsum) return tLSErdPsum(mi);
                    else return __shfl_sync(0xffffffff, tLSErdPsum(mi / 8), (mi % 8) * 4 + (thread_idx % 4));
                }();
                #pragma unroll
                for (int ni = 0; ni < size<1>(dS); ++ni) {
                    dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum_cur);
                    if constexpr (Has_softcap) { dS(mi, ni) *= dtanh(mi, ni); }
                }
            }

            // Convert scores from fp32 to fp16/bf16
            Tensor rP = make_tensor_like<Element>(tSrS);
            flash::convert_type_out(tSrS, rP);
            if constexpr (!Mma_dKV_is_RS) {
                Tensor tPaP = r2s_thr_copy_PdS.retile_S(rP);  // ((Atom,AtomNum), MMA_N, MMA_N)
#if 0
                if (cute::thread(DEBUG_TID)) {
                    printf("==== tPaP reg debug (thread%d) ====\n", DEBUG_TID);
                    print(tPaP.layout()); printf("\n");
                    const int total = int(size(tPaP));
                    for (int i = 0; i < total; ++i) {
                        printf("tPaP[%d] = %f\n", i, static_cast<float>(tPaP(i)));
                    }
                }
#endif
                cute::copy(r2s_tiled_copy_PdS, tPaP, tPsP);
            }
            Tensor rdS = make_tensor_like<Element>(tdPrdP);
            flash::convert_type_out(tdPrdP, rdS);
            if constexpr (!Mma_dKV_is_RS) { __syncthreads(); }  // Make sure P is written
            // For hdim 64, It's faster to write to smem_dS first before the dV gemm
            Tensor tdSadS = r2s_thr_copy_PdS.retile_S(rdS);   // ((Atom,AtomNum), MMA_N, MMA_N)
            cute::copy(r2s_tiled_copy_PdS, tdSadS, tdSsdS);

#if 0
            __syncthreads();
            if (cute::thread(DEBUG_TID)) {
                printf("==== sP smem debug (thread0) ====\n");
                const Element* sP_ptr = reinterpret_cast<const Element*>(sP.data().get());
                constexpr int kTileR = 8;
                constexpr int kTileC = 64;
                constexpr int kNumTileM = kBlockM / kTileR;
                constexpr int kNumTileN = kBlockN / kTileC;
                constexpr int kNumBlocks = kNumTileM * kNumTileN;
                // for (int blk = 0; blk < kNumBlocks; ++blk) {
                //     int tm = blk / kNumTileN;
                //     int tn = blk % kNumTileN;
                //     printf("-- sP block %d (tile %d,%d) [8x64] --\n", blk, tm, tn);
                //     for (int r = 0; r < kTileR; ++r) {
                //         for (int c = 0; c < kTileC; ++c) {
                //             printf("%f ", static_cast<float>(sP_ptr[blk * kTileR * kTileC + r * kTileC + c]));
                //         }
                //         printf("\n");
                //     }
                //     printf("\n");
                // }
                printf("==== sdS smem debug (thread0) ====\n");
                const Element* sdS_ptr = reinterpret_cast<const Element*>(sdS.data().get());
                for (int blk = 0; blk < kNumBlocks; ++blk) {
                    int tm = blk / kNumTileN;
                    int tn = blk % kNumTileN;
                    printf("-- sdS block %d (tile %d,%d) [8x64] --\n", blk, tm, tn);
                    for (int r = 0; r < kTileR; ++r) {
                        for (int c = 0; c < kTileC; ++c) {
                            printf("%.3f ", static_cast<float>(sdS_ptr[blk * kTileR * kTileC + r * kTileC + c]));
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
            }
            __syncthreads();
#endif
            /** dV = P^T*dO */
            Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(thr_mma_dKV, sdOt(_, _, _0{}));
            Tensor tdVsdO_cur = tdVsdOt(_, _, _, kStages_dO > 1 ? smem_pipe_read_do_cur : 0);
            if constexpr (Mma_dKV_is_RS) {
#ifdef USE_PPU
                Tensor tdVrP_acc = [&]() -> auto {
                    if constexpr (ArchTag::kMinComputeCapability >= 89) {
                        return make_tensor_like<Element>(tSrS);
                    } else {
                        return flash::convert_acc<Element>(tSrS);
                    }
                }();
                Tensor tdVrP = [&]() -> auto {
                    if constexpr (ArchTag::kMinComputeCapability >= 89) {
                        return make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
                    } else {
                        return make_tensor(tdVrP_acc.data(), make_layout(get<0>(tSrQ.layout()), get<1>(tSrS.layout()), get<2>(tSrS.layout())));
                    }
                }();
#else
                Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
#endif
                flash::gemm_rs_sm80(tdVrdV, tdVrP, tdVrdO, tdVsdO_cur, tiled_mma_dKV, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
            } else {
                Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(thr_mma_dKV, sPt);
                flash::gemm_sm80<false /*A_in_regs*/, false /*B_in_regs*/, /*SwapAB=*/dKV_swapAB>(
                    tdVrdV, tdVrP, tdVrdO, tdVsPt, tdVsdO_cur,
                    tiled_mma_dKV, smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt, nullptr);
            }
            // if (cute::thread0()) { print_tensor(tdVrdV); }
            __syncthreads();  // make sure sdS is written
            /** dQ = dS*K */
            auto do_mma_dQ = [&] (auto hook) {
                Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                clear(tdQrdQ);
                Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(thr_mma_dQ, sdS);
                Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(thr_mma_dQ, sKt);
                flash::gemm_sm80<false /*A_in_regs*/, false /*B_in_regs*/, /*SwapAB=*/dQ_swapAB>(
                    tdQrdQ, tdQrdS, tdQrK, tdQsdS, tdQsKt, tiled_mma_dQ,
                    // smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt, load_dO_next);
                    smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt, hook);
                // if (cute::thread0()) { print_tensor(tdQrdQ); }
                // We can reuse r2s_thr_copy_dQaccum for this partitioning
                Tensor tdQrdQ_atomic = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
                static_assert(CUTE_STATIC_V(size(tdQrdQ_atomic)) == CUTE_STATIC_V(size(tdQgdQaccum(_, _, 0))));
                if constexpr (Use_CVT_SWZL_LD) {
                    // SHUFFLING_GAIT causes bit3<->bit6 headdim swap in K's SMEM load.
                    // MMA acc[0..3] (first half after retile_S) belongs to one 48x8 tile,
                    // acc[4..7] (second half) belongs to the partner warp's 48x8 tile.
                    // Use two thread_slices to write each half to its correct destination.
                    // Partner warp: XOR with 4 (warp 0<->4, 1<->5, 2<->6, 3<->7)
                    constexpr int partner_warp_mask = kHeadDim == 256 ? 256 : 128;
                    int partner_tid = thread_idx ^ partner_warp_mask;  // same lane, partner warp = warp^4

                    auto r2s_thr_copy_partner = r2s_tiled_copy_dQaccum.get_thread_slice(partner_tid);
                    Tensor tdQgdQaccum_partner = r2s_thr_copy_partner.partition_D(gdQaccum);

                    Tensor tdQgdQaccum_atomic_own = tdQgdQaccum(_, _, m_block);
                    Tensor tdQgdQaccum_atomic_partner = tdQgdQaccum_partner(_, _, m_block);

                    constexpr int kTotal = CUTE_STATIC_V(size(decltype(tdQrdQ_atomic){}));

                    if ((thread_idx & partner_warp_mask) == 0) {
                        // Low N-warps (n<4): acc[0..3] (i%8<4) -> own, acc[4..7] (i%8>=4) -> partner
                        #pragma unroll
                        for (int i = 0; i < kTotal; ++i) {
                            if ((i % 8) < 4) {
                                atomicAdd(&tdQgdQaccum_atomic_own(i), tdQrdQ_atomic(i));
                            } else {
                                atomicAdd(&tdQgdQaccum_atomic_partner(i - 4), tdQrdQ_atomic(i));
                            }
                        }
                    } else {
                        // High N-warps (n>=4): acc[0..3] (i%8<4) -> partner, acc[4..7] (i%8>=4) -> own
                        #pragma unroll
                        for (int i = 0; i < kTotal; ++i) {
                            if ((i % 8) < 4) {
                                atomicAdd(&tdQgdQaccum_atomic_partner(i + 4), tdQrdQ_atomic(i));
                            } else {
                                atomicAdd(&tdQgdQaccum_atomic_own(i), tdQrdQ_atomic(i));
                            }
                        }
                    }
                    // __syncthreads();
                    // if(thread0()){
                    //     // Debug: print gdQaccum values for current m_block
                    //     float* dq_base = reinterpret_cast<float*>(&gdQaccum(0, m_block));
                    //     printf("=== gdQaccum HBM (m_block=%d) shape=(%d,%d) ===\n", m_block, kBlockM, kHeadDim);
                    //     for (int m = 0; m < kBlockM; ++m) {
                    //         printf("row[%d]: ", m);
                    //         for (int h = 0; h < kHeadDim; ++h) {
                    //             printf("%f ", dq_base[m * kHeadDim + h]);
                    //         }
                    //         printf("\n");
                    //     }
                    // }
                } else {
                    Tensor tdQgdQaccum_atomic = tdQgdQaccum(_, _, m_block);
                    #pragma unroll
                    for (int i = 0; i < size(tdQrdQ_atomic); ++i) {
                        atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
                    }
                }

                // __syncthreads();

                // if(thread(DEBUG_TID)){
                //     // Debug: print gdQaccum values for current m_block
                //     float* dq_base = reinterpret_cast<float*>(&gdQaccum(0, m_block));
                //     printf("=== gdQaccum HBM (m_block=%d) shape=(%d,%d) ===\n", m_block, kBlockM, kHeadDim);
                //     for (int m = 0; m < kBlockM; ++m) {
                //         printf("row[%d]: ", m);
                //         for (int h = 0; h < kHeadDim; ++h) {
                //             printf("%f ", dq_base[m * kHeadDim + h]);
                //         }
                //         printf("\n");
                //     }
                // }
            };
            // If kStages == 1, we want to do Mma_dK first so we can start loading Q for the next iteration
            if constexpr (kStages > 1) { do_mma_dQ(load_dO_next); }
            /** dK = dS^T * Q */
            Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(thr_mma_dKV, sQt(_, _, _0{}));
            Tensor tdKsQ_cur = tdKsQt(_, _, _, kStages > 1 ? smem_pipe_read : 0);
            if constexpr (Mma_dKV_is_RS) {
#ifdef USE_PPU
                Tensor tdKrdS_acc = [&]() -> auto {
                    if constexpr (ArchTag::kMinComputeCapability >= 89) {
                        return make_tensor_like<Element>(tdPrdP);
                    } else {
                        return flash::convert_acc<Element>(tdPrdP);
                    }
                }();
                Tensor tdKrdS = [&]() -> auto {
                    if constexpr (ArchTag::kMinComputeCapability >= 89) {
                        return make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
                    } else {
                        return make_tensor(tdKrdS_acc.data(), make_layout(get<0>(tdPrdO.layout()), get<1>(tdPrdP.layout()), get<2>(tdPrdP.layout())));
                    }
                }();
#else
                Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
#endif
                flash::gemm_rs_sm80(tdKrdK, tdKrdS, tdKrQ, tdKsQ_cur, tiled_mma_dKV, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
            } else {
                Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(thr_mma_dKV, sdSt);
                flash::gemm_sm80<false /*A_in_regs*/, false /*B_in_regs*/, /*SwapAB=*/dKV_swapAB>(
                    tdKrdK, tdKrdS, tdKrQ, tdKsdSt, tdKsQ_cur,
                    tiled_mma_dKV, smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt, cute::conditional_return<(kStages > 1)>(nullptr, load_dO_next));
            }
            if constexpr (kStages == 1) {
                __syncthreads();
                do_mma_dQ(load_Q_next);
            }
            // if (cute::thread0()) { print_tensor(tdKrdK); }

            smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
            smem_pipe_read_do = smem_pipe_read_do < kStages_dO - 1 ? smem_pipe_read_do + 1 : 0;
            smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
            smem_pipe_write_do = smem_pipe_write_do < kStages_dO - 1 ? smem_pipe_write_do + 1 : 0;

        };

        // We have separate iterations with causal masking. Not necessary for hdim 128 but for hdim 64
        // this helps quite a bit to not have to do causal masking for most of the iterations.
        if constexpr ((Is_causal || Is_local) && SeparateMaskingIterations) {
            auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block); };
            int const m_block_masking_max = ((n_block + 1) * kBlockN - 1 + seqlen_q - seqlen_k - params.window_size_right) / kBlockM + 1;
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < std::min(m_block_max, m_block_masking_max); ++m_block) {
                bwd_step(m_block, mask_fn);
            }
        }

        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const m_block_max_before_local_mask = !Is_local || !SeparateMaskingIterations
            ? m_block_max
            : std::min(m_block_max, (n_block * kBlockN + seqlen_q - seqlen_k + params.window_size_left) / kBlockM);

        auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply<true /*Seqlenk_mask*/, Is_causal && !SeparateMaskingIterations, Is_local && !SeparateMaskingIterations>(tSrS, m_block, n_block); };
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max_before_local_mask; ++m_block) {
            bwd_step(m_block, mask_fn);
        }

        if constexpr (Is_local && SeparateMaskingIterations) {
            auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply<true /*Seqlenk_mask*/, false /*Causal_mask*/, Is_local>(tSrS, m_block, n_block); };
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < m_block_max; ++m_block) {
                bwd_step(m_block, mask_fn);
            }
        }

        // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(tdVrdV); }
        #pragma unroll
        for (int i = 0; i < size(tdKrdK); ++i) { tdKrdK(i) *= params.softmax_scale; }

        return true;
    }

};

} // namespace flash
