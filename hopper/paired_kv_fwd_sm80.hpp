#pragma once

#include "cute/tensor.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/arch/copy_ppu0015_aiu.hpp"
#include "cute/atom/copy_traits_ppu0015_aiu.hpp"

namespace cute {

// Divide a 64x256 KV stage into 16x64 AIU tiles. Interleave adjacent
// head-dimension bands within each 16-row stripe.
template <typename Element, bool Transpose>
struct PPU0015_D256_PairedKVLoad {
    static_assert(sizeof(Element) == 2);
    static constexpr int Rows = 64;
    static constexpr int BandWidth = 64;
    static constexpr int Bands = 4;
    static constexpr int StripeRows = 16;
    static constexpr int StageElements = Rows * BandWidth * Bands;
    static constexpr int StripeElements = StripeRows * BandWidth;

    CUTE_HOST_DEVICE static constexpr int band_pair_offset(int band) {
        return (band & ~1) * Rows * BandWidth;
    }

    CUTE_HOST_DEVICE static constexpr int stripe_offset(int band, int stripe) {
        return band_pair_offset(band) + (band & 1) * StripeElements
            + 2 * stripe * StripeElements;
    }

    // Dense and paged producers share the same paired destination layout.
    CUTLASS_DEVICE static void copy_stripe(
        Element* stage, Element const* source, AiuDesc const& descriptor,
        int stripe, int band, int source_row = 0
    ) {
        using Aiu = PPU0015_AIU_LOAD<Int<StripeElements * 16>, Element, false, StripeRows, BandWidth>;
        Aiu::copy(stage + stripe_offset(band, stripe), source, descriptor,
                  band * BandWidth, source_row);
    }

    CUTLASS_DEVICE static void copy_stage(
        Element* smem, Element const* gmem, AiuDesc const& descriptor,
        int block, int slot, int warp
    ) {
        if (warp < Bands) {
            auto* stage = smem + slot * StageElements;
            #pragma unroll
            for (int stripe = 0; stripe < Rows / StripeRows; ++stripe) {
                copy_stripe(stage, gmem, descriptor, stripe, warp,
                            block * Rows + stripe * StripeRows);
            }
        }
    }

    CUTE_HOST_DEVICE static void copy(
        void* fragment, void* smem, int column, int row, int band, int slot
    ) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
        // Keep per-fragment offsets in the 16-byte units used by SWZL.
        // The shared stage base is aligned, so its conversion is common to all fragments.
        constexpr int ElementsPerUnit = 16 / sizeof(Element);
        uint32_t const base_units = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem) >> 4);
        int const offset_units = slot * (StageElements / ElementsPerUnit)
            + band_pair_offset(band) / ElementsPerUnit + (band & 1)
            + row * (2 * BandWidth / ElementsPerUnit) + column / ElementsPerUnit;
        int const address = static_cast<int>(base_units + offset_units);
        auto* destination = static_cast<int*>(fragment);
        if constexpr (Transpose) {
            // V traverses the paired TSM groups in order 0,1,2,3.
            PPU0015_TSM_LD_SWZL_IMPL<Element, true, false>()(destination, address, 128, 64, 0);
        } else {
            // K traverses 0,2,1,3. The sub-block adjustment follows band parity.
            int const sbo = 128 + ((band & 1) ? -1 : 1);
            PPU0015_TSM_LD_SWZL_IMPL<Element, false, false>()(destination, address, 64, sbo, 0);
        }
#else
        CUTE_INVALID_CONTROL_PATH("Paired KV SWZL requires PPU1.5");
#endif
    }
};

template <typename Element, bool Transpose>
struct Copy_Traits<PPU0015_D256_PairedKVLoad<Element, Transpose>> {
    using Base = Copy_Traits<PPU0015_TSM_LD_SWZL<Element, 64, 64, true, Transpose, 4>>;
    using ThrID = typename Base::ThrID;
    using SrcLayout = typename Base::SrcLayout;
    using DstLayout = typename Base::DstLayout;
    using RefLayout = typename Base::RefLayout;

    template <class TS, class SLayout, class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr void copy_unpack(
        Copy_Traits const&, Tensor<TS, SLayout> const& src, Tensor<TD, DLayout>& dst
    ) {
        static_assert(is_mix_iterator<typename TS::iterator>::value);
        auto const source = src.data();
        auto const coord = source.coord_;
        constexpr int Column = Transpose ? 1 : 0;
        constexpr int Row = Transpose ? 0 : 1;
        PPU0015_D256_PairedKVLoad<Element, Transpose>::copy(
            raw_pointer_cast(dst.data()), source.ptr_.get(),
            get<Column>(coord), get<Row>(coord), get<2>(coord), get<3>(coord));
    }
};

} // namespace cute

namespace flash {

// Match the paired K reduction order by interleaving D64 cubes every 8 Q rows.
struct D256PairedQLayout {
    static constexpr int Rows = 256;
    static constexpr int CubeRows = 8;
    static constexpr int CubeWidth = 64;
    static constexpr int ElementsPerPair = Rows * 2 * CubeWidth;
    using Layout = decltype(cute::composition(
        cute::Swizzle<3, 3, 3>{},
        cute::Layout<cute::Shape<cute::Shape<cute::_8, cute::_32>,
                                 cute::Shape<cute::_64, cute::_2, cute::_2>>,
                     cute::Stride<cute::Stride<cute::_64, cute::Int<1024>>,
                                  cute::Stride<cute::_1, cute::_512, cute::Int<ElementsPerPair>>>>{}));

    // The AIU applies the same row swizzle that the packed Q tensor encodes.
    template <typename Element>
    CUTLASS_DEVICE static void copy_row_group(
        Element* smem, Element const* gmem, cute::AiuDesc const& desc, int row_group, int src_row
    ) {
        using Aiu = cute::PPU0015_AIU_LOAD<cute::Int<CubeRows * CubeWidth * 16>,
                                          Element, false, CubeRows, CubeWidth>;
        static_assert(sizeof(Element) == 2);
        cute::for_each(cute::make_int_sequence<4>{}, [&](auto d64) {
            constexpr int band = decltype(d64)::value;
            constexpr int band_offset = (band / 2) * ElementsPerPair + (band % 2) * CubeRows * CubeWidth;
            auto* dst = smem + row_group * (2 * CubeRows * CubeWidth) + band_offset;
            Aiu::copy(dst, gmem, desc, band * CubeWidth, src_row);
        });
    }

    // Read a 16x16 MMA fragment in the same reduction order as the paired K reader.
    template <int Atom, typename Element, typename Fragment>
    CUTLASS_DEVICE static void load_fragment(Element const* smem, int warp, Fragment& fragment) {
        CUTE_STATIC_ASSERT_V(size(fragment) == cute::_8{});
        CUTE_STATIC_ASSERT_V(cosize(fragment.layout()) == cute::_8{});
        static_assert(sizeof(Element) == 2 && Atom >= 0 && Atom < 16);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
        constexpr int ElementsPerUnit = 16 / sizeof(Element);
        constexpr int Column = (Atom / 8) * ElementsPerPair + ((Atom / 4) % 2) * ElementsPerUnit
            + (Atom % 4) * 16;
        constexpr int Lbo = 64 + (Atom / 4 % 2 == 0 ? 1 : -1);
        uint32_t const base = cute::cast_smem_ptr_to_uint(smem) >> 4;
        int const address = base + warp * (4 * CubeRows * CubeWidth / ElementsPerUnit) + Column / ElementsPerUnit;
        auto* destination = reinterpret_cast<int*>(cute::raw_pointer_cast(fragment.data()));
        cute::PPU0015_TSM_LD_SWZL_IMPL<Element, false, false>()(destination, address, Lbo, 128, 0);
#else
        CUTE_INVALID_CONTROL_PATH("Paired Q SWZL requires PPU1.5");
#endif
    }
};

} // namespace flash
