/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

#if defined(USE_PPU) && USE_AIU
template <int kBlockN, int kHeadDim, int NumThreads, typename Element, bool KV_Same_Iter=false, int LoadsPerRow_LB=1, bool PagedAiuKV=false, int kBlockNPagedPerAiuLoad=16>
#else
template <int kBlockN, int kHeadDim, int NumThreads, typename Element, bool KV_Same_Iter=false, int LoadsPerRow_LB=1>
#endif
struct PagedKVManager {
    // If KV_Same_Iter=false, then we do load_page_table(0), load_K(0), load_page_table(1), load_K(1), load_V(0),
    // load_page_table(2), load_K(2), load_V(1), etc.
    // So we need to compute the V pointers for the previous iteration.

    // LoadsPerRow_LB is the lower bound on number of loads per row in the K direction. This is useful for
    // rotary where we want each thread to have at least 2 loads per row.

    // We use CpAsync for K and V if PagedKV, since TMA doesn't work there
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g. if hdim=128, we want each
    // thread to have 4 loads in the M direction and 2 vectorized load in the K direction.
    // In the case of PackGQA, this reduces the number of times we need to call divmod.
    static_assert(kHeadDim % LoadsPerRow_LB == 0, "Headdim must be a multiple of LoadsPerRow_LB");
    static constexpr int kBytePerRow = kHeadDim / LoadsPerRow_LB * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
    // We assume threads loading the same row are in the same warp. This is for an optimization in PagedKV where
    // these threads share the same page table entry and share the work of computing pointers to paged K and paged V.
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using GmemCopyAtomCpAsync = cute::Copy_Atom<PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
    using GmemLayoutAtomKVCpAsync = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                           Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyKVCpAsync = decltype(
        make_tiled_copy(GmemCopyAtomCpAsync{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load
    using GmemTiledCopyKVStore = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    using ShapeKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapePageTable = cute::Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)
    using StridePageTable = cute::Stride<int64_t, _1>;

    using TensorPageTable = decltype(make_tensor(make_gmem_ptr(static_cast<int const*>(nullptr)), ShapePageTable{}, StridePageTable{})(int(0), _));
    using TensorKV = decltype(make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeKV{}, StrideKV{})(_, _, int(0), _));
    using GmemThrCopyKVCpAsync = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)));
    using TensortKcK = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{})));
    using TensortKpK = decltype(make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{}));

    // For PagedKV, it's expensive the calculate the pointers to K and V for each page table entry,
    // since those require int64_t arithmetic. We optimize by having threads split this work.
    // Typically there are 8 threads loading per row (e.g. hdim 64 and 128), and there are 11 rows
    // that each thread needs to load for the case of hdim 128 and kBlockN = 176.
    // So each of those 8 threads will calculate the K_ptr and V_ptr for 11 / 8 = 2 rows.
    // We then use __shfl_sync to broadcast the pointers to the other threads in the warp.
    static constexpr int kPageEntryPerThread = cute::ceil_div(size<1>(TensortKcK{}), kGmemThreadsPerRow);
    using TensorPageOffset = decltype(make_tensor<cute::tuple<int, int>>(Shape<Int<kPageEntryPerThread>>{}));
    using TensorKVPtr = decltype(make_tensor<Element*>(Shape<Int<kPageEntryPerThread>>{}));

    GmemTiledCopyKVCpAsync gmem_tiled_copy_kv;
    cutlass::FastDivmod const &page_size_divmod;
    int const thread_idx;
    int const seqlen_k;
    int const leftpad_k;
    GmemThrCopyKVCpAsync const gmem_thr_copy_kv;
    TensorPageTable mPageTable;
    TensorKV mK_paged, mV_paged;
    TensortKpK tKpK;
    TensorPageOffset tPrPageOffset;
    TensorKVPtr tPrVPtr;
#if defined(__HGGC_ARCH__) && defined(USE_PPU) && USE_AIU
    static constexpr int bits_per_aiu_KV = kBlockNPagedPerAiuLoad * kBlockKGmem * sizeof(Element) * 8;
#if __HGGC_ARCH__ == 100
    using Gmem_copy_struct_KV = PPU0010_AIU_LOAD<cute::C<bits_per_aiu_KV>, Element, false>;
#elif __HGGC_ARCH__ == 150
    using Gmem_copy_struct_KV = PPU0015_AIU_LOAD<cute::C<bits_per_aiu_KV>, Element, false, kBlockNPagedPerAiuLoad, kBlockKGmem>;
#endif
    using GmemTiledCopyKVAiu = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct_KV, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Int<kBlockNPagedPerAiuLoad>, Int<kBlockKGmem>>>{}));
    using GmemThrCopyKVAiu = decltype(GmemTiledCopyKVAiu{}.get_thread_slice(int(0)));
    static constexpr int kNWarps = NumThreads / 32;
    int const warp_idx;
    static constexpr int kNAiuLoads = cute::ceil_div(kBlockN, kBlockNPagedPerAiuLoad);
    static constexpr int kPageEntryPerWarp = cute::ceil_div(kNAiuLoads, kNWarps);
    using TensorPageOffsetAiu = cute::tuple<int, int>;
    using TensorKVPtrAiu = Element*;
    GmemTiledCopyKVAiu gmem_tiled_copy_k_aiu;
    GmemTiledCopyKVAiu gmem_tiled_copy_v_aiu;
    GmemThrCopyKVAiu const gmem_thr_copy_k_aiu;
    GmemThrCopyKVAiu const gmem_thr_copy_v_aiu;
    TensorPageOffsetAiu tPrPageOffsetAiu;
    TensorKVPtrAiu tPrVPtrAiu;
#endif


    CUTLASS_DEVICE
    PagedKVManager(int const* const ptr_page_table,
                   ShapePageTable const &shape_pagetable, StridePageTable const &stride_pagetable,
                   Element* const ptr_K, ShapeKV const &shape_K, StrideKV const &stride_K,
                   Element* const ptr_V, StrideKV const &stride_V,
                   cutlass::FastDivmod const &page_size_divmod,
                   int const bidb, int const bidh, int const thread_idx, int const seqlen_k, int const leftpad_k
                   )
        : page_size_divmod(page_size_divmod)
        , thread_idx(thread_idx)
        , seqlen_k(seqlen_k)
        , leftpad_k(leftpad_k)
#if defined(__HGGC_ARCH__) &&  __HGGC_ARCH__ >= 100 && defined(USE_PPU) && USE_AIU
        , gmem_thr_copy_k_aiu(gmem_tiled_copy_k_aiu.get_thread_slice(thread_idx))
        , gmem_thr_copy_v_aiu(gmem_tiled_copy_v_aiu.get_thread_slice(thread_idx))
        , warp_idx(thread_idx / 32)
#endif
        , gmem_thr_copy_kv(gmem_tiled_copy_kv.get_thread_slice(thread_idx))

    {
        mPageTable = make_tensor(make_gmem_ptr(ptr_page_table), shape_pagetable, stride_pagetable)(bidb, _);
        mK_paged = make_tensor(make_gmem_ptr(ptr_K), shape_K, stride_K)(_, _, bidh, _);
        mV_paged = make_tensor(make_gmem_ptr(ptr_V), shape_K, stride_V)(_, _, bidh, _);
        tKpK = make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{});

        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        #pragma unroll
        for (int k = 0; k < size<1>(tKpK); ++k) { tKpK(_0{}, k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(shape_K); }
#if defined(__HGGC_ARCH__) && defined(USE_PPU) && USE_AIU
#if __HGGC_ARCH__ == 100
        int aiu_offset_k = get<1>(shape_K) == kHeadDim ? 0 : (get<0>(stride_K) - get<1>(shape_K));
        int aiu_offset_v = get<1>(shape_K) == kHeadDim ? 0 : (get<0>(stride_V) - get<1>(shape_K));
        gmem_tiled_copy_k_aiu.desc_ = AiuDesc{nullptr, kBlockNPagedPerAiuLoad, get<0>(stride_K), kBlockNPagedPerAiuLoad, kBlockKGmem, aiu_offset_k};
        gmem_tiled_copy_v_aiu.desc_ = AiuDesc{nullptr, kBlockNPagedPerAiuLoad, get<0>(stride_V), kBlockNPagedPerAiuLoad, kBlockKGmem, aiu_offset_v};
#elif __HGGC_ARCH__ == 150
        gmem_tiled_copy_k_aiu.desc_.init(nullptr, kBlockNPagedPerAiuLoad, get<1>(shape_K), get<0>(stride_K));
        gmem_tiled_copy_v_aiu.desc_.init(nullptr, kBlockNPagedPerAiuLoad, get<1>(shape_K), get<0>(stride_V));
#endif
#endif
    };

#if defined(__HGGC_ARCH__) &&  __HGGC_ARCH__ >= 100 && defined(USE_PPU) && USE_AIU
    CUTLASS_DEVICE
    TensorKVPtrAiu compute_K_ptr_aiu() {
        auto [page, page_offset] = tPrPageOffsetAiu;
        TensorKVPtrAiu tPrKPtrAiu = &mK_paged(page_offset, _0{}, page);
        return tPrKPtrAiu;
    };

    CUTLASS_DEVICE
    void compute_V_ptr_aiu() {
        auto [page, page_offset] = tPrPageOffsetAiu;
        tPrVPtrAiu = &mV_paged(page_offset, _0{}, page);
    };
#endif

    template <bool Seqlenk_mask=false, bool First_iter=false>
    CUTLASS_DEVICE
    void load_page_table(const int n_block) {
#if defined(__HGGC_ARCH__) &&  __HGGC_ARCH__ >= 100 && defined(USE_PPU) && USE_AIU
        if constexpr (PagedAiuKV) {
            int const page_entry_idx = thread_idx % kPageEntryPerWarp;
            int const row = (page_entry_idx * kNWarps + warp_idx) * kBlockNPagedPerAiuLoad;
            int const row_idx = n_block * kBlockN + row;
            int page_idx, page_offset;
            page_idx = page_size_divmod.divmod(page_offset, row_idx + leftpad_k);
            int const page = (row < kBlockN) && (!Seqlenk_mask || row_idx < seqlen_k) ? mPageTable[page_idx] : 0;
            tPrPageOffsetAiu = {page, page_offset};
            if constexpr (First_iter && !KV_Same_Iter) { compute_V_ptr_aiu(); }
            return;
        }
#endif
        // The uncoalesced gmem load is intentional. This is so that each thread only loads the page table entries
        // it needs, and we don't need any sync between warps.
        // Assuming 8 threads per row, and 176 rows, then the rows from 0 to 175 are loaded by
        // threads 0, 8, 16, ..., 120, 1, 9, ..., 121, 2, 10, ..., 122, etc.
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            int const row = i * NumThreads + (thread_idx % kGmemThreadsPerRow) * (NumThreads / kGmemThreadsPerRow) + (thread_idx / kGmemThreadsPerRow);
            int const row_idx = n_block * kBlockN + row;
            int page_idx, page_offset;
            page_idx = page_size_divmod.divmod(page_offset, row_idx + leftpad_k);
            // Add the condition (i + 1) * NumThreads <= kBlockN since that is an upper bound of row
            // and is known at compile time. It avoids branching when e.g., kBlockN = 176 and i = 0.
            int const page = ((i + 1) * NumThreads <= kBlockN || row < kBlockN) && (!Seqlenk_mask || row_idx < seqlen_k) ? mPageTable[page_idx] : 0;
            tPrPageOffset[i] = {page, page_offset};
            // if (cute::thread0()) { printf("row = %d, page_idx = %d, page_offset = %d, page = %d, leftpad_k = %d, seqlen_k = %d\n", row, page_idx, page_offset, page, leftpad_k, seqlen_k); }
        }
        if constexpr (First_iter && !KV_Same_Iter) { compute_V_ptr(); }
    };

    CUTLASS_DEVICE
    TensorKVPtr compute_K_ptr() {
        Tensor tPrKPtr = make_tensor<Element*>(Shape<Int<kPageEntryPerThread>>{});
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            auto [page, page_offset] = tPrPageOffset[i];
            tPrKPtr[i] = &mK_paged(page_offset, _0{}, page);
        }
        return tPrKPtr;
    };

    CUTLASS_DEVICE
    void compute_V_ptr() {
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            auto [page, page_offset] = tPrPageOffset[i];
            tPrVPtr[i] = &mV_paged(page_offset, _0{}, page);
        }
    };

    template <bool Seqlenk_mask=false, typename TensorK>
    CUTLASS_DEVICE
    void load_K(const int n_block, TensorK &&sK) {
#if defined(__HGGC_ARCH__) &&  __HGGC_ARCH__ >= 100 && defined(USE_PPU) && USE_AIU
        if constexpr (PagedAiuKV) {
            TensorKVPtrAiu tPrKPtrAiu = compute_K_ptr_aiu();
            Tensor tKsK = gmem_thr_copy_k_aiu.partition_D(sK);
            #pragma unroll
            for (int m = 0; m < kPageEntryPerWarp; m++) {
                int row_idx = warp_idx + m * kNWarps;
                if (row_idx * kBlockNPagedPerAiuLoad >= kBlockN) { break; }
                Element* k_ptr;
                if constexpr (kPageEntryPerWarp > 1) {
                    k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtrAiu), m, kPageEntryPerWarp));
                } else {
                    k_ptr = tPrKPtrAiu;
                }
                Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), make_shape(kBlockNPagedPerAiuLoad, shape<1>(mK_paged)), select<0, 1>(stride(mK_paged)));
                Tensor mK_paged_cur_copy = cute::tiled_divide(make_mix_tensor_like(mK_paged_cur), Shape<Int<kBlockNPagedPerAiuLoad>, Int<kBlockKGmem>>{})(_, _0{}, _);
                cute::copy(gmem_tiled_copy_k_aiu, mK_paged_cur_copy, tKsK(_, row_idx, _));
            }
            return;
        }
#endif
        // Do we need bound check to make sure the row doesn't go above kBlockN
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;

        Tensor tPrKPtr = compute_K_ptr();

        // Only for index calculation, since all the indices of thread 0 are known at compile time
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

        // We want to use the row indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first row index of this thread (get<0>(tKcK(_0{}, _0{}, _0{})))
        int const seqlenk_row_limit = -int(get<0>(tKcK(_0{}, _0{}, _0{}))) + (EvenN
            ? seqlen_k - n_block * kBlockN
            : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k - n_block * kBlockN, kBlockN)));
        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            bool const should_load = EvenN
                ? (!Seqlenk_mask || get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit)
                : get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            Element const* k_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
            Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
            Tensor mK_paged_cur_copy = cute::tiled_divide(mK_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKsK); ++k) {
                    int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    cute::copy(gmem_tiled_copy_kv.with(tKpK(_0{}, k)), mK_paged_cur_copy(_, ki), tKsK(_, m, k));
                }
            }  // Don't need to clear out the rest of the smem since we'll mask out the scores anyway
        }
    };

    template <bool Seqlenk_mask=false, typename TensorV>
    CUTLASS_DEVICE
    void load_V(const int n_block, TensorV &&sV) {
#if defined(__HGGC_ARCH__) &&  __HGGC_ARCH__ >= 100 && defined(USE_PPU) && USE_AIU
        if constexpr (PagedAiuKV) {
            if constexpr (KV_Same_Iter) { compute_V_ptr_aiu(); }
            Tensor tVsV = gmem_thr_copy_v_aiu.partition_D(sV);
            #pragma unroll
            for (int m = 0; m < kPageEntryPerWarp; m++) {
                int row_idx = warp_idx + m * kNWarps;
                if (row_idx * kBlockNPagedPerAiuLoad >= kBlockN) { break; }
                Element* v_ptr;
                if constexpr (kPageEntryPerWarp > 1) {
                    v_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtrAiu), m, kPageEntryPerWarp));
                } else {
                    v_ptr = tPrVPtrAiu;
                }
                Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), make_shape(kBlockNPagedPerAiuLoad, shape<1>(mV_paged)), select<0, 1>(stride(mV_paged)));
                Tensor mV_paged_cur_copy = cute::tiled_divide(make_mix_tensor_like(mV_paged_cur), Shape<Int<kBlockNPagedPerAiuLoad>, Int<kBlockKGmem>>{})(_, _0{}, _);
                cute::copy(gmem_tiled_copy_v_aiu, mV_paged_cur_copy, tVsV(_, row_idx, _));
            }
            if constexpr (!KV_Same_Iter) { compute_V_ptr_aiu(); }
            return;
        }
#endif
        // Do we need bound check to make sure the row doesn't go above kBlockN
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;

        if constexpr (KV_Same_Iter) { compute_V_ptr(); }
        // Only for index calculation, since all the indices of thread 0 are known at compile time
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

        int const seqlenk_row_limit = seqlen_k - n_block * kBlockN - get<0>(tKcK(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tVsV); ++m) {
            // Faster to rely on the cp.async to clear smem that are out of bound,
            // rather than calling cute::clear directly.
            // We have to be careful not to write to smem past `kBlockN` if !EvenN.
            // If kBlockN doesn't evenly divide the tiled copy, only the last `m` needs to checked
            if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tKcK(_0{}, m, _0{})) < kBlockN) {
                bool const should_load = !Seqlenk_mask || get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
                Element const* v_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
                Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDim>>{});
                Tensor mV_paged_cur_copy = cute::tiled_divide(mV_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tVsV); ++k) {
                    int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    cute::copy(gmem_tiled_copy_kv.with(tKpK(_0{}, k) && should_load), mV_paged_cur_copy(_, ki), tVsV(_, m, k));
                }
            }
        }
        if constexpr (!KV_Same_Iter) { compute_V_ptr(); }
    };

    template <typename TensorK>
    CUTLASS_DEVICE
    void store_K(const int n_block, TensorK &&tKrK) {
        Tensor tPrKPtr = compute_K_ptr();
        // We're using the same partitioning as GmemTiledCopyKVCpAsync (used for loading)
        // Only for index calculation, since all the indices of thread 0 are known at compile time
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

        GmemTiledCopyKVStore gmem_tiled_copy_kv_store;
        // We want to use the row indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first row index of this thread (get<0>(tKcK(_0{}, _0{}, _0{})))
        // int const seqlenk_row_limit = seqlen_k - n_block * kBlockN - get<0>(tKcK(_0{}, _0{}, _0{}));
        int const seqlenk_row_limit = std::min(seqlen_k - n_block * kBlockN, kBlockN) - get<0>(tKcK(_0{}, _0{}, _0{}));
        // if (threadIdx.x == 128) { printf("bidx = %d, bidy = %d, bidz = %d, seqlen_k = %d, seqlenk_row_limit = %d\n", blockIdx.x, blockIdx.y, blockIdx.z, seqlen_k, seqlenk_row_limit); }
        #pragma unroll
        for (int m = 0; m < size<1>(tKrK); ++m) {
            bool const should_load = get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            Element* k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
            Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
            Tensor mK_paged_cur_copy = cute::tiled_divide(mK_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKrK); ++k) {
                    int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    if (tKpK(_0{}, k)) {
                        cute::copy(gmem_tiled_copy_kv_store, tKrK(_, m, k), mK_paged_cur_copy(_, ki));
                    }
                }
            }
        }
    };

    template <typename TensorV>
    CUTLASS_DEVICE
    void store_V(const int n_block, TensorV &&tVrV) {
        if constexpr (KV_Same_Iter) { compute_V_ptr(); }
        // Only for index calculation, since all the indices of thread 0 are known at compile time
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

        GmemTiledCopyKVStore gmem_tiled_copy_kv_store;
        int const seqlenk_row_limit = std::min(seqlen_k - n_block * kBlockN, kBlockN) - get<0>(tKcK(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tVrV); ++m) {
            bool const should_load = get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            Element* v_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDim>>{});
            Tensor mV_paged_cur_copy = cute::tiled_divide(mV_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tVrV); ++k) {
                    int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    if (tKpK(_0{}, k)) {
                        cute::copy(gmem_tiled_copy_kv_store, tVrV(_, m, k), mV_paged_cur_copy(_, ki));
                    }
                }
            }
        }
        if constexpr (!KV_Same_Iter) { compute_V_ptr(); }
    };


};

} // namespace flash
