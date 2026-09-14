// QSA maps one query token per block and packs heads before SplitKV.
#pragma once
#include "../tile_scheduler.hpp"

namespace flash {
template<bool Varlen, bool Split>
class QsaSingleTileScheduler : public SingleTileScheduler<Varlen, Split> {
    using Base = SingleTileScheduler<Varlen, Split>;

public:
    using Base::Base;
    using typename Base::Params;
    using typename Base::WorkTileInfo;

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(!Split || !Varlen || args.num_splits < (1 << 16)); // We use the top 16 bits to store num_splits
        return {args.num_blocks, args.num_head, args.num_batch, !Split ? 1 : args.num_splits,
                args.qhead_per_khead, args.seqlen,
                cutlass::FastDivmod(args.num_head),
                !Varlen ? nullptr : args.cu_seqlens, !Varlen ? nullptr : args.seqused,
                args.num_splits_dynamic_ptr};
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        WorkTileInfo work_info {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), 0};
        if constexpr (Split) {
            int const packed = work_info.bidh;
            work_info.split_idx = params.nsplits_divmod.divmod(work_info.bidh, packed);
        }
        bool is_valid_tile = true;
        if constexpr (Varlen) {
            int seqlen = params.seqused
#ifdef USE_PPU
                ? __ld_smem(&params.seqused[work_info.bidb])
                : (params.cu_seqlens ? __ld_smem(&params.cu_seqlens[work_info.bidb + 1]) - __ld_smem(&params.cu_seqlens[work_info.bidb]) : params.seqlen);
#else
                ? params.seqused[work_info.bidb]
                : (params.cu_seqlens ? params.cu_seqlens[work_info.bidb + 1] - params.cu_seqlens[work_info.bidb] : params.seqlen);
#endif
            // QSA schedules one query token per block; tile rows are GQA heads.
            is_valid_tile = work_info.block_idx < seqlen;
        }
        if constexpr (Varlen && Split) {
#ifdef USE_PPU
            int num_splits_dynamic = params.num_splits_dynamic_ptr ? __ld_smem(&params.num_splits_dynamic_ptr[work_info.bidb]) : params.num_splits;
#else
            int num_splits_dynamic = params.num_splits_dynamic_ptr ? params.num_splits_dynamic_ptr[work_info.bidb] : params.num_splits;
#endif
            is_valid_tile &= work_info.split_idx < num_splits_dynamic;
            // Use the top 16 bits to store num_splits
            work_info.split_idx |= (num_splits_dynamic << 16);
        }
        work_info.bidb = is_valid_tile ? work_info.bidb : -1;
        return work_info;
    }

};
}  // namespace flash
