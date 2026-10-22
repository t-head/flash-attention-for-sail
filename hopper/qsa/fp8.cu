// Copyright (c) 2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
#define FLASHATTENTION_QSA_FP8 1
#include "flash_fwd_launch_template.h"
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150 && !defined(PPU_FP8_CVT_ENABLED)
#error "QSA FP8 requires ACTLIZE hardware FP8 conversion support"
#endif
namespace qsa {
struct Fp8SingleQuery : flash::QsaConfig {
    static constexpr bool TsmQ = true;
    static constexpr int Group = 6, M = 16, N = 32, Warps = 1, Stages = 1;
    static constexpr int RowBytes = 128, KVStride = 256;
    static constexpr bool QRegs = false, TsmKV = true, FixedGroup = false;
    static constexpr bool ShareKV = true;
    static constexpr int VParts = 1;
    static constexpr bool DirectIndex = false, SingleTile = true;
};
struct Fp8SixHead : Fp8SingleQuery {
    static constexpr bool FixedGroup = true;
};
// H24 uses two independent V partitions for both decode and prefill.
struct Fp8TwoVPart : Fp8SingleQuery {
    static constexpr bool ShareKV = false;
    static constexpr bool QRegs = true;
    static constexpr int VParts = 2;
};
// The generic path uses the existing persistent QSA scheduler for arbitrary
// query lengths, KV head counts, page sizes and strided physical caches.
struct Fp8Generic : Fp8SingleQuery {
    static constexpr int KVStride = 0;
    static constexpr bool ShareKV = false, QRegs = true, SingleTile = false;
};
struct Fp8Generic32 : Fp8Generic {
    static constexpr int M = 32, Warps = 2;
};
template <bool Split, class Config = Fp8SingleQuery>
void launch_fp8(Flash_fwd_params &p, hggcStream_t stream) {
    run_flash_fwd<89, 256, 256 / Config::VParts, 1, cutlass::float_e4m3_t, cutlass::bfloat16_t,
                  true, false, false, true, true, false, 1, false, false, true, Split, false, false,
                  true, false, true, Config>(p, stream);
}

}  // namespace qsa

bool run_qsa_fp8(Flash_fwd_params &p, hggcStream_t stream) {
    using namespace qsa;
    if (!qsa_fp8_config_supported(p) || !p.pack_gqa) {
        return false;
    }
    if (qsa_fp8_uses_tuned_layout(p)) {
        if (p.num_splits == 1) {
            if (p.h_k == 2) { launch_fp8<false, Fp8TwoVPart>(p, stream); }
            else { launch_fp8<false, Fp8SixHead>(p, stream); }
        } else {
            if (p.h_k == 2) { launch_fp8<true, Fp8TwoVPart>(p, stream); }
            else { launch_fp8<true, Fp8SixHead>(p, stream); }
        }
    } else if (p.h / p.h_k <= 16) {
        if (p.num_splits == 1) { launch_fp8<false, Fp8Generic>(p, stream); }
        else { launch_fp8<true, Fp8Generic>(p, stream); }
    } else {
        if (p.num_splits == 1) { launch_fp8<false, Fp8Generic32>(p, stream); }
        else { launch_fp8<true, Fp8Generic32>(p, stream); }
    }
    // flash_api.cpp invokes the existing combine dispatch after forward.
    return true;
}
