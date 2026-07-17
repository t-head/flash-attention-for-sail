// Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
__global__ void
__attribute__((hggc_warpsync_del_nse))
flash_fwd_kernel<
    Flash_fwd_kernel_traits<
        128, 128, 64, 4,
        false, false,
        cutlass::bfloat16_t
    >,
    /* Is_dropout     */ false,
    /* Is_causal      */ false,
    /* Is_local       */ false,
    /* Has_alibi      */ false,
    /* Is_even_MN     */ false,
    /* Is_even_K      */ true,
    /* Is_softcap     */ false,
    /* Return_softmax */ false
>(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)
{
#if defined(ARCH_SUPPORTS_FLASH)
    static_assert(!(false && false));
    // if (cute::thread0()) {
    //     printf(">>> specialized flash_fwd_kernel<128,128,64,bf16,false...> is running!\n");
    // }
    FLASH_NAMESPACE::compute_attn<
        Flash_fwd_kernel_traits<
            128, 128, 64, 4,
            false, false,
            cutlass::bfloat16_t
        >,
        false, false, false, false,
        false, true, false, false
    >(params);
#else
    FLASH_UNSUPPORTED_ARCH
#endif
}

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 128, false>(Flash_fwd_params &params, hggcStream_t stream) {
    run_mha_fwd_hdim128<cutlass::bfloat16_t, false>(params, stream);
}

} // namespace FLASH_NAMESPACE