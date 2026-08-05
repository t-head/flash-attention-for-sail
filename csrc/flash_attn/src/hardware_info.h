/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdio>
#include <cstdlib>
#include <tuple>

#if !defined(__HGGCCC_RTC__)
#include "hggc_runtime.h"
#endif

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    hggcError_t status_ = call;                                                \
    if (status_ != hggcSuccess) {                                              \
      fprintf(stderr, "HGGC error (%s:%d): %s\n", __FILE__, __LINE__,          \
              hggcGetErrorString(status_));                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


inline int get_current_device() {
    int device;
    CHECK_CUDA(hggcGetDevice(&device));
    return device;
}

inline std::tuple<int, int> get_compute_capability(int device) {
    int capability_major, capability_minor;
    CHECK_CUDA(hggcDeviceGetAttribute(&capability_major, hggcDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(hggcDeviceGetAttribute(&capability_minor, hggcDevAttrComputeCapabilityMinor, device));
    return {capability_major, capability_minor};
}

inline int get_num_sm(int device) {
    int multiprocessor_count;
    CHECK_CUDA(hggcDeviceGetAttribute(&multiprocessor_count, hggcDevAttrMultiProcessorCount, device));
    return multiprocessor_count;
}
