/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdio>
#include <cstdlib>
#include <tuple>

#ifdef __HGGCCC__

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    hggcError_t status_ = call;                                                \
    if (status_ != hggcSuccess) {                                              \
      fprintf(stderr, "HGGC error (%s:%d): %s\n", __FILE__, __LINE__,          \
              hggcGetErrorString(status_));                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#else

#include "hggc_runtime.h"

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    hggcError_t status_ = call;                                                \
    if (status_ != hggcSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,          \
              hggcGetErrorString(status_));                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#endif

inline int get_current_device() {
    int device;
#ifdef __HGGCCC__
    CHECK_CUDA(hggcGetDevice(&device));
#else
    CHECK_CUDA(hggcGetDevice(&device));
#endif
    return device;
}

inline std::tuple<int, int> get_compute_capability(int device) {
    int capability_major, capability_minor;
#ifdef __HGGCCC__
    CHECK_CUDA(hggcDeviceGetAttribute(&capability_major, hggcDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(hggcDeviceGetAttribute(&capability_minor, hggcDevAttrComputeCapabilityMinor, device));
#else
    CHECK_CUDA(hggcDeviceGetAttribute(&capability_major, hggcDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(hggcDeviceGetAttribute(&capability_minor, hggcDevAttrComputeCapabilityMinor, device));
#endif
    return {capability_major, capability_minor};
}

inline int get_num_sm(int device) {
    int multiprocessor_count;
#ifdef __HGGCCC__
    CHECK_CUDA(hggcDeviceGetAttribute(&multiprocessor_count, hggcDevAttrMultiProcessorCount, device));
#else
    CHECK_CUDA(hggcDeviceGetAttribute(&multiprocessor_count, hggcDevAttrMultiProcessorCount, device));
#endif
    return multiprocessor_count;
}
