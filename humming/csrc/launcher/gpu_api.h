#pragma once
// GPU driver API used by the launcher.
//
// CUDA: exactly <cuda.h>.
// ROCm (HUMMING_HIP): the subset of the CUDA driver API the launcher uses, implemented on
// the HIP runtime. Contexts are per-device (HIP has no user-visible contexts), PDL launch
// attributes are ignored, and CTA clusters and TMA descriptors are rejected.

#ifndef HUMMING_HIP

#include <cuda.h>

#else  // HUMMING_HIP

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <vector>

// Skips the "headers older than CUDA 12.8" fallbacks in tma.h.
#define CUDA_VERSION 12080

using CUresult = hipError_t;
using CUdevice = hipDevice_t;
using CUmodule = hipModule_t;
using CUfunction = hipFunction_t;
using CUstream = hipStream_t;

constexpr CUresult CUDA_SUCCESS = hipSuccess;
constexpr CUresult CUDA_ERROR_NOT_SUPPORTED = hipErrorNotSupported;

inline CUresult cuGetErrorName(CUresult result, const char **name) {
  *name = hipGetErrorName(result);
  return hipSuccess;
}

inline CUresult cuGetErrorString(CUresult result, const char **description) {
  *description = hipGetErrorString(result);
  return hipSuccess;
}

// ---------------------------------------------------------------------------------------
// Devices and contexts. A context is identified by its device: (device ordinal + 1).
// ---------------------------------------------------------------------------------------
struct HummingHipContext;
using CUcontext = HummingHipContext *;

inline CUcontext humming_hip_context_of_device(int device) {
  return reinterpret_cast<CUcontext>(static_cast<intptr_t>(device) + 1);
}

inline int humming_hip_device_of_context(CUcontext context) {
  return static_cast<int>(reinterpret_cast<intptr_t>(context) - 1);
}

inline CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  return hipDeviceGet(device, ordinal);
}

inline CUresult cuDeviceGetName(char *name, int length, CUdevice device) {
  return hipDeviceGetName(name, length, device);
}

enum CUdevice_attribute_humming {
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = hipDeviceAttributeMultiprocessorCount,
};

inline CUresult cuDeviceGetAttribute(int *value, CUdevice_attribute_humming attribute, CUdevice device) {
  return hipDeviceGetAttribute(value, static_cast<hipDeviceAttribute_t>(attribute), device);
}

inline CUresult cuCtxGetCurrent(CUcontext *context) {
  int device = 0;
  CUresult result = hipGetDevice(&device);
  *context = humming_hip_context_of_device(device);
  return result;
}

inline CUresult cuCtxGetDevice(CUdevice *device) {
  return hipGetDevice(device);
}

inline CUresult cuDevicePrimaryCtxRetain(CUcontext *context, CUdevice device) {
  *context = humming_hip_context_of_device(device);
  return hipSuccess;
}

inline CUresult cuDevicePrimaryCtxRelease(CUdevice) {
  return hipSuccess;
}

inline std::vector<int> &humming_hip_device_stack() {
  thread_local std::vector<int> previous_devices;
  return previous_devices;
}

inline CUresult cuCtxPushCurrent(CUcontext context) {
  int previous_device = 0;
  CUresult result = hipGetDevice(&previous_device);
  if (result != hipSuccess) return result;
  humming_hip_device_stack().push_back(previous_device);
  return hipSetDevice(humming_hip_device_of_context(context));
}

inline CUresult cuCtxPopCurrent(CUcontext *context) {
  cuCtxGetCurrent(context);
  auto &previous_devices = humming_hip_device_stack();
  if (previous_devices.empty()) return hipErrorInvalidContext;
  int previous_device = previous_devices.back();
  previous_devices.pop_back();
  return hipSetDevice(previous_device);
}

// ---------------------------------------------------------------------------------------
// Modules and functions
// ---------------------------------------------------------------------------------------
inline CUresult cuModuleLoadData(CUmodule *module, const void *image) {
  return hipModuleLoadData(module, image);
}

inline CUresult cuModuleGetFunction(CUfunction *function, CUmodule module, const char *name) {
  return hipModuleGetFunction(function, module, name);
}

inline CUresult cuModuleUnload(CUmodule module) {
  return hipModuleUnload(module);
}

enum CUfunction_attribute_humming {
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
};

// CDNA kernels may use the whole LDS for dynamic shared memory without opting in, so this
// only validates the request against the device limit.
inline CUresult cuFuncSetAttribute(CUfunction, CUfunction_attribute_humming, int value) {
  int device = 0;
  int max_shared_memory = 0;
  CUresult result = hipGetDevice(&device);
  if (result != hipSuccess) return result;
  result = hipDeviceGetAttribute(&max_shared_memory, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
  if (result != hipSuccess) return result;
  return value <= max_shared_memory ? hipSuccess : hipErrorInvalidValue;
}

// ---------------------------------------------------------------------------------------
// cuLaunchKernelEx
// ---------------------------------------------------------------------------------------
enum CUlaunchAttributeID {
  CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
  CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
};

struct CUlaunchAttribute {
  CUlaunchAttributeID id;
  union {
    struct {
      unsigned int x;
      unsigned int y;
      unsigned int z;
    } clusterDim;
    int programmaticStreamSerializationAllowed;
  } value;
};

struct CUlaunchConfig {
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  CUlaunchAttribute *attrs;
  unsigned int numAttrs;
};

inline CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction function, void **kernel_params, void **extra) {
  for (unsigned int attribute_index = 0; attribute_index < config->numAttrs; attribute_index++) {
    const CUlaunchAttribute &attribute = config->attrs[attribute_index];
    if (attribute.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
      const auto &cluster = attribute.value.clusterDim;
      if (cluster.x * cluster.y * cluster.z > 1) return hipErrorNotSupported;
    }
    // CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: no PDL on ROCm; the kernels
    // compile griddepcontrol to no-ops, so a plain stream-ordered launch is equivalent.
  }
  return hipModuleLaunchKernel(
      function,
      config->gridDimX,
      config->gridDimY,
      config->gridDimZ,
      config->blockDimX,
      config->blockDimY,
      config->blockDimZ,
      config->sharedMemBytes,
      config->hStream,
      kernel_params,
      extra);
}

// ---------------------------------------------------------------------------------------
// TMA: types only. No AMD equivalent; encoding always fails with "not supported".
// ---------------------------------------------------------------------------------------
#define CU_TENSOR_MAP_NUM_QWORDS 16
typedef uint64_t cuuint64_t;
typedef uint32_t cuuint32_t;
typedef struct CUtensorMap_st {
  alignas(64) cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
} CUtensorMap;

enum CUtensorMapDataType {
  CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
  CU_TENSOR_MAP_DATA_TYPE_UINT16,
  CU_TENSOR_MAP_DATA_TYPE_UINT32,
  CU_TENSOR_MAP_DATA_TYPE_INT32,
  CU_TENSOR_MAP_DATA_TYPE_UINT64,
  CU_TENSOR_MAP_DATA_TYPE_INT64,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ,
  CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
  CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
  CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B,
};

enum CUtensorMapInterleave { CU_TENSOR_MAP_INTERLEAVE_NONE = 0 };

enum CUtensorMapSwizzle {
  CU_TENSOR_MAP_SWIZZLE_NONE = 0,
  CU_TENSOR_MAP_SWIZZLE_32B,
  CU_TENSOR_MAP_SWIZZLE_64B,
  CU_TENSOR_MAP_SWIZZLE_128B,
};

enum CUtensorMapL2promotion { CU_TENSOR_MAP_L2_PROMOTION_NONE = 0 };

enum CUtensorMapFloatOOBfill { CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0 };

inline CUresult cuTensorMapEncodeTiled(
    CUtensorMap *,
    CUtensorMapDataType,
    cuuint32_t,
    void *,
    const cuuint64_t *,
    const cuuint64_t *,
    const cuuint32_t *,
    const cuuint32_t *,
    CUtensorMapInterleave,
    CUtensorMapSwizzle,
    CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

#endif  // HUMMING_HIP
