#include <cuda.h>

#include <cstdio>

struct MockDevice {
  int sm_version;
  int memory_bus_width;
  int sm_count;
};

static constexpr MockDevice devices[] = {
    {121, 0, 48},
    {121, 128, 40},
    {120, 0, 32},
    {89, 0, 24},
    {121, 0, 16},
    {121, -1, 24},
};

static CUresult mock_init(unsigned int) { return CUDA_SUCCESS; }

static CUresult mock_device_count(int *count) {
  *count = sizeof(devices) / sizeof(devices[0]);
  return CUDA_SUCCESS;
}

static CUresult mock_device_get(CUdevice *device, int index) {
  *device = index;
  return CUDA_SUCCESS;
}

static CUresult mock_device_name(char *name, int length, CUdevice) {
  std::snprintf(name, length, "Mock CUDA device");
  return CUDA_SUCCESS;
}

static CUresult mock_device_attribute(int *value, CUdevice_attribute attribute, CUdevice device) {
  const auto &info = devices[device];
  switch (attribute) {
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *value = info.sm_version / 10; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *value = info.sm_version % 10; break;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: *value = info.memory_bus_width; break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: *value = info.sm_count; break;
    case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: *value = 1048576; break;
    default: *value = 1;
  }
  return CUDA_SUCCESS;
}

static CUresult mock_current_context(CUcontext *context) {
  *context = nullptr;
  return CUDA_SUCCESS;
}

static CUresult mock_context_device(CUdevice *device) {
  *device = 0;
  return CUDA_SUCCESS;
}

static CUresult mock_error_text(CUresult, const char **text) {
  *text = "mock CUDA error";
  return CUDA_SUCCESS;
}

#define cuInit mock_init
#define cuDeviceGetCount mock_device_count
#define cuDeviceGet mock_device_get
#define cuDeviceGetName mock_device_name
#define cuDeviceGetAttribute mock_device_attribute
#define cuCtxGetCurrent mock_current_context
#define cuCtxGetDevice mock_context_device
#define cuGetErrorName mock_error_text
#define cuGetErrorString mock_error_text

#include "device_info.cpp"
