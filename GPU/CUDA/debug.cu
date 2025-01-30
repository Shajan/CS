#include <cuda_runtime.h>

int main() {
    int deviceId;
    cudaDeviceProp prop;
    cudaError_t err;

    err = cudaGetDevice(&deviceId);
    if (err != cudaSuccess) {
      printf("Failed to get CUDA device: %s\n", cudaGetErrorString(err));
      return -1;
    }

    err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess) {
      printf("Failed to get device properties: %s\n", cudaGetErrorString(err));
      return -1;
    }

    printf("Device Name: %s\n", prop.name);
    printf("CUDA Capability: %d.%d\n", prop.major, prop.minor);

    return 0;
}

