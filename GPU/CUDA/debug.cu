#include <cuda_runtime.h>

int main() {
    // Check device
    int deviceId;
    cudaDeviceProp prop;
    cudaError_t err;

    err = cudaGetDevice(&deviceId);
    if (err != cudaSuccess) {
      std::cerr << "Failed to get CUDA device: " << cudaGetErrorString(err) << std::endl;
      return -1;
    }

    err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess) {
      std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
      return -1;
    }

    printf("Device Name: %s\n", prop.name);
    printf("CUDA Capability: %d.%d\n", prop.major, prop.minor);

    return 0;
}

