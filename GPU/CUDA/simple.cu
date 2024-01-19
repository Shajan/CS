#include <iostream>
#include <cuda_runtime.h>
#include <vector>

__global__ void sumKernel(float* input, float* output, int n);

int main() {
    // Check device
    int deviceId;
    cudaDeviceProp prop;
    cudaError_t err;

    // Find max number of threads from the device
    int max_threads_per_block = -1;

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

    // 1024 : T4, V100, A100
    // std::cout << "Maximum number of threads per block: " << prop.maxThreadsPerBlock << std::endl;
    //max_threads_per_block = prop.maxThreadsPerBlock;

    // See GPU cache below, where this constant is used.
    max_threads_per_block = 256;

    const int numElements = 1000000;
    size_t size = numElements * sizeof(float);

    // h_ is for RAM (host memory)
    // d_ is for GPU memory (device memory)

    // Allocate and initialize host memory
    std::vector<float> h_input(numElements, 1.0f); // Example: a vector of 1.0f
    float h_output = 0.0f;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int num_parallel_threads = std::min(64, max_threads_per_block);

    // Make sure all data (numElements) are covered
    int num_blocks = (numElements + num_parallel_threads - 1) / num_parallel_threads;

    // Launch 'num_parallel_threads' threads on GPU
    // Do 'num_blocks' passes
    sumKernel<<<num_blocks, num_parallel_threads>>>(d_input, d_output, numElements);

    // Copy result back to host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sum of 1 million elements: " << h_output << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

// Adds contents of a vector of floats
// This code executes on a GPU
//
// Called from CPU : sumKernel<<<num_blocks, num_parallel_threads>>>(d_input, d_output, numElements);
//
__global__ void sumKernel(float* input, float* output, int n) {
    //
    // Computation is done by 'num_parallel_threads' in parallel
    // There are 'num_blocks' blocks each of size 'num_parallel_threads'
    //
    // threadIdx.x : Thread number (helps find the data offset for this execution)
    // blockIdx.x : Block number (helps find which block each of the thread is at)
    // blockDim.x : Size of a block (number of threads)
    // gridDim.x : Number of blocks in the 'grid'
    //
    // *.x variables are provided in CUDA, in addtion for 2 D, *.y etc is also provided
    //

    // Find the offset of data that needs to be processed by this thread.
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    // Each thread sums a subset of the data
    // NOTE:
    //   Memory locality is better when every thread processes 'n'-th item instead of sequential.
    //   This is because all threads are expect to be moving along at the same speed.
    //   Cache memory locality is achieved across threads!
    for (int i = index; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Write the result to shared memory
    // 256 => Number of threads, one cache entry / thread
    __shared__ float cache[256];
    int cacheIndex = threadIdx.x;
    cache[cacheIndex] = sum;

    // Wait for all 256 threads to finish updating cache
    __syncthreads();

    // Reduction in shared memory
    //
    // Algorithm:
    ///
    //   Higher half (threads) does not do anything.
    //   Lower half (threads) adds to itself one higher thread's data
    //   Reduce threads to consider by 1/2 and repeat
    //
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Thread 0 writes the block's result to output
    if (cacheIndex == 0) {
        atomicAdd(output, cache[0]);
    }
}
