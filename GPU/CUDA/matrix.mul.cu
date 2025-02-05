#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define WARP_COUNT 32
#define EPSILON 1e-4 // Used for checking the result
#define PERF_RUNS 20

//#define TRACE 1

#define FREE(p) if (p) free(p)
#define CUDA_FREE(p) if (p) cudaFree(p)

#define ERR_EXIT(err, api)    \
    if (err != cudaSuccess) { \
      printf("Error:CUDA:%s:%s\n", #api, cudaGetErrorString(err)); \
      goto Exit; \
    } \

// Used for perf measurements
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int compare_numbers(float* a, float* b, int size) {
  int mismatches = 0;
  for (int i = 0; i < size; i++) {
    if (fabs(a[i] - b[i]) > EPSILON) { // Expect a small difference between GPU and CPU compute
      #ifdef TRACE
      printf("Mismatch at index %d: CPU=%.6f, GPU=%.6f, Diff=%.6f\n", 
                    i, a[i], b[i], fabs(a[i] - b[i]));
      #endif
      mismatches++;
    }
  }
  #ifdef TRACE
  printf("Total Missmatches: %d\n", mismatches);
  #endif

  return mismatches == 0 ? 1 : 0;
}

int split_and_round_up(int a, int b) {
  // Split a into equal parts of b
  return (a + b - 1) / b;
}

float* allocate_and_fill(int count) {
    float *p = (float*) malloc(count * sizeof(float));

    for (int i=0; i<count; ++i) {
      p[i] = (float) rand() / RAND_MAX; // Normalize to [0,1]
    }

    return p;
}

__global__ void kernel_mul_1(float* A, float* B, float* C, int m, int x, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) return; // Prevent out-of-bounds access

    int dest = row * n + col;

    C[dest] = 0.0;
    for (int i=0; i<x; ++i) {
      C[dest] += A[row*x + i] * B[i*n + col];
    }
}

void mul_1(float* d_A, float* d_B, float* d_C, int m, int x, int n) {
    // Each thread computes one element in the output
    // Total number of threads : m * n

    int threads_per_block_x = WARP_COUNT;
    int threads_per_block_y = 8;
    int blocks_per_grid_x = split_and_round_up(m, threads_per_block_x);
    int blocks_per_grid_y = split_and_round_up(n, threads_per_block_y);

    dim3 blockDim(threads_per_block_x, threads_per_block_y);
    dim3 gridDim(blocks_per_grid_x, blocks_per_grid_y);
    
    #ifdef TRACE
    printf("Total threads needed: %d\n", m * n);
    printf("blockDim.x: %d\n", threads_per_block_x);
    printf("blockDim.y: %d\n", threads_per_block_y);
    printf("gridDim.x: %d\n", blocks_per_grid_x);
    printf("gridDim.y: %d\n", blocks_per_grid_y);
    printf("Total capacity: %d\n",
      threads_per_block_x * threads_per_block_y *
      blocks_per_grid_x * blocks_per_grid_y);
    #endif

    kernel_mul_1<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, x, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error:kernel_mul_1:%s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

void mul_cpu(float* h_A, float* h_B, float* h_C, int m, int x, int n) {
  // Create one output element at a time
  for (int i=0; i<m; ++i) {
    for (int j=0; j<n; ++j) {
      h_C[i * n + j] = 0.0; 
      for (int k=0; k<x; ++k) {
        h_C[i * n + j] += h_A[i * x + k] * h_B[k * n + j];
      }
    }
  }
}

int main() {
    // Initialize random number generator to make this code repeatable
    srand(1234);
    cudaError_t err=cudaSuccess;
    double cpu_total_time=0, gpu_total_time=0;
    double cpu_avg_time=0, gpu_avg_time=0;

    // Dimensions for A, B, C
    // A(m, x)  B(x, n) => C(m, n)
    int m = 512;
    int x = 256;
    int n = 128;

    int a_count = m * x;
    int b_count = x * n;
    int c_count = m * n;

    // 2D Matrix layed out in 1D
    float *h_A = (float*) allocate_and_fill(a_count);
    float *h_B = (float*) allocate_and_fill(b_count);
    float *h_C = (float*) malloc(c_count * sizeof(float));
    float *h_C_test = (float*) malloc(c_count * sizeof(float));


    // Allocate GPU memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    err = cudaMalloc((void **)&d_A, a_count * sizeof(float));
    ERR_EXIT(err, cudaMalloc)
    err = cudaMalloc((void **)&d_B, b_count * sizeof(float));
    ERR_EXIT(err, cudaMalloc)
    err = cudaMalloc((void **)&d_C, c_count * sizeof(float));
    ERR_EXIT(err, cudaMalloc)

    // Copy input from host to device memory
    err = cudaMemcpy(d_A, h_A, a_count * sizeof(float), cudaMemcpyHostToDevice);
    ERR_EXIT(err, cudaMemcpy)
    err = cudaMemcpy(d_B, h_B, b_count * sizeof(float), cudaMemcpyHostToDevice);
    ERR_EXIT(err, cudaMemcpy)

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
      mul_cpu(h_A, h_B, h_C_test, m, x, n);
      mul_1(d_A, d_B, d_C, m, x, n);
    }

    // Copy result from device to host
    err = cudaMemcpy(h_C, d_C, c_count * sizeof(float), cudaMemcpyDeviceToHost);
    ERR_EXIT(err, cudaMemcpy)

    if (compare_numbers(h_C, h_C_test, c_count) != 1) {
      printf("Error, result from CPU and GPU has sufficient difference!\n");
      goto Exit;
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    cpu_total_time = 0.0;
    for (int i = 0; i < PERF_RUNS; i++) {
      double start_time = get_time();
      mul_cpu(h_A, h_B, h_C_test, m, x, n);
      double end_time = get_time();
      cpu_total_time += end_time - start_time;
    }
    cpu_avg_time = cpu_total_time / PERF_RUNS;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    gpu_total_time = 0.0;
    for (int i = 0; i < PERF_RUNS; i++) {
      double start_time = get_time();
      mul_1(d_A, d_B, d_C, m, x, n);
      double end_time = get_time();
      gpu_total_time += end_time - start_time;
    }
    gpu_avg_time = gpu_total_time / PERF_RUNS;

    printf("CPU avg time: %f micro seconds\n", (cpu_avg_time * 1e6f));
    printf("GPU avg time: %f micro seconds\n", (gpu_avg_time * 1e6f));
    printf("Size %d, Speedup: %f times\n", (m * x * n), cpu_avg_time / gpu_avg_time);

  Exit:
    // Cleanup
    FREE(h_A);
    FREE(h_B);
    FREE(h_C);
    FREE(h_C_test);

    CUDA_FREE(d_A);
    CUDA_FREE(d_B);
    CUDA_FREE(d_C);

    return 0;
}

