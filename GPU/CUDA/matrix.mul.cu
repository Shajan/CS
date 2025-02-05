#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WARP_COUNT 32

int split_and_round_up(int a, int b) {
  // Split a into equal parts of b
  return (a + b - 1) / b;
}

__global__ void kernel_mul_1(float* A, float* B, float* C, int m, int x, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int dest = row * n + col;

    C[dest] = 0.0;
    for (int i=0; i<x; ++i) {
      C[dest] += A[row*x + i] * B[x*i + col];
    }
}

void mul_1(float* d_A, float* d_B, float* d_C, int m, int x, int n) {
    // Each thread computes one element in the output
    // Total number of threads : m * n

    int total_threads = m * n;
    int threads_per_block_x = WARP_COUNT;
    int threads_per_block_y = 8;
    int threads_per_block = threads_per_block_x * threads_per_block_y;
    int blocks_per_grid = split_and_round_up(total_threads, threads_per_block);
    int blocks_per_grid_x = 4;
    int blocks_per_grid_y = split_and_round_up(blocks_per_grid, blocks_per_grid_x);

    dim3 blockDim(threads_per_block_x, threads_per_block_y);
    dim3 gridDim(blocks_per_grid_x, blocks_per_grid_y);

    printf("Total threads needed: %d\n", total_threads);
    printf("blockDim.x: %d\n", threads_per_block_x);
    printf("blockDim.y: %d\n", threads_per_block_y);
    printf("gridDim.x: %d\n", blocks_per_grid_x);
    printf("gridDim.y: %d\n", blocks_per_grid_y);
    printf("Total capacity: %d\n", threads_per_block * blocks_per_grid_x * blocks_per_grid_y);

    kernel_mul_1<<<blockDim, gridDim>>>(d_A, d_B, d_C, m, x, n);
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

float* allocate_and_fill(int count) {
    float *p = (float*) malloc(count * sizeof(float));

    for (int i=0; i<count; ++i) {
      p[i] = (float) rand() / RAND_MAX; // Normalize to [0,1]
    }

    return p;
}

int main() {
    // Initialize random number generator to make this code repeatable
    srand(1234);

    // Dimensions for A, B, C
    // A(m, x)  B(x, n) => C(m, n)
    int m = 256;
    int x = 64;
    int n = 128;

    int a_count = m * x;
    int b_count = x * n;
    int c_count = m * n;

    // 2D Matrix layed out in 1D
    float *h_A = (float*) allocate_and_fill(a_count);
    float *h_B = (float*) allocate_and_fill(b_count);
    float *h_C = (float*) malloc(c_count * sizeof(float));

    mul_cpu(h_A, h_B, h_C, m, x, n);
    mul_1(NULL, NULL, NULL, m, x, n);
/*
    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, a_count * sizeof(float));
    cudaMalloc((void **)&d_B, b_count * sizeof(float));
    cudaMalloc((void **)&d_C, c_count * sizeof(float));

    // Copy input from host to device memory
    cudaMemcpy(d_A, h_A, a_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_A, b_count * sizeof(float), cudaMemcpyHostToDevice);

    mul_1(d_A, d_B, d_C, m, x, n);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, c_count * sizeof(float), cudaMemcpyDeviceToHost);
*/

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);

/*
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
*/

    return 0;
}

