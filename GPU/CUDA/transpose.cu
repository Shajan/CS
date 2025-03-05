#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_transpose(const float* input, float* output, int rows, int cols) {
    int input_col = blockIdx.x * blockDim.x + threadIdx.x;
    int input_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (input_col >= cols || input_row >= rows)
        return;
    int output_col = input_row;
    int output_row = input_col;
    int output_cols = rows;
    int output_rows = cols;
    output[output_row * output_cols + output_col] = input[input_row * cols + input_col];
}

#define TILE_SIZE 16  // Adjust this based on your GPU architecture

__global__ void matrix_transpose_shared(const float* input, float* output, int rows, int cols) {
    // Expectation:
    // Read row order (contigious memory)
    // Keep content in shared memory, wait for all threads for one TILE_SIZE (size of one block)
    // Write row order (contigious in target memory)
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // Padding to avoid bank conflicts

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load data from global memory into shared memory in row order of 'input'
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();  // Ensure all threads have loaded data before transposing

    // Compute transposed indices
    int transposed_x = blockIdx.y * TILE_SIZE + threadIdx.x;
    int transposed_y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Store transposed data back to global memory, in row order of 'output'
    if (transposed_x < rows && transposed_y < cols) {
        output[transposed_y * rows + transposed_x] = tile[threadIdx.x][threadIdx.y];
    }
}

void solve(const float* input, float* output, int rows, int cols) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch the kernel
    //matrix_transpose<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    matrix_transpose_shared<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);

    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
