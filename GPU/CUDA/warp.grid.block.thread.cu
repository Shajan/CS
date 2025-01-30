#include <stdio.h>

// Credit
//   https://github.com/Infatoshi/cuda-course/blob/master/05_Writing_your_First_Kernels/01%20CUDA%20Basics/01_idxing.cu

///////////////////////////////////////////////////////////
// Print all the different parition parameters
//
// __global__ defines a kernel function
//
///////////////////////////////////////////////////////////
__global__ void kernel_trace(void) {
    // SM:
    //   Streaming Multiprocessor, a set of CUDA Cores, Tensor Cores and L1 shared memory
    //   H100 : 128 CUDA cores, 4 tensor cores, 256KB L1/shared memory
    //   CUDA Core (ALU):
    //     Performs integer & floating-point arithmetic 
    //   Tensor Core:
    //     Performs matrix-multiplication
    // Thread:
    //   One unit of execution control flow
    // Warp:
    //   Composed of 32 threads, the smallest unit for scheduling on an SM
    //   If a warp ends up having less than 32 threads, rest of the GPUs idle. 
    //   If a warp is blocked on memory, other warps get scheduled.
    //   Warp scheduler:
    //     Schedules work on an SM, part of an SM
    // Block:
    //   Can share data across threads by declaring gpu memory as __shared__
    //   Ideally a block is a multiple of 32 threads
    // Grid:
    //   Composed of multiple blocks
    //
    // Programming:
    //   Break a parallelizable problem into a compute (a kernel function) that operates on small portion of data
    //     Data needs to partitionable so that kernel works on one small portion of data
    //     A kernel works on it's small portion of data
    //   Partitioning:
    //     There are 6 dimensions that can be used to partition data (3 inside blocks and 3 outside)
    //     These dimensions allows for creating a 'partitioned view' of contigious data
    //       Consider data locality when selecting the partitioning scheme
    //     If data sharing is necessary across threads, use block level shared memory
    //       Using global memory is slow use shared memory where possible
    //       Block size needs to be carefully designed as shared memory is limitted
    //   Kernel:
    //     When defining a kernel, define the logic as well and expected paritioning scheme
    //     Compile the kernel into a binary, which will have a few parts
    //       Code that runs on the CPU that schedules the kernel
    //       Kernel that runs on the GPU
    //   Execution:
    //     On CPU
    //       Allocate memory on device and copy data from host (CPU RAM) to device (GPU VRAM)
    //       Schedule the kernel and wait
    //       Copy results from device to host and free memory on device
    //     On GPU
    //       Execute the kernel

    // This kernel makes the following assumptions
    // Data is layed out in a linear fasion and is partitioned by 6 paramters (blockIdx.x, y, z), (threadIdx.x, y, z)
    // The number of threads per block dimension is blockDim.x, y, z
    // The number of blocks per grid dimension is gridDim.x, y, z

    // Position of this block
    int block_idx =
        blockIdx.x +
        blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;

    // Offset of this block from start of data, considering block size 
    int block_offset =
        block_idx *
        blockDim.x * blockDim.y * blockDim.z;

    // Position of this thread within the block
    int thread_offset =
        threadIdx.x +  
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    // Unique idx of data this function needs to process
    // Use this as an index into the input data buffer and process input starting here
    int id = block_offset + thread_offset;

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_idx,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main(int argc, char **argv) {
    // We get to decide the dimensions
    const int b_x = 2, b_y = 3, b_z = 4; // How different blocks are partitioned
    const int t_x = 4, t_y = 4, t_z = 4; // Partions within a block, block size is 64 (4*4*4)

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z); // 3d cube of shape 2*3*4 = 24
    dim3 threadsPerBlock(t_x, t_y, t_z); // 3d cube of shape 4*4*4 = 64

    // Note that instead of dim3, we can also use a single integer
    //   When a sigle integer 'n' is used, partions become ('n', 1, 1)
    kernel_trace<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}
