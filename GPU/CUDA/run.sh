#!/bin/bash

# Check if nvcc is installed
if ! command -v nvcc &> /dev/null
then
    echo "nvcc not found. Please install CUDA toolkit."
    exit 1
fi

# Add compatibility version for PTX binary to nvcc command if required
# Compile debug.cu and run to find the compatible version if necessary
#
# A100
#  -arch=sm_80
# T4
#  -arch=sm_75

nvcc -o simple simple.cu

# Profiling
#
# nvcc -o profile profile.cu -lnvToolsExt
# nsys profile --stats=true -o profile ./profile
# nsys stats profile.nsys-rep
# nsys stats --report cuda_gpu_trace profile.nsys-rep
