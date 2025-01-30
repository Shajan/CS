#!/bin/bash

# Check if nvcc is installed
if ! command -v nvcc &> /dev/null
then
    echo "nvcc not found. Please install CUDA toolkit."
    exit 1
fi

# Add compatibility version for PTX binary to nvcc command if required
# Compile and /debug.cu to find the compatible version if necessary
#
#  example: -arch=sm_80

nvcc -o simple simple.cu


