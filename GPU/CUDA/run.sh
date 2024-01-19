#!/bin/bash

# Check if nvcc is installed
if ! command -v nvcc &> /dev/null
then
    echo "nvcc not found. Please install CUDA toolkit."
    exit 1
fi

nvcc -o simple simple.cu

