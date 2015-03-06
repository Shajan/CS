#!/bin/bash

if [ "$1" == "clean" ]; then
  rm -rf benchmark
  rm -rf benchmark.dSYM
  exit 0
fi

if [ "$1" == "run" ]; then
  ./benchmark
fi

# Compile using C++ sources
# -g : Source level debug info
# -I : Include folders
# -L : Lib folder
# -l : link with library
# -stdlib=libc++ : use clang headers in /usr/lib/c++/v1 example <functional>
#  -std=c++11 : Use c++11
clang++ -std=c++11 -stdlib=libc++ -g -o benchmark benchmark.cpp pipe.cpp

