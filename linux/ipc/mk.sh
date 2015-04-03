#!/bin/bash

if [ "$1" == "clean" ]; then
  rm -rf benchmark
  rm -rf benchmark.dSYM
  exit 0
fi

if [ "$1" == "run" ]; then
  ./benchmark
  exit 0
fi

rm -rf benchmark
rm -rf benchmark.dSYM
# Compile using C++ sources
# -g : Source level debug info
# -I : Include folders
# -L : Lib folder
# -l : link with library
# -stdlib=libc++ : use clang headers in /usr/lib/c++/v1 example <functional>
#  -std=c++11 : Use c++11
#g++ -std=c++11 -stdlib=libc++ -g -o benchmark -pthread benchmark.cpp backtrace.cpp common.cpp message.cpp payload.cpp stdio.cpp pipe.cpp memmap.cpp mutex.cpp ipc.cpp
#g++ -g -o benchmark -lrt benchmark.cpp backtrace.cpp common.cpp message.cpp payload.cpp stdio.cpp pipe.cpp memmap.cpp mutex.cpp ipc.cpp
g++ -g -o benchmark benchmark.cpp backtrace.cpp common.cpp message.cpp payload.cpp stdio.cpp pipe.cpp memmap.cpp mutex.cpp ipc.cpp

./benchmark
