#!/bin/bash

TARGET=${1:-sample}

if [ "$2" == "clean" ]; then
  rm -rf $TARGET
  rm -rf $TARGET.dSYM
  exit 0
fi

if [ "$2" == "run" ]; then
  ./$TARGET
fi

# Compile using C++ sources
# -g : Source level debug info
# -I : Include folders
# -L : Lib folder
# -l : link with library
# -stdlib=libc++ : use clang headers in /usr/lib/c++/v1 example <functional>
#  -std=c++11 : Use c++11
clang++ -std=c++11 -stdlib=libc++ -g -o $TARGET $TARGET.cpp

