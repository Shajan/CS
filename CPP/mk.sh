#!/bin/bash

if [ "$1" == "clean" ]; then
  rm -rf sample
  rm -rf sample.dSYM
  rm -rf test.data
  exit 0
fi

if [ "$1" == "run" ]; then
  rm -rf test.data
  ./sample
fi

# Compile using C++ sources
# -g : Source level debug info
# -I : Include folders
# -L : Lib folder
# -l : link with library
clang++ -g -o sample sample.cpp

