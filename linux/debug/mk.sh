#!/bin/bash

if [ "$1" == "clean" ]; then
  rm -rf backtrace
  rm -rf backtrace.dSYM
  exit 0
fi

if [ "$1" == "run" ]; then
  ./backtrace
fi

# Compile using C++ sources
# -g : Source level debug info
# -I : Include folders
# -L : Lib folder
# -l : link with library
# -stdlib=libc++ : use clang headers in /usr/lib/c++/v1 example <functional>
# -rdynamic  : symbol info for glibc
clang++ -std=c++11 -stdlib=libc++ -g -rdynamic -o backtrace backtrace.cpp
#gcc -g -rdynamic -o backtrace backtrace.cpp

