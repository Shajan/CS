#!/bin/bash

# prerec install valgrind (see http://valgrind.org/)
# see file massif.out.<pid> after running

if [ "$1" == "clean" ]; then
  rm -rf memleak
  rm -rf memleak.dSYM
  rm -rf massif.out.*
  exit 0
fi

if [ "$1" == "run" ]; then
  valgrind --tool=massif ./memleak
  exit 0
fi

if [ "$1" == "compile" ]; then
# Compile using C++ sources
# -g : Source level debug info
  clang++ -g -o memleak memleak.cpp
  exit 0
fi

clang++ -g -o memleak memleak.cpp
valgrind --tool=massif ./memleak

