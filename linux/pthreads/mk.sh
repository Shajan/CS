#!/bin/bash

if [ "$1" == "clean" ]; then
  rm -rf pthreads
  rm -rf pthreads.dSYM
  exit 0
fi

if [ "$1" == "run" ]; then
  ./pthreads
  exit 0
fi

rm -rf pthreads
rm -rf pthreads.dSYM

g++ -g -pthread -o pthreads pthread.cpp

./pthreads
