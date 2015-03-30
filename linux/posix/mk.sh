#!/bin/bash

if [ "$1" == "clean" ]; then
  rm -rf posix
  rm -rf posix.dSYM
  exit 0
fi

if [ "$1" == "run" ]; then
  ./posix
  exit 0
fi

rm -rf posix
rm -rf posix.dSYM

g++ -g -o posix posix.cpp

./posix
