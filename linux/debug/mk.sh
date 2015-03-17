#!/bin/bash

if [ "$1" == "clean" ]; then
  rm -rf backtrace
  rm -rf backtrace.dSYM
  exit 0
fi

if [ "$1" == "run" ]; then
  ./backtrace
fi

g++ -g -rdynamic -o backtrace backtrace.cpp

