#!/bin/bash

function clean {
  rm -rf h264
  rm -rf h264.dSYM
}

function build {
  # -g : Source level debug info
  cc -g -o h264 h264.c
}

function run {
  ./h264
}

if [ "$1" == "clean" ]; then
  clean
  exit 0
fi

if [ "$1" == "build" ]; then
  build
  exit 0
fi

if [ "$1" == "run" ]; then
  run
  exit 0
fi

# Default
clean
build
run
