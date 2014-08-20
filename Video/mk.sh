#!/bin/bash

BIN_FOLDER=~/tmp/del/h264
DATA_FOLDER=$BIN_FOLDER

function clean {
  rm -rf h264
  rm -rf h264.dSYM
  rm -rf $DATA_FOLDER/output.*
}

function build {
  # -g : Source level debug info
  cc -g -o h264 h264.c
}

function run {
  # Get ffmpeg from https://www.ffmpeg.org/
  # Convert an mp4 video from h.264 to SQCIF (128×96) planar YUV format sampled at 4:2:0:
  $BIN_FOLDER/ffmpeg -i $DATA_FOLDER/input.mp4 -s sqcif -pix_fmt yuv420p $DATA_FOLDER/output.yuv
  ./h264 $DATA_FOLDER/output.yuv $DATA_FOLDER/output.264

  # Use ffmpeg to copy the raw h.264 NAL units into an MP4 file
  $BIN_FOLDER/ffmpeg -f h264 -i $DATA_FOLDER/output.264 -vcodec copy $DATA_FOLDER/output.mp4
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
