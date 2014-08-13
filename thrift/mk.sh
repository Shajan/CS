#!/bin/bash

if [ "$1" == "clean" ]; then
  rm sample
  rm -rf gen-cpp
  rm -rf sample.dSYM
  exit 0
fi

# generate cpp files
thrift --gen cpp sample.thrift

# -g : Source level debug info
# -I : Include folders
# -L : Lib folder
# -l : Thrift lib
clang++ -g -I /usr/local/include/thrift -I /opt/twitter/Cellar/boost/1.54.0/include/boost/ -o sample sample.cpp gen-cpp/sample_types.cpp -L/opt/twitter/Cellar/boost/1.54.0/lib -lthrift

