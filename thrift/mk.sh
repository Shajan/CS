#!/bin/bash

if [ "$1" == "clean" ]; then
  rm sample
  rm -rf gen-cpp
  rm -rf sample.dSYM
  exit 0
fi

# generate cpp files
thrift --gen cpp sample.thrift

# -g : source level debug info
# thrift include folder : /usr/local/include/thrift
# boost include folder : /usr/local/include/boost
cc -g -I /usr/local/include/thrift -I /usr/local/include/boost -o sample sample.cpp

