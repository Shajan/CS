#!/bin/bash

if [ "$1" == "clean" ]; then
  rm -f sample
  rm -f data.bin
  rm -rf gen-cpp
  rm -rf gen-java
  rm -f *.class
  rm -rf sample.dSYM
  exit 0
fi

# generate cpp & java files
thrift -r --gen cpp --gen java sample.thrift

# Compile using C++ sources
# -g : Source level debug info
# -I : Include folders
# -L : Lib folder
# -l : Thrift lib
clang++ -g -I /usr/local/include/thrift -I /opt/twitter/Cellar/boost/1.54.0/include/boost/ -o sample sample.cpp gen-cpp/sample_types.cpp -L/opt/twitter/Cellar/boost/1.54.0/lib -lthrift

# Compile using java sources
javac -cp .:/usr/local/lib/libthrift-0.9.1.jar:/usr/local/lib/slf4j-api-1.5.8.jar gen-java/serializer/KeyVal.java ./Sample.java
java -cp .:./gen-java:/usr/local/lib/libthrift-0.9.1.jar:/usr/local/lib/slf4j-api-1.5.8.jar Sample

