#!/bin/bash

if [ "$1" == "clean" ]; then
  rm -f sample
  rm -f data.bin
  rm -rf gen-cpp
  rm -rf gen-java
  rm -rf gen-scala
  rm -f *.class
  rm -rf sample.dSYM
  rm -rf serializer
  exit 0
fi

rm -f ./data.bin

# Write and read using C++
if [ "$1" == "c2c" ]; then
  ./sample write
  ./sample
  exit 0
fi
# Write and read using java
if [ "$1" == "j2j" ]; then
  java -cp .:./gen-java:/usr/local/lib/libthrift-0.9.1.jar:/usr/local/lib/slf4j-api-1.5.8.jar:/usr/local/lib/slf4j-log4j12-1.5.8.jar:/usr/local/lib/log4j-1.2.14.jar Sample write
  java -cp .:./gen-java:/usr/local/lib/libthrift-0.9.1.jar:/usr/local/lib/slf4j-api-1.5.8.jar:/usr/local/lib/slf4j-log4j12-1.5.8.ja:/usr/local/lib/log4j-1.2.14.jar  Sample
  exit 0
fi
# Write using C++ read using java
if [ "$1" == "c2j" ]; then
  ./sample write
  java -cp .:./gen-java:/usr/local/lib/libthrift-0.9.1.jar:/usr/local/lib/slf4j-api-1.5.8.jar:/usr/local/lib/slf4j-log4j12-1.5.8.ja:/usr/local/lib/log4j-1.2.14.jar  Sample
  exit 0
fi
# Write using java read using C++
if [ "$1" == "j2c" ]; then
  java -cp .:./gen-java:/usr/local/lib/libthrift-0.9.1.jar:/usr/local/lib/slf4j-api-1.5.8.jar:/usr/local/lib/slf4j-log4j12-1.5.8.jar:/usr/local/lib/log4j-1.2.14.jar Sample write
  ./sample
  exit 0
fi

if [ "$1" == "s" ]; then
  scala Sample
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

# Generate scala files
# Assuming scrooge is present at ~/src/tools/scrooge
SCROOGE=~/src/tools/scrooge
cp sample.thrift $SCROOGE
if [ -d $SCROOGE ]; then 
  pushd ~/src/tools/scrooge
  ./sbt "scrooge-generator/run-main com.twitter.scrooge.Main sample.thrift"
  rm sample.thrift
  popd
  rm -rf gen-scala
  mkdir gen-scala
  mv $SCROOGE/serializer gen-scala 
# see http://randonom.com/blog/tag/scrooge/
#  scalac -cp .:$SCROOGE/target/scala-2.10/scrooge_2.10-3.16.3.jar gen-scala/serializer/KeyVal.scala Sample.scala
else
  echo "Unable to find scrooge at $SCROOGE see https://github.com/Shajan/CS/blob/master/thrift/readme.txt"
fi

