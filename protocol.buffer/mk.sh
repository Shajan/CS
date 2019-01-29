mkdir ./java
protoc --java_out=./java ./sample.proto 
mkdir ./cpp
protoc --cpp_out=./cpp ./sample.proto 
