mkdir -p ./java
protoc --java_out=./java ./sample.proto 
mkdir -p ./cpp
protoc --cpp_out=./cpp ./sample.proto 
mkdir -p ./python
protoc --python_out=./python ./sample.proto 
