#!/bin/bash

FOLDER=/jni_sample
# Start container, open interactive shell
docker run -t -d --name jni_sample centos7.jdk8.jni /bin/bash

docker exec -it jni_sample mkdir ${FOLDER}

docker cp mk.sh jni_sample:${FOLDER}
docker cp HelloWorld.java jni_sample:${FOLDER}/
docker cp HelloWorld.c jni_sample:${FOLDER}/

docker exec -it jni_sample /bin/bash -c "cd ${FOLDER}; ./mk.sh"

# Copy binaries back to host
docker cp jni_sample:${FOLDER}/HelloWorld.so .
docker cp jni_sample:${FOLDER}/HelloWorld.class .

# Cleanup
docker stop jni_sample
docker container prune
