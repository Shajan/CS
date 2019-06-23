#!/bin/bash
# See https://hub.docker.com/r/centos/devtoolset-6-toolchain-centos7/

# Login to docker
docker login

# Pull down centos images
docker pull centos/devtoolset-6-toolchain-centos7

# Create docker image with name 'centos7.jdk8'
docker build - --tag=centos7.jdk8.jni < Dockerfile.jni
