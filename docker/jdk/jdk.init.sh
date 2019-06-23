#!/bin/bash
# See https://hub.docker.com/_/openjdk

# Login to docker
docker login

# Pull down centos images
docker pull openjdk

# Create docker image with name 'centos7.jdk8'
docker build - --tag=centos7.jdk8 < Dockerfile.jdk
