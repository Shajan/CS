#!/bin/bash
# See https://hub.docker.com/_/centos/

easy_create() {
  # Method (1) easy to do
  # Pull down centos images
  docker pull centos
}

manual_create() {
  # Method (2) manually
  # Copy raw centos7 build
  curl https://buildlogs.centos.org/centos/7/docker/CentOS-7-20140625-x86_64-docker_01.img.tar.xz

  # Copy Dockerfile for centos7
  curl -o Dockerfile https://raw.githubusercontent.com/CentOS/sig-cloud-instance-images/7c2e214edced0b2f22e663ab4175a80fc93acaa9/docker/Dockerfile

  # Create image on localhost
  docker build -t sd/centos7 .
}

# Login to docker
docker login

# Create image
easy_create

# Start container, open interactive shell
docker run -it centos:7
