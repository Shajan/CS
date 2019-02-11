#!/bin/bash

# Build tensorflow from source
# One time setup for Mac OSX
#
# See https://www.tensorflow.org/install/source for prereqs
# Better instructions on https://www.pyimagesearch.com/2019/01/30/macos-mojave-install-tensorflow-and-keras-for-deep-learning/
#  Install homebrew
#  Install XCode
#  Python 2.7
#    brew install python@2
#  Python 3.7
#    brew upgrade python
#  Python 3.6 [Tensorflow 1.12 requires this version]
#    https://stackoverflow.com/questions/51125013/how-can-i-install-a-previous-version-of-python-3-in-macos-using-homebrew/51125014#51125014
#    brew unlink python
#    brew install  --debug --ignore-dependencies https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb
#    To switch between versions (brew info python)
#    brew switch python 3.6.5_1
#    brew switch python 3.7.0
#  pip3 install virtualenv
set -e

source ./env.vars.sh

# Create folders
mkdir -p ${WORKSPACE}/${TF_NAME}
mkdir -p ${WORKSPACE}/${TF_NAME}/${TF_VERSION}

# Create python virtual env for this version
cd ${TF_BASE}
virtualenv -p python3 ${TF_ENV}

# Switch to python virtualenv
source ${TF_BASE}/${TF_ENV}/bin/activate

# if pip install gives SSL warnings try
# curl https://bootstrap.pypa.io/get-pip.py | python

# Install python packages
pip3 install -U pip six numpy wheel mock
pip3 install -U keras_applications==1.0.6 --no-deps
pip3 install -U keras_preprocessing==1.0.5 --no-deps

# Install bazel
#   bazel version 0.18.0 for tf 1.12.0
#   https://github.com/tensorflow/tensorflow/issues/23401
cd ${TF_BASE}
curl -L https://github.com/bazelbuild/bazel/releases/download/0.18.1/bazel-0.18.1-installer-darwin-x86_64.sh -o bazel-0.18.1-installer-darwin-x86_64.sh
chmod +x ./bazel-0.18.1-installer-darwin-x86_64.sh
./bazel-0.18.1-installer-darwin-x86_64.sh --prefix=${TF_BASE}/${TF_ENV}
bazel version
bazel shutdown

# Get tensorflow source code
cd ${TF_BASE}
git clone --branch ${TF_VERSION} https://github.com/tensorflow/tensorflow.git

cd ${TF_SRC}

# Note about this step
#   Make sure python3 is used - `brew upgrade python`
#   Ignore the following uptions unless dependencies are installed
./configure

cd ${TF_SRC}
# Test source setup
bazel test -c opt -- //tensorflow/... -//tensorflow/compiler/... -//tensorflow/contrib/lite/... 2>&1 | tee ../build.log.1
bazel test -c opt -- //tensorflow/... -//tensorflow/compiler/... -//tensorflow/lite/... 2>&1 | tee ../build.log.2
