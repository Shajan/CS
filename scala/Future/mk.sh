#!/bin/bash
TARGET=${1:-Futures}

scalac -feature -language:postfixOps $TARGET.scala
scala $TARGET
