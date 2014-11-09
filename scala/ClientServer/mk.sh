#!/bin/bash
TARGET=${1:-ClientServer}

scalac -feature -language:postfixOps $TARGET.scala
scala $TARGET
