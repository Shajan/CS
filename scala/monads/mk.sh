#!/bin/bash
TARGET=${1:-Monads}

scalac -feature -language:postfixOps $TARGET.scala
scala $TARGET
