#!/bin/bash
TARGET=${1:-Sample}

scalac -feature -language:postfixOps $TARGET.scala
scala $TARGET $2
