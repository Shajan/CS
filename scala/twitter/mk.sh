#!/bin/bash
TARGET=${1:-TwitterFuture}

scalac -classpath ~/tmp/jars/com.twitter-util-core_2.10.jar -feature -language:postfixOps $TARGET.scala
scala -classpath ~/tmp/jars/com.twitter-util-core_2.10.jar:. $TARGET
