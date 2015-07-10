#!/bin/bash
TARGET=${1:-TwitterFuture}

JAR_ROOT=~/tmp/jars
JARS=$JAR_ROOT/com.twitter-util-core_2.10.jar:$JAR_ROOT/com.twitter-finagle-core_2.10.jar:$JAR_ROOT/netty-3.10.1.Final.jar

scalac -classpath $JARS -feature -language:postfixOps $TARGET.scala
scala -classpath $JARS:. $TARGET
