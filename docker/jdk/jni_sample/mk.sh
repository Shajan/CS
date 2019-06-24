#!/bin/sh
  
# location of jni.h
JNI_H_PATH=${JAVA_HOME}include

javac HelloWorld.java
javah HelloWorld
gcc -fPIC -shared HelloWorld.c -o HelloWorld.so -I${JNI_H_PATH} -I${JNI_H_PATH}/linux
java HelloWorld

