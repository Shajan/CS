#!/bin/sh
 
# location of jni.h
JNI_H_PATH=/System/Library/Frameworks/JavaVM.framework/Versions/Current/Headers

javac HelloWorld.java
javah HelloWorld
gcc -shared HelloWorld.c -o HelloWorld.so -I$JNI_H_PATH
java HelloWorld
