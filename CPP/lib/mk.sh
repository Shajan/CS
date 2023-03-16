#!/bin/bash

LIB=cpplib
MAIN=main

if [ "$1" == "clean" ]; then
  rm ./bin/static/*
  rm ./bin/shared/*
  rm ./bin/slib/*
  rm ./bin/dlib/*
  rm ./bin/*
  rm ./lib/*
  exit 0
fi

# https://renenyffenegger.ch/notes/development/languages/C-C-plus-plus/GCC/create-libraries/index

# Compile obj file
gcc -I ./inc -c ./libsrc/${LIB}.cpp -o ./bin/static/${LIB}.o

# Object files for shared libraries need to be compiled as position independent
# code (-fPIC) because they are mapped to any position in the address space.
gcc -I ./inc -c -fPIC ./libsrc/${LIB}.cpp -o ./bin/shared/${LIB}.o

# Create static library using archiver (ar)
ar rcs ./bin/slib/${LIB}.a ./bin/static/${LIB}.o

# Create dynamic library
gcc -shared ./bin/shared/${LIB}.o -o ./bin/dlib/${LIB}.so
cp  ./bin/dlib/${LIB}.so ./lib
nm ./lib/${LIB}.so

# Create main executables
gcc -c -I ./inc test/${MAIN}.cpp -o bin/${MAIN}.o

# Link statically
gcc ./bin/${MAIN}.o ./bin/slib/${LIB}.a -o ./bin/s${MAIN}

# Execute static binary
./bin/s${MAIN}

# Link dynamically
gcc ./bin/${MAIN}.o ./bin/dlib/${LIB}.so -o ./bin/d${MAIN}

# Execute static binary
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH ./bin/d${MAIN}

# NOTE!!!
#
# When using '-l' option, the library file is expeced to have a 'lib' prefix
# Example: -lfoo to import library, ld looks for 'libfoo.a' or 'libfoo.so' 
