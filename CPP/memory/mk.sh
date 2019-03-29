#!/bin/bash
set -e

# prerec install valgrind (see http://valgrind.org/)
# example usages:
#  valgrind --tool=massif ./memleak
#    see file massif.out.<pid> after running
#    ms_print massif.out.<pid> for better readability
#  valgrind --tool=memcheck --leak-check=full ./memleak
#
# Compile using C++ sources
# -g : Source level debug info

if [ "$1" == "init" ]; then
  wget https://raw.githubusercontent.com/lattera/glibc/master/malloc/mtrace.pl
fi

if [ "$1" == "clean" ]; then
  rm -rf memleak.valgrind
  rm -rf memleak.mtrace
  rm -rf memleak.dSYM
  rm -rf massif.out.*
  exit 0
fi

if [ "$1" == "valgrind" ]; then
  clang++ -g -o memleak.valgrind memleak.cpp
  #valgrind --tool=massif ./memleak
  valgrind --tool=memcheck --leak-check=full ./memleak
  exit 0
fi

if [ "$1" == "mtrace" ]; then
  # run on linux
  gcc -g -DM_CHECK=1 -o memleak.mtrace memleak.cpp
  export MALLOC_TRACE=./mtrace.txt
  ./memleak.mtrace
  # The downloaded perl file does not appear to work
  perl ./mtrace.pl ./memleak.mtrace $MALLOC_TRACE
  exit 0
fi
