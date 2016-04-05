#!/bin/bash

echo "Number of params not counding 0 $#"
echo "All params except 0:"
echo "$*"    # "$1" "$2"..
echo $*      # $1 $2 ... 

echo "All params except 0: $@"

echo "0: $0"
echo "1: $1"
echo "2: $2"

shift

echo "After shift"
echo "0: $0"
echo "1: $1"
echo "2: $2"

echo "for arg loop"
for arg
do
  echo "$arg"
done

echo "null check"
while [ "$1" ]
do
    echo "$1"
    shift
done

# Comment out earlier loops to make following work because of shift

echo "Better null check"
while [ "${1+defined}" ];
do
  echo "$1"
  shift
done

echo "C Style for loop"
numargs=$#
for ((i=1; i<=numargs; i++))
do
    echo "$1"
    shift
done

