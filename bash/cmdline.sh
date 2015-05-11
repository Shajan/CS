#!/bin/bash

while getopts ":h:a:b:" OPTION
do
  case "$OPTION" in
    h)
      echo "option help"
      exit 1
      ;;
    a)
      a=${OPTARG}
      echo "option a $a"
      exit 0 
      ;;
    b)
      b=${OPTARG}
      echo "option b $b"
      exit 0 
      ;;
  esac
done

if [ -z "${a}"] && [ -z "${b}"]; then
  echo "provide vlaue for a or b"
  echo "Usage: -a abc"
  exit 1
fi

echo "a = ${a}, b= ${b}"
