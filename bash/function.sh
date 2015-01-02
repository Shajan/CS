#!/bin/sh

function foo() {
  local a=$1
  local b=$2
  echo "$a,$b"
}

foo Hello world!

result=$(foo x y)
echo "return val $result"
