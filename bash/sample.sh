#!/bin/sh 
#Check number of command line args
#set -x #echo lines for debugging
#[ "$#" -eq 3 ] || { echo "Usage :" `basename $0` "YYYY MM DD" >&2; exit 1; }

#Get command line args, provide detaults
YEAR=${1:-2014}
MONTH=${2:-11}
DAY=${3:-12}

function sequence() 
{
  START=5
  END=10
  for i in $(seq $START $END); do echo $i; done
}

function for_with_variable()
{
  echo "for loops in bash"

  # prints 5 6 7 8 9
  for i in {5..9}
  do
    echo $i
  done

  # prints {5..9}, this does not work as expected
  START=5
  END=9
  for i in {$START..$END}
  do
    echo $i
  done

  # Use c style instead, prints 5, 6, 7, 8, 9
  for (( i=$START; i<=$END; i++ ))
  do
    echo $i
  done
}

function sum() 
{
  #Single digits left padded with 0
  for HOUR in 0{0..9} {10..11}
  do
    echo "$YEAR/$MONTH/$DAY $HOUR"
    SUM=0
    for VER in {13..14}
    do
        CUR=`egrep "foo" input.txt | wc -l`
        SUM=`expr $SUM + $CUR`
    done
    RESULT="$RESULT$SUM "
  done
}

function foo_bar()
{
  local var1=$1
  echo $var1
  if [ "$var1" == "1971" ]
  then
    RESULT="$var1"
  else
    RESULT="test"
  fi
}

RESULT=""
foo_bar $YEAR
echo $RESULT

#String operations ...........................
STR="Hello World!"

#String length
#Operator #
echo "Length of '$STR' : ${#STR}"

#Substring based on position
#Operator :
echo "Substring of '$STR' starting at 6th char: ${STR:6}"
echo "Substring of '$STR' starting at 6th char, length 3 chars: ${STR:6:3}"

STR="foo/bar/baz.txt"
#Substring based on '*' pattern
#Operators # % ## %%
echo "Drop chars before first '/' in '$STR' : ${STR#*/}"
echo "Drop chars after last '/' in '$STR' : ${STR%/*}"

echo "Drop chars before last '/' in '$STR' : ${STR##*/}"
echo "Drop chars after first '/' in '$STR' : ${STR%%/*}"

sequence


for_with_variable
