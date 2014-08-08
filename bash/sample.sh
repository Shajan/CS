#!/bin/sh 
#Check number of command line args
#set -x #echo lines for debugging
[ "$#" -eq 3 ] || { echo "Usage :" `basename $0` "YYYY MM DD" >&2; exit 1; }

#Get command line args
YEAR=$1
MONTH=$2
DAY=$3 

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

function foo_bar()
{
  local var1=$1
  echo $var1
  RESULT="test"
}

RESULT=""
foo_bar abc
echo $RESULT

