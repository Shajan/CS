#!/bin/sh 
#Check number of command line args
[ "$#" -eq 3 ] || { echo "Usage :" `basename $0` "YYYY MM DD" >&2; exit 1; }

#Get command line args
YEAR=$1
MONTH=$2
DAY=$3 

#Single digits left padded with 0
for HOUR in 0{0..9} {10..11}
do
  echo "$YEAR/$MONTH/$DAY $HOUR"
done
