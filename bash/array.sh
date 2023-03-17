#!/bin/bash
a=('a' 'b' 'c')
bar=$(printf ",%s" "${a[@]}")
bar=${bar:1}

echo $bar
