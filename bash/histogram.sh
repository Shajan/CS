awk '{n[$2]++} END {for (i in n) print n[i],i}' | sort -n
