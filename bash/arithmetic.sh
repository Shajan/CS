count=0
for i in {0..10}
do
  if [ `expr $i % 3` -eq '0' ]
  then
    echo $i
    count=$((count + 1))
  fi
done
echo "count = $count"
