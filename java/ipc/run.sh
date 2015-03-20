#Usage : ipc [-client] [-stdio|-memmap] [-debug] [-direct] [-nio] [-p port] [-f fileName] [-b bufferSize] [-m dataMultiplier] [-i iterations]

echo "Effect of stream vs nio vs direct buffer"
echo "streams"
java ipc -stdio -i 5 -b 100000
echo "nio"
java ipc -stdio -nio -i 5 -b 100000
echo "direct buffer"
#java ipc -stdio -direct -nio -i 5 -b 100000

echo "Effect of buffer size"
for bufferSize in 10000 100000 1000000
do
  echo Buffer size : $bufferSize
  #java ipc -stdio -i 5 -b $bufferSize
  #java ipc -memmap -f ./data -i 5 -b $bufferSize
done
