#Usage : ipc [-client] [-stdio|-memmap] [-debug] [-f fileName] [-b bufferSize] [-m dataMultiplier] [-i iterations]
for bufferSize in 10000 100000 1000000
do
  echo Buffer size : $bufferSize
  java ipc -stdio -i 5 -b $bufferSize
  java ipc -memmap -f ./data -i 5 -b $bufferSize
done
