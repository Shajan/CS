# Either source './setenv.sh' or '. ./setenv.sh' from bash

# Make sure CUDA env variables are set - for using GPU
#export PATH=/Developer/NVIDIA/CUDA-8.0/bin:${PATH}
#export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib:${DYLD_LIBRARY_PATH}
# Verify cuda is loaded
#if [[ $(kextstat | grep -i cuda | wc -l) -lt 1 ]]
#then
#  echo "CUDA not loaded"
#  exit 1
#fi

source ~/tensorflow/bin/activate
