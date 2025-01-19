prereq() {
  # Linux Ubuntu
  echo sudo apt install gcc
 
  # Mac
  echo xcode-select --install

  # Windows
  echo MinGW
}

compile() {
  # Linux/Mac
  gcc -shared -o libsum.so -fPIC sum.c

  # Windows:
  # gcc -shared -o sum.dll -Wl,--out-implib,libsum.a sum.c
}

clean() {
  rm libsum.so 
}

# step 1: compile the sum.c into libsum.so
compile

# Step 2: Run
python use_sum.py

# Optional : Clean everything up
clean
