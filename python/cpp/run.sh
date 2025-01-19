prereq() {
  # Linux Ubuntu
  echo sudo apt install gcc
 
  # Mac
  echo xcode-select --install

  # Windows
  echo MinGW
}

config() {
  echo "Includes"
  python3-config --includes
  # /opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/include/python3.12

  echo "link"
  python3-config --ldflags
  # /opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/config-3.12-darwin

  python3.12-config --cflags --ldflags
  # .. -lpython3.12 -ldl -framework CoreFoundation
}

compile() {
  # Linux/Mac
  gcc -shared -o libsum.so -fPIC sum.c
  gcc -shared -o libsumstr.so -fPIC sum_str.c

  # Mac with python libraries linked in
  gcc -shared -o py_mem.so -fPIC \
    -I/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/include/python3.12 \
    -L/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/config-3.12-darwin \
    -lpython3.12 -ldl -framework CoreFoundation \
    py_mem.c

  # Windows:
  # gcc -shared -o sum.dll -Wl,--out-implib,libsum.a sum.c
}

debug() {
  # Print all the functions in the compiled binary

  echo "libsum.so:"
  nm libsum.so | egrep T
  #otool -v -s __TEXT __text libsum.so

  echo "libsumstr.so:"
  nm libsumstr.so | egrep T

  echo "py_mem.so:"
  nm py_mem.so | egrep T
}

clean() {
  rm libsum.so 
  rm libsumstr.so 
  rm py_mem.so 
}

# Step 0: 
#prereq
#config

# step 1: compile the sum.c into libsum.so
compile

# Step 1.1 : Optional, print the exports
debug

# Step 2: Run
python use_sum.py
python use_sum_str.py
python py_mem.py

# Optional : Clean everything up
clean
