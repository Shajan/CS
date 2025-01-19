# This does not appear to be able to print the function names

import ctypes

libs = [
  './libsum.so',
  './libsumstr.so',
  './py_mem.so']

for lib_file_name in libs: 
  # Load the shared library
  lib = ctypes.CDLL(lib_file_name)

  # List all available functions and symbols
  functions = dir(lib)

  # Print the functions
  functions = [func for func in functions if not func.startswith('__')]
  print(f"{lib_file_name}: {functions}")

