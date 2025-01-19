import ctypes

# Load the shared library
lib = ctypes.CDLL('./libsum.so')  # On Windows, use './sum.dll'

# Define the function signature
lib.add.argtypes = [ctypes.c_int, ctypes.c_int]
lib.add.restype = ctypes.c_int

# Call the function
result = lib.add(3, 5)
print(f"The sum of 3 and 5 is: {result}")

