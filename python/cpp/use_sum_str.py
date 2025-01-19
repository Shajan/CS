import ctypes

# Load the shared library
lib = ctypes.CDLL('./libsumstr.so')

# Define function signatures
lib.sum_str.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.sum_str.restype = ctypes.c_void_p  # Use c_void_p to prevent automatic memory management

lib.free_str.argtypes = [ctypes.c_void_p]
lib.free_str.restype = None

# Call the function
str1 = "Hello, "
str2 = "World!"
result_ptr = lib.sum_str(str1.encode('utf-8'), str2.encode('utf-8'))

# Convert the pointer to a Python string
result = ctypes.cast(result_ptr, ctypes.c_char_p).value.decode('utf-8')
print(f"The concatenated string is: {result}")

# Free the memory allocated by the C function
lib.free_str(result_ptr)
