import ctypes

# Load the shared library
lib = ctypes.CDLL('./libsumstr.so')

# Define the function signature
lib.sum_str.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.sum_str.restype = ctypes.c_char_p  # Python will automatically manage the memory

# Call the function
str1 = "Hello, "
str2 = "World!"
result = lib.sum_str(str1.encode('utf-8'), str2.encode('utf-8'))

# Print the result
print(f"The concatenated string is: {result.decode('utf-8')}")
