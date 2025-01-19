#include <Python.h>
#include <string.h>

// Function to concatenate two strings
char* sum_str(const char* str1, const char* str2) {
    size_t len1 = strlen(str1);
    size_t len2 = strlen(str2);

    // Allocate memory using Python's allocator
    char* result = (char*)PyMem_Malloc(len1 + len2 + 1);
    if (!result) {
        return NULL; // Return NULL if allocation fails
    }

    strcpy(result, str1);
    strcat(result, str2);

    return result; // Python will manage the memory
}

