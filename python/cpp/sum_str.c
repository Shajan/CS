// sum_str.c
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

char* sum_str(const char* str1, const char* str2);
void free_str(char* str);

#ifdef __cplusplus
}
#endif

char* sum_str(const char* str1, const char* str2) {
    size_t len1 = strlen(str1);
    size_t len2 = strlen(str2);
    char* result = (char*)malloc(len1 + len2 + 1); // +1 for null terminator

    if (!result) {
        return NULL; // Return NULL if allocation fails
    }

    strcpy(result, str1);
    strcat(result, str2);

    return result; // Memory is allocated here and should be freed later
}

void free_str(char* str) {
    free(str);
}

