// sum.c
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int add(int a, int b);

#ifdef __cplusplus
}
#endif

int add(int a, int b) {
    return a + b;
}

