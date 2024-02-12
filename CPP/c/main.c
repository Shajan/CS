#include <stdio.h>

#include "vector.h"

int main(int argc, char* argv[]) {
  Vector *v = create_int_vector();

  for (int i=0; i<100; ++i) {
    vector_append_int(v, i);
    int a = vector_get_int(v, i);
    printf("[%d:%d] len:%d capacity:%d\n", i, a, vector_len(v), vector_capacity(v));
  }

  vector_free(v);
  return 0;
}

