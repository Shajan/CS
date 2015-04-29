/*
 * References
 *  https://www.gnu.org/software/libc/manual/html_node/Allocation-Debugging.html
 *  http://valgrind.org/docs/manual/ms-manual.html
 */
#include <stdlib.h>

void leakA() {
  malloc(40000);
}

void leakB() {
  malloc(2000);
  leakA();
}

int main(void) {
  int i;
  void* a[10];

  for (i = 0; i < 10; i++) {
     a[i] = malloc(1000);
  }

  leakA();
  leakB();

  for (i = 0; i < 10; i++) {
     free(a[i]);
  }

  return 0;
}
