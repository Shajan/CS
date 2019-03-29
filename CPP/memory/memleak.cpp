/*
 * References : Using valgrind - runtime only checks
 *  https://www.gnu.org/software/libc/manual/html_node/Allocation-Debugging.html
 *  http://valgrind.org/docs/manual/ms-manual.html
 *
 * Reference : Using mtrace (not required for valgrind) - need code change at compile time
 *  http://man7.org/linux/man-pages/man3/mtrace.3.html
 */
#include <stdlib.h>

#ifdef M_CHECK
#include <mcheck.h>
#else
#define mtrace() 
#endif

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

  mtrace();
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
