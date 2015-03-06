#include <stdio.h>

extern void pipe();

int main(int argc, const char* argv[]) {
  printf("%d:%s\n", argc, argv[0]);
  pipe();
  return 0;
}
