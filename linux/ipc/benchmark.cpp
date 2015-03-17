#include <stdio.h>

extern void pipe();
extern void stdio();
extern void memmap();

int main(int argc, const char* argv[]) {
  printf("%d:%s\n", argc, argv[0]);
  //pipe();
  //stdio();
  memmap();
  return 0;
}
