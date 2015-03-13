#include <stdio.h>

extern void pipe();
extern void stdio();

int main(int argc, const char* argv[]) {
  printf("%d:%s\n", argc, argv[0]);
  //pipe();
  stdio();
  return 0;
}
