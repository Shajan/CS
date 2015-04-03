#include "common.h"

int main(int argc, const char* argv[]) {
  init_payload(10);
  //stdio();
  //pipe();
  //mutex_test();
  //memmap();
  ipc();
  free_payload();
  return 0;
}
