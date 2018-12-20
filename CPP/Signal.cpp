#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <signal.h>
#include <unistd.h>

#define SIGTERM_MSG "SIGTERM received.\n"

void sig_term_handler(int signum, siginfo_t *info, void *ptr) {
  write(STDERR_FILENO, SIGTERM_MSG, sizeof(SIGTERM_MSG));
  sleep(30000);
}

void catch_sigterm() {
  static struct sigaction _sigact;

  memset(&_sigact, 0, sizeof(_sigact));
  _sigact.sa_sigaction = sig_term_handler;
  _sigact.sa_flags = SA_SIGINFO;

  sigaction(SIGTERM, &_sigact, NULL);
}

int main() {
  catch_sigterm();
  printf("I am sleeping...\n");
  sleep(3000);
  return 0;
}
