#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include "common.h"

static void test();

void stdio() {
  log("stdio start");
  test();
  log("stdio end");
}

static void test() {
  int fd[2];
  pid_t childpid;

  pipe(fd);
  if ((childpid = fork()) == -1) {
    sys_error_exit("fork");
  }

  if (childpid == 0) {
    /* Child process writes */
    close(fd[0]);
    if (dup2(fd[1], STDOUT_FILENO) == -1) {
      sys_error_exit("Child dup2");
    }
    int nbytes = write(STDOUT_FILENO, get_payload(), payload_size());
    if (nbytes == -1) {
      sys_error_exit("Child write");
    } else {
      log("Wrote %d bytes", nbytes);
    }
    exit(0);
  } else {
    /* Parent process reads */
    close(fd[1]);
    if (dup2(fd[0], STDIN_FILENO) == -1) {
      sys_error_exit("Parent dup2");
    }
    void* buffer = malloc(payload_size());
    int nbytes = read(STDIN_FILENO, buffer, payload_size());
    if (nbytes != payload_size()) {
      error_exit("Error, recived %d bytes, expected %d bytes", nbytes, payload_size());
    }
    if (!verify_payload(buffer)) {
      error_exit("Payload corrupt");
    }
    free(buffer);
    log("Received %d bytes", nbytes);
  }
}
