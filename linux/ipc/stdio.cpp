#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include "common.h"

#define STDIN 0
#define STDOUT 1

static void test();

void stdio() {
  log("stdio start\n");
  test();
  log("stdio end\n");
}

static void test() {
  int fd[2];
  pid_t childpid;
  char string[] = "Hello, World!";
  char readbuffer[80];

  pipe(fd);
  if ((childpid = fork()) == -1) {
    error_exit("fork");
  }

  if (childpid == 0) {
    /* Child process writes */
    close(fd[0]);
    if (dup2(fd[1], STDOUT_FILENO) == -1) {
      error_exit("Child dup2");
    }
    log("Writing string: %s\n", string);
    int nbytes = write(STDOUT_FILENO, string, strlen(string) + 1); 
    if (nbytes == -1) {
      error_exit("Child write");
    } else {
      log("Wrote %d bytes\n", nbytes);
    }
    exit(0);
  } else {
    /* Parent process reads */
    close(fd[1]);
    if (dup2(fd[0], STDIN_FILENO) == -1) {
      error_exit("Parent dup2");
    }
    int nbytes = read(STDIN_FILENO, readbuffer, sizeof(readbuffer));
    if (nbytes <= 0) {
      log_error("Error, recived %d bytes\n", nbytes);
      readbuffer[0] = 0;
    }
    log("Received %d bytes: %s\n", nbytes, readbuffer);
  }
}
