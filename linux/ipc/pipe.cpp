#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include "common.h"

static void test();

void pipe() {
  log("pipe start\n");
  test();
  log("pipe end\n");
}

static void test() {
  int fd[2];
  pid_t childpid;
  char string[] = "Hello, World!\n";
  char readbuffer[80];

  pipe(fd);
  if ((childpid = fork()) == -1) {
    perror("fork");
    exit(1);
  }

  if (childpid == 0) {
    /* Child process closes up input side of pipe */
    close(fd[0]);
    log("Writing string: %s", string);
    write(fd[1], string, strlen(string) + 1);
  } else {
    /* Parent process closes up output side of pipe */
    close(fd[1]);
    int nbytes = read(fd[0], readbuffer, sizeof(readbuffer));
    log("Received %d bytes: %s", nbytes, readbuffer);
  }
}
