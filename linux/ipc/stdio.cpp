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
  char string[] = "Hello, World!\n";
  char readbuffer[80];

  pipe(fd);
  if ((childpid = fork()) == -1) {
    perror("fork");
    exit(1);
  }

  if (childpid == 0) {
    /* Child process writes */
    dup2(fd[0], STDOUT_FILENO);
    close(fd[0]);
    //log("Writing string: %s", string);
    write(STDOUT, string, strlen(string) + 1);
  } else {
    /* Parent process reads */
    dup2(fd[1], STDIN_FILENO);
    close(fd[1]);
    int nbytes = read(STDIN, readbuffer, sizeof(readbuffer));
    log("Received %d bytes: %s", nbytes, readbuffer);
  }
}
