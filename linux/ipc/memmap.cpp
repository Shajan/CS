#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include "common.h"

static void test();

void memmap() {
  log("memmap start\n");
  test();
  log("memmap end\n");
}

/*
 * References
 * http://man7.org/linux/man-pages/man2/mmap.2.html
 * https://groups.google.com/forum/#!topic/comp.os.linux.development.system/BZoz6GBc7Vw
 *   msync(MS_INVALIDATE) will essentially throw out the page tables for that process: including the dirty state. 
 *   kernel might have written the pages out earlier in an attempt to reclaim some memory, but if you have enough
 *   memory then it should be a fairly good way of getting rid of some unnecessary disk IO.
 *   Essentially, you _should_ be able to just do the MS_INVALIDATE just
 *   before unmapping, and then you shouldn't see the flurry of disk activity of syncing the area to disk.
 */
static void test() {
  int fd[2];
  pid_t childpid;
  char string[] = "Hello, World!\n";
  char readbuffer[80];

  pipe(fd);
  //mmap(0, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  mmap(0, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
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
