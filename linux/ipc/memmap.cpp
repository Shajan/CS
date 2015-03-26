#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include "common.h"

//#define FILE_NAME "/dev/shm/sdasan"
//#define FILE_NAME "/dev/zero"
#define FILE_NAME "/tmp/sdasan"

static void test_fork();
static void test_open();

void memmap() {
  log("memmap start\n");
  test_fork();
  test_open();
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
 *
 * MAP_ANONYMOUS The mapping is not backed by any file; its contents are
 * initialized to zero.  The fd and offset arguments are ignored;
 * however, some implementations require fd to be -1 if
 * MAP_ANONYMOUS (or MAP_ANON) is specified, and portable
 * applications should ensure this.  The use of MAP_ANONYMOUS in
 * conjunction with MAP_SHARED is supported on Linux only since kernel 2.4.
 *
 * MAP_ANON Synonym for MAP_ANONYMOUS
 *
 * MAP_SHARED Share this mapping. Updates to the mapping are visible to
 * other processes that map this file, and are carried
 * through to the underlying file.  The file may not actually
 * be updated until msync(2) or munmap() is called.
 *
 * MAP_PRIVATE Create a private copy-on-write mapping. Updates to the
 * mapping are not visible to other processes mapping the
 * same file, and are not carried through to the underlying
 * file. It is unspecified whether changes made to the file
 * after the mmap() call are visible in the mapped region.
 *
 */

static void test_fork() {
  int fd;
  pid_t childpid;
  void* pmap;

  if ((fd = open(FILE_NAME, O_RDWR)) == -1) {
    sys_error_exit(FILE_NAME);
  }

  if ((pmap = mmap(0, payload_size(), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)) == (void*) -1) {
    sys_error_exit("mmap");
  }

  if (close(fd) == -1) {
    sys_error_exit("close");
  }

  if ((childpid = fork()) == -1) {
    sys_error_exit("fork");
  }

  if (childpid == 0) {
    /* Child process */
    memcpy(pmap, get_payload(), payload_size());
    exit(0);
  } else {
    /* Parent process */
    sleep(1); // Idealy this is a signal, but for now just sleeep
    if (!verify_payload(pmap)) {
      error_exit("Payload corrupt");
    }
  }
}

static void test_open() {
}
