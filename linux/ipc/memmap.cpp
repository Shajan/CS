#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "common.h"

//#define FILE_NAME "/dev/shm/sdasan"
//#define FILE_NAME "/dev/zero"
#define FILE_NAME "/tmp/sdasan"
#define NAME "/sdasan"

static void test_fork();
static void test_open_shared_filehandle();
static void test_open();

#define TO_PTR(x) ((void*)x)

void memmap() {
  log("memmap start");
  //test_fork();
  //test_open_shared_filehandle();
  //test_open();
  log("memmap end");
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
    sys_error_exit("open " FILE_NAME);
  }

  if (ftruncate(fd, payload_size()) == -1) {
    sys_error_exit("ftruncate %s", FILE_NAME);
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
    unmap(pmap, payload_size());
    exit(0);
  } else {
    /* Parent process */
    sleep(1); // Idealy this is a signal, but for now just sleep
    if (!verify_payload(pmap)) {
      error_exit("Payload corrupt");
    }
    unmap(pmap, payload_size());
  }
}

/*
 * http://man7.org/linux/man-pages/man3/shm_open.3.html
 */
map get_map(const char* name, int size, bool create) {
  int fd;
  void* pmap;
  int flags = (create ? (O_RDWR|O_CREAT|O_EXCL) : O_RDWR);

  if (create && (shm_unlink(name) == -1)) {
    sys_warn("shm_unlink %s", name);
  }

  if ((fd = shm_open(name, flags, S_IRUSR|S_IWUSR)) == -1) {
    sys_error_exit("shm_open %s", name);
  }

  if (create && (ftruncate(fd, payload_size()) == -1)) {
    sys_error_exit("ftruncate name:%s fd:%d size:%d", name, fd, payload_size());
  }

  if ((pmap = mmap(0, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)) == (void*) -1) {
    sys_error_exit("mmap");
  }

  if (close(fd) == -1) {
    sys_error_exit("close");
  }
  return (map)pmap;
}

void unmap(map m, int size) {
  if (munmap(TO_PTR(m), size) == -1)
    sys_warn("munmap");
}

static void test_open_shared_filehandle() {
  pid_t childpid;
  map m;

  m = get_map(NAME, payload_size(), true);
  if ((childpid = fork()) == -1) {
    sys_error_exit("fork");
  }

  if (childpid == 0) {
    /* Child process */
    memcpy(TO_PTR(m), get_payload(), payload_size());
    unmap(m, payload_size());
    exit(0);
  } else {
    /* Parent process */
    sleep(1); // Idealy this is a signal, but for now just sleep
    if (!verify_payload(TO_PTR(m))) {
      error_exit("Payload corrupt");
    }
    unmap(m, payload_size());
  }
}

// Assume no parent/child process relation - no fork
static void test_open() {
  pid_t childpid;
  int fd;
  map m;

  if ((childpid = fork()) == -1) {
    sys_error_exit("fork");
  }
  if (childpid == 0) {
    /* Child process */
    sleep(1); // Idealy this is a signal, but for now just sleep
    log("Child: opening map");
    m = get_map(NAME, payload_size(), false);
    memcpy(TO_PTR(m), get_payload(), payload_size());
    log("Child: wrote data");
    unmap(m, payload_size());
    exit(0);
  } else {
    /* Parent process */
    m = get_map(NAME, payload_size(), true);
    log("Parent: created map");
    sleep(2); // Idealy this is a signal, but for now just sleep
    log("Parent: verifying data");
    if (!verify_payload(TO_PTR(m))) {
      error_exit("Payload corrupt");
    }
    unmap(m, payload_size());
  }
}
