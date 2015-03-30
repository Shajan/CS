#define _POSIX_SOURCE
#define _XOPEN_SOURCE
#define _XOPEN_SOURCE_EXTENDED 1
#include <unistd.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <semaphore.h>
//#include <time.h>
#include <errno.h>
#include "common.h"

#define SEM_T(mutex) ((sem_t*) mutex)

/*
 * Implement mutex using semaphore for cross process synchronization.
 *
 * http://man7.org/linux/man-pages/man3/sem_open.3.html
 */
mutex create_mutex(const char* name, bool locked) {
  if (sem_unlink(name) == -1)
    sys_warn("sem_unlink");
  mutex m = (mutex) sem_open(name, O_CREAT|O_EXCL, S_IRUSR|S_IWUSR, (locked ? 0 : 1));
  if (m == SEM_FAILED)
    sys_error_exit("create sem_open");
  return m;
}

void destroy_mutex(const char* name, mutex m) {
  if (sem_close(SEM_T(m)) == -1)
    sys_warn("sem_close");
  if (sem_unlink(name) == -1)
    sys_warn("sem_unlink");
}

mutex open_mutex(const char* name, bool create) {
  mutex m = (mutex) sem_open(name, 0);
  if (m == SEM_FAILED)
    sys_error_exit("open sem_open");
  return m;
}

void close_mutex(mutex m) {
  if (sem_close(SEM_T(m)) == -1)
    sys_warn("sem_close");
}

void lock_mutex(mutex m) {
  if (sem_wait(SEM_T(m)) == -1)
    sys_error_exit("lock sem_wait");
}

bool trylock_mutex(mutex m) {
  int ret = sem_trywait(SEM_T(m));
  if (ret == -1 && errno != EAGAIN)
    sys_error_exit("trylock sem_trywait");
  return (ret == 0);
}

/*
 * http://pubs.opengroup.org/stage7tc1/functions/sem_timedwait.html
 */
bool lock_withtimeout_mutex(mutex m, int seconds) {
  struct timespec ts;
#ifndef CLOCK_REALTIME 
  timeval tv;
  if (gettimeofday(&tv, NULL) == -1)
    sys_error_exit("lock with timeout gettimeofday");
  ts.tv_sec = tv.tv_sec;
  ts.tv_nsec = tv.tv_usec * 1000;
#else
  if (clock_gettime(CLOCK_REALTIME, &ts) == -1)
    sys_error_exit("lock with timeout clock_gettime");
#endif
  ts.tv_sec += seconds;
  int ret;
  while (((ret = sem_timedwait(SEM_T(m), &ts)) == -1) && (errno == EINTR))
    continue; // Restart if interrupted
  if (ret == -1 && errno != ETIMEDOUT)
    sys_error_exit("lock with timeout sem_timedwait");
  return (ret == 0);
}

void unlock_mutex(mutex m) {
  if (sem_post(SEM_T(m)) == -1)
    sys_error_exit("unlock sem_post");
}
