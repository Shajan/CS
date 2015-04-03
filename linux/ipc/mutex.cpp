#define _POSIX_SOURCE
#define _XOPEN_SOURCE
#define _XOPEN_SOURCE_EXTENDED 1
#include <unistd.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <semaphore.h>
#include <time.h>
#include <errno.h>
#include <stdlib.h>
#include "common.h"

static void test();

void mutex_test() {
  log("mutex start");
  test();
  log("mutex end");
}

#define MUTEX_NAME "/sdasan"

static void test() {
  pid_t childpid;
  mutex mparent = create_mutex(MUTEX_NAME, true);

  if ((childpid = fork()) == -1) {
    sys_error_exit("fork");
  }

  if (childpid == 0) {
    /* Child process */
    mutex m = open_mutex(MUTEX_NAME);
    log("Child try lock mutex");
    if (trylock_mutex(m))  // Should not succeed
      error_exit("Child trylock_mutex: mutex not locked by parent");
    log("Child try to lock, 1 second timeout");
    if (lock_withtimeout_mutex(m, 1))
      error_exit("Child lock_withtimeout_mutex: mutex not locked by parent");
    log("Child try to lock, 4 second timeout");
    if (!lock_withtimeout_mutex(m, 4))
      error_exit("Child lock_withtimeout_mutex: mutex locked by parent");
    log("Child release lock");
    unlock_mutex(m);
    close_mutex(m);
    exit(0);
  } else {
    /* Parent process */
    log("Parent sleep 2 second with lock");
    sleep(2);
    log("Parent unlock");
    unlock_mutex(mparent);
    log("Parent sleep 2 seconds without lock");
    sleep(2);
    log("Parent wait for lock");
    lock_mutex(mparent);
    log("Parent got lock");
    destroy_mutex(MUTEX_NAME, mparent);
  }
}

#define SEM_T(mutex) ((sem_t*) mutex)

/*
 * Implement mutex using semaphore for cross process synchronization.
 *
 * http://man7.org/linux/man-pages/man3/sem_open.3.html
 */
mutex create_mutex(const char* name, bool locked) {
  if (sem_unlink(name) == -1)
    sys_warn("create sem_unlink [%s]", name);
  mutex m = (mutex) sem_open(name, O_CREAT|O_EXCL, S_IRUSR|S_IWUSR, (locked ? 0 : 1));
  if (m == SEM_FAILED)
    sys_error_exit("create sem_open [%s]", name);
  return m;
}

void destroy_mutex(const char* name, mutex m) {
  if (sem_close(SEM_T(m)) == -1)
    sys_warn("destroy sem_close [%s]", name);
  if (sem_unlink(name) == -1)
    sys_warn("destroy sem_unlink [%s]", name);
}

mutex open_mutex(const char* name) {
  mutex m = (mutex) sem_open(name, 0);
  if (m == SEM_FAILED)
    sys_error_exit("open sem_open [%s]", name);
  return m;
}

void close_mutex(mutex m) {
  if (sem_close(SEM_T(m)) == -1)
    sys_warn("close sem_close");
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

void get_time(timespec* p_ts) {
#ifndef CLOCK_REALTIME
  timeval tv;
  if (gettimeofday(&tv, NULL) == -1)
    sys_error_exit("lock with timeout gettimeofday");
  p_ts->tv_sec = tv.tv_sec;
  p_ts->tv_nsec = tv.tv_usec * 1000;
#else
  if (clock_gettime(CLOCK_REALTIME, p_ts) == -1)
    sys_error_exit("lock with timeout clock_gettime");
#endif
}

/*
 * http://pubs.opengroup.org/stage7tc1/functions/sem_timedwait.html
 */
bool lock_withtimeout_mutex(mutex m, int seconds) {
  int ret = 0;
  struct timespec wait_end;
  get_time(&wait_end);
  wait_end.tv_sec += seconds;
#ifdef __APPLE__
  // Naive implemenatation for mac
  struct timespec current_time;
  do {
    if (trylock_mutex(m))
      return 0;
    sleep(1); // Sleep 1 sec, not a good hack
    get_time(&current_time);
  } while (current_time.tv_sec >= wait_end.tv_sec);
  return 1;
#else
  while (((ret = sem_timedwait(SEM_T(m), &wait_end)) == -1) && (errno == EINTR))
    continue; // Restart if interrupted
  if (ret == -1 && errno != ETIMEDOUT)
    sys_error_exit("lock with timeout sem_timedwait");
  return (ret == 0);
#endif
}

void unlock_mutex(mutex m) {
  if (sem_post(SEM_T(m)) == -1)
    sys_error_exit("unlock sem_post");
}
