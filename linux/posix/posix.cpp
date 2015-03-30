#define _POSIX_SOURCE
/* Note for that UNIX 98 that _XOPEN_SOURCE should be set to 500 */

#define _XOPEN_SOURCE
#define _XOPEN_SOURCE_EXTENDED 1

#include <unistd.h>
#include <stdio.h>

/*
 * References
 *  http://linux.die.net/include/unistd.h
 *  http://www.opengroup.org/infosrv/unsupported/stdtools/ckstdvers/ckstdvers.c
 */

int main(int argc, char *argv[]) {
  int posix_version = 0;
  long user_sc_value = 0;

#ifdef _POSIX_VERSION
  switch (_POSIX_VERSION) {
  case 199009L: /* classic dot1 - ISO version */
    printf("_POSIX_VERSION=%ldL (ISO 9945-1:1990[IEEE Std POSIX.1-1990])\n", _POSIX_VERSION);
    posix_version = 90;
    break;
  case 198808L:  /* classic dot 1 - non ISO version */
    printf("_POSIX_VERSION=%ldL (IEEE Std POSIX.1-1988)\n", _POSIX_VERSION);
    posix_version = 88;
    break;
  case 199309L: /* POSIX realtime */
    printf("_POSIX_VERSION=%ldL (IEEE Std POSIX.1b-1993)\n", _POSIX_VERSION);
    posix_version = 93;
    break;
  case 199506L:  /* POSIX threads */
    printf("_POSIX_VERSION=%ldL (ISO 9945-1:1996 [IEEE Std POSIX.1-1996])\n", _POSIX_VERSION);
    posix_version = 95;
    break;
  default:
    printf("Unknown _POSIX_VERSION=%ldL\n", _POSIX_VERSION);
    break;
  }
  /* check consistency with sysconf */
  user_sc_value = sysconf(_SC_VERSION);
  if (user_sc_value != _POSIX_VERSION )
    printf("Warning: sysconf(_SC_VERSION) returned %ldL, expected %ldL\n", user_sc_value, _POSIX_VERSION);
#else
  printf("_POSIX_VERSION not defined\n");
#endif

#ifdef _POSIX_JOB_CONTROL
  printf("\t_POSIX_JOB_CONTROL supported\n");
#else
  printf("\t_POSIX_JOB_CONTROL not supported\n");
#endif

#ifdef _POSIX_SAVED_IDS
  printf("\t_POSIX_SAVED_IDS supported\n");
#else
  printf("\t_POSIX_SAVED_IDS not supported\n");
#endif

  if (posix_version >= 95 ) {
#ifdef _POSIX_THREADS
    printf("\t_POSIX_THREADS supported\n");
#else
    printf("\t_POSIX_THREADS not supported\n");
#endif
#ifdef _POSIX_THREAD_SAFE_FUNCTIONS
    printf("\t_POSIX_THREAD_SAFE_FUNCTIONS supported\n");
#else
    printf("\t_POSIX_THREAD_SAFE_FUNCTIONS not supported\n");
#endif
#ifdef _POSIX_THREAD_ATTR_STACKADDR
    printf("\t_POSIX_THREAD_ATTR_STACKADDR supported\n");
#else
    printf("\t_POSIX_THREAD_ATTR_STACKADDR not supported\n");
#endif
#ifdef _POSIX_THREAD_ATTR_STACKSIZE
    printf("\t_POSIX_THREAD_ATTR_STACKSIZE supported\n");
#else
    printf("\t_POSIX_THREAD_ATTR_STACKSIZE not supported\n");
#endif
  }
}
