#define _POSIX_SOURCE
#define _XOPEN_SOURCE
#define _XOPEN_SOURCE_EXTENDED 1

#include <unistd.h>
#include <stdio.h>
#include <time.h>

/*
 * References
 *  http://linux.die.net/include/unistd.h
  * http://pubs.opengroup.org/onlinepubs/009695399/basedefs/unistd.h.html
 *  http://www.opengroup.org/infosrv/unsupported/stdtools/ckstdvers/ckstdvers.c
 */

int main(int argc, char *argv[]) {
  int posix=0, xopen_unix=0, xcu=0, xsh=0;
  long user_sc_value=0;

#ifdef _POSIX_VERSION
  switch (_POSIX_VERSION) {
  case 199009L: /* classic dot1 - ISO version */
    printf("_POSIX_VERSION=%ldL (ISO 9945-1:1990[IEEE Std POSIX.1-1990])\n", _POSIX_VERSION);
    posix = 90;
    break;
  case 198808L:  /* classic dot 1 - non ISO version */
    printf("_POSIX_VERSION=%ldL (IEEE Std POSIX.1-1988)\n", _POSIX_VERSION);
    posix = 88;
    break;
  case 199309L: /* POSIX realtime */
    printf("_POSIX_VERSION=%ldL (IEEE Std POSIX.1b-1993)\n", _POSIX_VERSION);
    posix = 93;
    break;
  case 199506L:  /* POSIX threads */
    printf("_POSIX_VERSION=%ldL (ISO 9945-1:1996 [IEEE Std POSIX.1-1996])\n", _POSIX_VERSION);
    posix = 95;
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

  if (posix >= 95) {
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

#ifdef _XOPEN_VERSION
  switch (_XOPEN_VERSION) {
  case 3:
    printf("_XOPEN_VERSION=%d (XPG3 System Interfaces)\n", _XOPEN_VERSION);
    xsh = 3;
    break;
  case 4:
    printf("_XOPEN_VERSION=%d (XPG4 System Interfaces)\n", _XOPEN_VERSION);
    xsh = 4;
    break;
  case 500:
    printf("_XOPEN_VERSION=%d (XSH5 System Interfaces)\n", _XOPEN_VERSION);
    xsh = 5;
    break;
  default:
    printf("Unknown _XOPEN_VERSION=%d\n", _XOPEN_VERSION);
    break;
  }

#ifdef _POSIX2_C_VERSION
  if (_POSIX2_C_VERSION == 199209L)
    printf("_POSIX2_C_VERSION=199209L : ISO POSIX.2 C Languages Binding is supported\n");
  else
    printf("_POSIX2_C_VERSION!=199209L :[%d] ISO POSIX.2 C Language Binding not supported\n", _POSIX2_C_VERSION);
#else
  printf("_POSIX2_C_VERSION not defined: ISO POSIX.2 C Language Binding  not supported\n");
#endif
#else
  printf("_XOPEN_VERSION not defined\n");
#endif


#ifdef _XOPEN_XCU_VERSION
  switch (_XOPEN_XCU_VERSION) {
  case 3:
    printf("_XOPEN_XCU_VERSION=%d (Historical Commands)\n", _XOPEN_XCU_VERSION);
    xcu = 3;
    break;
  case 4:
    printf("_XOPEN_XCU_VERSION=%d (POSIX.2 Commands)\n", _XOPEN_XCU_VERSION);
    xcu = 4;
    break;
  case 5:
    printf("_XOPEN_XCU_VERSION=%d (POSIX.2 Commands)\n", _XOPEN_XCU_VERSION);
    xcu = 5;
    break;
  default:
    printf("Unknown _XOPEN_XCU_VERSION=%d\n", _XOPEN_XCU_VERSION);
    break;
  }
#else
  printf("_XOPEN_XCU_VERSION not defined\n");
#endif

#ifdef _POSIX2_VERSION
  if (_POSIX2_VERSION == 199209L)
    printf("_POSIX2_VERSION=199209L : ISO POSIX.2 is supported\n");
  else
    printf("_POSIX2_VERSION != 199209L : ISO POSIX.2 not supported\n");
#else
  printf("_POSIX2_VERSION not defined: ISO POSIX.2 not supported\n");
#endif

#ifdef _XOPEN_UNIX
  printf("_XOPEN_UNIX support is claimed\n");
  xopen_unix = 1;
#else
  printf("_XOPEN_UNIX is not supported\n");
#endif

/* check valid combinations */

#if (defined(_POSIX_SOURCE) && defined(_XOPEN_SOURCE) && defined(_XOPEN_XCU_VERSION) && defined(_XOPEN_VERSION))
  if (xopen_unix == 1)  {
    if (xcu != 4 && xcu != 5) 
      printf("Invalid value found for _XOPEN_XCU_VERSION (%d) when _XOPEN_UNIX is supported\n", _XOPEN_XCU_VERSION);
    if (xsh != 4  && xsh != 500 )
      printf("Invalid value found for _XOPEN_VERSION (%d) when _XOPEN_UNIX is supported\n", _XOPEN_VERSION);
    if (posix < 90)
      printf("Invalid value found for _POSIX_VERSION (%ld) when _XOPEN_UNIX is supported\n", _POSIX_VERSION);
  }

  if (xsh == 4) {
    if (posix < 90)
      printf("Invalid value found for _POSIX_VERSION (%ld) when _XOPEN_VERSION is set to 4\n", _POSIX_VERSION);
   if ((xcu != 3) && (xcu != 4))
      printf("Invalid value found for _XOPEN_XCU_VERSION (%d) when _XOPEN_VERSION is set to 4\n", _XOPEN_XCU_VERSION);
  }
#endif

#ifdef _POSIX_TIMERS
  printf("_POSIX_TIMERS defined\n");
#else
  printf("_POSIX_TIMERS not defined\n");
#endif

#ifdef CLOCK_REALTIME 
  printf("CLOCK_REALTIME defined\n");
#else
  printf("CLOCK_REALTIME not defined\n");
#endif

  return 0;
}
