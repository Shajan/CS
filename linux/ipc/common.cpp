#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>

#define ERRORSTREAM stderr
//#define LOGSTREAM stdout
#define LOGSTREAM stderr

void log_error(const char* fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(ERRORSTREAM, fmt, argptr);
  va_end(argptr);
}

void error_exit(const char* msg) {
  log_error("%s : %s\n", msg, strerror(errno));
  exit(1);
}

void log(const char* fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(LOGSTREAM, fmt, argptr);
  va_end(argptr);
}