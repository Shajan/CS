#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>

#define ERRORSTREAM stderr
#define LOGSTREAM stdout

void log_error(const char* fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(ERRORSTREAM, fmt, argptr);
  va_end(argptr);
  fprintf(ERRORSTREAM, "\n");
}

void error_exit(const char* fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(ERRORSTREAM, fmt, argptr);
  va_end(argptr);
  fprintf(ERRORSTREAM, "\n");
  exit(1);
}

void sys_error_exit(const char* fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(ERRORSTREAM, fmt, argptr);
  va_end(argptr);
  fprintf(ERRORSTREAM, " [%s]\n", strerror(errno));
  exit(1);
}

void sys_warn(const char* fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(ERRORSTREAM, fmt, argptr);
  va_end(argptr);
  log_error(" [%s]\n", strerror(errno));
}

void log(const char* fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(LOGSTREAM, fmt, argptr);
  va_end(argptr);
  fprintf(LOGSTREAM, "\n");
}

