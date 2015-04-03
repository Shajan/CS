#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include "common.h"

#define ERRORSTREAM stderr
#define LOGSTREAM stdout

void log_error(const char* fmt, ...) {
  fprintf(ERRORSTREAM, "Error ");
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(ERRORSTREAM, fmt, argptr);
  va_end(argptr);
  print_stack();
}

void error_exit(const char* fmt, ...) {
  fprintf(ERRORSTREAM, "Error exit ");
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(ERRORSTREAM, fmt, argptr);
  va_end(argptr);
  print_stack();
  exit(1);
}

void sys_error_exit(const char* fmt, ...) {
  fprintf(ERRORSTREAM, "Error exit [%s] ", strerror(errno));
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(ERRORSTREAM, fmt, argptr);
  va_end(argptr);
  print_stack();
  exit(1);
}

void sys_warn(const char* fmt, ...) {
  fprintf(ERRORSTREAM, "Warning [%s] ", strerror(errno));
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(ERRORSTREAM, fmt, argptr);
  va_end(argptr);
  fprintf(ERRORSTREAM, "\n");
}

void log(const char* fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(LOGSTREAM, fmt, argptr);
  va_end(argptr);
  fprintf(LOGSTREAM, "\n");
}

