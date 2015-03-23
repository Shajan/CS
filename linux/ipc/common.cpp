#include <stdio.h>
#include <stdarg.h>

void logError(const char* fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(stderr, fmt, argptr);
  va_end(argptr);
}

void log(const char* fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);
  vfprintf(stdout, fmt, argptr);
  va_end(argptr);
}
