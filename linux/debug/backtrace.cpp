#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cxxabi.h>

/*
 * Reference
 *  http://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes
 */
#define MAX_DEPTH 20

void print_call_stack(int frames_to_skip) {
  int frame_count;
  void* frames[MAX_DEPTH];
  char** raw_symbols;

  frame_count = backtrace(frames, MAX_DEPTH);
  raw_symbols = backtrace_symbols(frames, frame_count);

  if (raw_symbols == NULL) {
    fprintf(stderr, "no symbols\n");
    exit(1);
  }
  
  // +1 to skip this frame ('print_call_stack')
  for (int i = frames_to_skip + 1; i < frame_count; i++) {
    // find parantheses and +address offset surrounding mangled name
    char *mangled_name = 0, *offset_begin = 0, *offset_end = 0;
    for (char *p = raw_symbols[i]; *p; ++p) {
      if (*p == '(') {
        mangled_name = p; 
      } else if (*p == '+') {
        offset_begin = p;
      } else if (*p == ')') {
        offset_end = p;
        break;
      }
    }

    // sdasan: This part is library specific.. change it as necessary

    // if the line could be processed, attempt to demangle the symbol
    if (mangled_name && offset_begin && offset_end && mangled_name < offset_begin) {
      *mangled_name++ = '\0';
      int status = 0;
      char* symbol = abi::__cxa_demangle(raw_symbols[i], 0, 0, &status);
      printf("%d : %s\n", status, raw_symbols[i]);
      if (status == 0) {
        printf("%s\n", symbol);
        free(symbol);
      } else {
        printf("%s\n", raw_symbols[i]);
      }
    } else {
      printf("%s\n", raw_symbols[i]);
    }
  }
  free(raw_symbols);
}

void test_frames(int depth) {
  if (depth > 1)
    test_frames(depth - 1);
  else
    print_call_stack(0);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <Depth>\n", argv[0]);
    exit(1);
  }
  test_frames(atoi(argv[1]));
  return 0;
}
