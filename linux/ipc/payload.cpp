#include <stdlib.h>
#include <string.h>
#include "common.h"

#define CANARY 0x1BADF00D

void* g_payload = 0;

struct envelope {
  int canary;
  int size;
  char data[0];
};

void init_payload(int size) {
  void* payload = malloc(size + sizeof(envelope));
  envelope* p = (envelope*) payload;
  p->canary = CANARY;
  p->size = size;
  g_payload = payload;
}

void* get_payload() {
  return g_payload;
}

int verify_payload(const void* payload) {
  if (payload == g_payload) return 1;
  envelope* reference = (envelope* ) g_payload;
  envelope* p = (envelope*) payload;
  if (p->canary != reference->canary) {
    log_error("Canary : expected %d, found %d", reference->canary, p->canary);
    return 0;
  }
  if (p->size != reference->size) {
    log_error("Payload size : expected %d, found %d", reference->size, p->size);
    return 0;
  }

  return (memcmp(p->data, reference->data, p->size) == 0 ? 1 : 0);
}

