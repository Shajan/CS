#include <stdlib.h>
#include <string.h>
#include "common.h"

#define CANARY 0x1BADF00D

void* g_payload = NULL;

struct envelope {
  int canary;
  int size;
  char data[0];
};

void set_data(void* p, int size) {
  char* pc = (char *) p;
  for (int i=0; i<size; ++i)
    pc[i] = i % 0xFF;
}

void free_payload() {
  if (g_payload != NULL) {
    void *payload = g_payload;
    g_payload = NULL;
    free(payload);
  }
}

void init_payload(int size) {
  free_payload();
  void* payload = malloc(size + sizeof(envelope));
  envelope* p = (envelope*) payload;
  p->canary = CANARY;
  p->size = size;
  set_data(p->data, size);
  g_payload = payload;
}

void* get_payload() {
  return g_payload;
}

int payload_size() {
  envelope* e = (envelope* ) g_payload;
  return e->size + sizeof(envelope);
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
