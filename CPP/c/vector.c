#include <stdlib.h>
#include <string.h>

#include "vector.h"

#define INITIAL_SIZE 10

int int_compare(void* p_a, void* p_b) {
  int a = *(int*) p_a;
  int b = *(int*) p_b;
  return a - b;
}

int vector_len(const Vector *v) { return v->len; }
int vector_capacity(const Vector *v) { return v->capacity; }

// Globals
TypeInfo intType = { sizeof(int), int_compare };

Vector* create_int_vector() {
  return create_int_vector_ex(INITIAL_SIZE);
}

Vector* create_int_vector_ex(int initial_capacity) {
  return create_vector(initial_capacity, intType);
}

Vector* create_vector(int initial_capacity, TypeInfo type) {
  Vector *v = malloc(sizeof(Vector));

  int size = type.size * initial_capacity;
  void *p_buffer = malloc(size);
  memset(p_buffer, 0, size);

  v->capacity = initial_capacity;
  v->len = 0;
  v->type = type;
  v->p_buffer = p_buffer;

  return v;
}

int vector_get_int(const Vector *v, int idx) {
  int * p = (int *) v->p_buffer;
  return p[idx];
}

void* vector_get(const Vector *v, int idx) {
  return v->p_buffer + idx * v->type.size;
}

void vector_resize(Vector *v, int capacity) {
  v->p_buffer = realloc(v->p_buffer, capacity * v->type.size);
  v->capacity = capacity;

  if (v->len > v->capacity)
    v->len = v->capacity;
}

void _ensure_capacity(Vector *v, int capacity) {
  if (capacity >= v->capacity) {
    int new_size = v->capacity * 2;
    if (new_size < capacity)
      new_size = capacity;
    vector_resize(v, new_size);
  }
}

void vector_append_int(Vector *v, int a) {
  _ensure_capacity(v, v->len + 1);
  int* p = (int*) v->p_buffer;
  p[v->len++] = a;
}

void vector_append(Vector *v, void* p) {
  _ensure_capacity(v, v->len + 1);

  memcpy(v->p_buffer + (v->len * v->type.size), p, v->type.size);
  ++v->len;
}

void vector_free(Vector *v) {
  free(v->p_buffer);
  free(v);
}
