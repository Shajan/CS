#ifndef __C_VECTOR_H__
#define __C_VECTOR_H__

// Represent a record (or a primitive type)
typedef struct {
    int size;
    int (*fn_compare)(void*, void*); // Compare values
} TypeInfo;

typedef struct {
    int capacity;
    int len;
    TypeInfo type;
    void *p_buffer;
} Vector;

// Declare the global variable
extern TypeInfo intType;

Vector* create_int_vector();
Vector* create_int_vector_ex(int initial_capacity);
Vector* create_vector(int initial_capacity, TypeInfo type);
void vector_resize(Vector *v, int capacity);
void vector_free(Vector *v);

int vector_len(const Vector *v);
int vector_capacity(const Vector *v);

void vector_append_int(Vector *v, int a);
void vector_append(Vector *v, void* p);

int vector_get_int(const Vector *v, int idx);
void* vector_get(const Vector *v, int idx);

#endif
