void log_error(const char* fmt, ...);
void log(const char* fmt, ...);
void error_exit(const char* fmt, ...);
void sys_error_exit(const char* fmt, ...);
void sys_warn(const char* fmt, ...);
void print_stack();

void init_payload(int size);
void free_payload();
void* get_payload();
int payload_size();
int verify_payload(const void*);

// mutex definitions
typedef void* mutex;
mutex create_mutex(const char* name, bool locked);
void destroy_mutex(const char* name, mutex m);
mutex open_mutex(const char* name);
void close_mutex(mutex m);
void lock_mutex(mutex m);
bool trylock_mutex(mutex m);
bool lock_withtimeout_mutex(mutex m, int seconds);
void unlock_mutex(mutex m);

// shared memory
typedef void* map;
map get_map(const char* name, int size, bool create);
void unmap(map m, int size);
#define get_ptr(map) (void*)map

void pipe();
void stdio();
void mutex_test();
void memmap();
void ipc();
