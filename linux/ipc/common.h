void log_error(const char* fmt, ...);
void log(const char* fmt, ...);
void error_exit(const char* msg);

void init_payload(int size);
void* get_payload();
int payload_size();
int verify_payload(const void*);

void pipe();
void stdio();
void memmap();

