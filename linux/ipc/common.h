void logError(const char* fmt, ...);
void log(const char* fmt, ...);

void init_payload(int size);
void* get_payload();
int verify_payload(const void*);

void pipe();
void stdio();
void memmap();

