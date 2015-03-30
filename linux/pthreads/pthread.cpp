#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define NUM_THREADS     5

/*
 * Reference https://computing.llnl.gov/tutorials/pthreads/
 */
void* thread_start(void* data) {
   long id = (long)data;
   printf("Inside thread #%ld!\n", id);
   pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
   pthread_t threads[NUM_THREADS];
   for (long t=0; t<NUM_THREADS; t++) {
      printf("Creating thread %ld\n", t);
      int rc = pthread_create(&threads[t], NULL, thread_start, (void *)t);
      if (rc) {
         printf("Error pthread_create %d, [%s]\n", rc, strerror(errno));
         exit(-1);
      }
   }
   pthread_exit(NULL);
}
