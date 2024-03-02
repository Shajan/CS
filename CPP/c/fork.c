#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: childcreates <iterations>\n");
        exit(1);
    }

    printf("---- %d ----\n", getpid());
    int iterations = strtol(argv[1], NULL, 10);

    for (int i = 0; i < iterations; i++) {
        printf("[i:%d] %d", i, getpid());
        if (i < iterations - 1)
            printf(" -> ");
        else
            printf("\n");

        if (i < iterations - 1) {
            fflush(stdout);
            int n = fork();
            if (n < 0) {
                perror("fork");
                exit(1);
            }
            if (n != 0) {
                // Parent process
                printf(" P(i:%d, n:%d, pid:%d) ", i, n, getpid());
                break;
            }
            printf(" C(i:%d, n:%d, pid:%d) ", i, n, getpid());
        }
    }

    return 0;
}

