#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int max_val(int a, int b) { return ((a) > (b) ? (a) : (b)); }

int LongsetSubString(const char* A, const char* B);

int DynamicProgramming(int argc, char* argv[]) {
  if (argc == 3) {
      printf("(%s, %s) : %d\n", argv[1], argv[2], LongsetSubString(argv[1], argv[2]));
      getchar();
      return 0;
  }
  return 1;
}

void log(int* a, int M, int N) {
    for (int i=0; i<M; ++i) {
        for (int j=0; j<N; ++j) {
            printf("%d ", a[i,j]);
        }
        printf("\n");
    }
}

// Wrong! it accumulates, does not start over.
int LongsetSubString(const char* A, const char* B) {
    int M = strlen(A) + 1;
    int N = strlen(B) + 1;

    int* dp = new int[M,N]; // 2 D Array, with first row/col 0

    for (int i=0; i<M; ++i) {
        dp[i,0] = 0; // First row
    }
    for (int j=0; j<N; ++j) {
        dp[0,j] = 0; // First col
    }

    for (int i=1; i<M; ++i) {
        for (int j=1; j<N; ++j) {
            dp[i,j] = max_val(dp[i-1,j], dp[i,j-1]);
            if ((A[i-1] == B[j-1]) && (A[i-1] != 0))
                dp[i,j] = max_val(dp[i,j], dp[i-1,j-1] + 1);
        }
    }

    log(dp, M, N);
    int ret = dp[M-1,N-1];
    delete [] dp;
    return ret;
}
