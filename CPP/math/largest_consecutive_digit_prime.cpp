// Copy this code to http://cpp.sh then click 'Run'

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

typedef unsigned long int Num;
vector<Num> g_primes;

// Get rid of the first decimal
Num decimalShiftLeft(Num n) {
    char str[10];
    sprintf(str, "%lu", n);
    if (strlen(str) == 1)
      return 0;
    return atoi(&str[1]);
}

bool is_prime(Num n) {
    for (Num prime : g_primes) {
        if ((n % prime) == 0)
          return false;
    }
    return true;
}

// Build a list of primes, which will come in handy to check for
// prime
void init_primes() {
    // Optimization: No need to search all the way to 123456789,
    // It can be shown that the biggest possible prime with consecutive digits
    // is much smaller - example
    //   123456789 is not prime
    //   anything ending with 8 is not prime
    //   ... many more individual cases to bring down the search space
    // Keeping this code simple by not optimizing    //Num max = sqrt(123456789);
    Num max = sqrt(123456789);
    g_primes.push_back(2);
    for (Num i=3; i<max; ++i) {
        if (is_prime(i))
          g_primes.push_back(i);
    }
}

Num largest_consecutive_digit_prime(Num n) {
    if (n < 2)
      return 0;

    if (is_prime(n))
      return n;
    
    Num a = largest_consecutive_digit_prime(decimalShiftLeft(n));
    Num b = largest_consecutive_digit_prime(n/10);
    return max(a, b);
}

int main() {
  init_primes();  
  printf("Largest : %lu\n", largest_consecutive_digit_prime(123456789));
  return 0;
}
