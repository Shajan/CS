import java.util.*;

class Prime {
  private static List<Long> primes;
  public static void main(String[] args) {
    primes = new ArrayList<Long>();
    primes.add(2l);

    for (long l=3; l<100; ++l) {
      if (isPrime(l))
        primes.add(l);
    }

    for (long prime : primes) {
      System.out.print(prime);
      System.out.print(",");
    }
    System.out.println();

    computePrimes(2, 3, 1);
    computePrimes(3, 5, 2);
  }

  private static boolean isPrime(long l) {
    for (long prime : primes) {
      if ((l % prime) == 0)
        return false;
    }
    return true;
  }

  private static void computePrimes(long a, long b, long c) {
    System.out.println(String.format("%d, %d, %d ==> %d, %d", a, b, c, (a * b - c), (a * b + c)));
  }
}
