import java.util.*;

class Prime {
  private static List<Long> primes;

  public static void main(String[] args) {
    generatePrimes(100);

    print("Primes :", primes);
    print("Binary Sum :", binarySum(primes));
    print("Binary Diff :", binaryDiff(primes));
  }

  private static void print(String s, List<Long> list) {
    System.out.print(s);
    for (long l : list) {
      System.out.print(l);
      System.out.print(' ');
    }
    System.out.println();
  }

  private static boolean isPrime(long l) {
    for (long prime : primes) {
      if ((l % prime) == 0)
        return false;
    }
    return true;
  }

  private static void generatePrimes(long max) {
    primes = new ArrayList<Long>();

    for (long l=2; l<max; ++l) {
      if (isPrime(l))
        primes.add(l);
    }
  }

  private static List<Long> binarySum(List<Long> input) {
    List<Long> output = new ArrayList<Long>();

    long a = input.get(0);
    for (int i=1; i<input.size(); ++i) {
      long b = input.get(i);
      output.add(a + b);
      a = b;
    }
    return output;
  }

  private static List<Long> binaryDiff(List<Long> input) {
    List<Long> output = new ArrayList<Long>();

    long a = input.get(0);
    for (int i=1; i<input.size(); ++i) {
      long b = input.get(i);
      output.add(b - a);
      a = b;
    }
    return output;
  }
}
