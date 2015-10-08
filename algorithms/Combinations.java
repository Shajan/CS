import java.util.HashMap;

class Combinations {
  public static void main(String[] args) {
    System.out.println("Catalan Number 1 to 25");
    for (int i=1; i<=25; ++i)
      System.out.println(CatalanNumber(i));
  }

  private static HashMap<Integer, Long> catalanNum = new HashMap();

  public static long CatalanNumber(int n) {
    if (n<=1)
      return 1;
    if (catalanNum.get(n) != null)
      return catalanNum.get(n);

    long sum = 0;
    for (int i=1; i<n; ++i) {
      long a = CatalanNumber(i);
      long b = CatalanNumber(n-i);
      sum += a*b;
    }
    catalanNum.put(n, sum);
    return sum;
  }
}
