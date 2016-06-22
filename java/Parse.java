import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

class Parse {
  public static void main(String args[]) {
    test(null, 0);
    test("", 0);
    test(",", 0);
    test("a", 1);
    test("a,", 1);
    test("a,b", 2);
    test("a,b,c,d", 4);
  }

  private static void test(String s, int expect) {
    Set<String> set = csvToSet(s);
    System.out.println(String.format("[%s] %s => %s", (set.size() == expect), s, set));
  }

  private static Set<String> csvToSet(String s) {
    if (s == null || s == "")
      return new HashSet<String>();

    String[] sArray = s.split(",");
    return new HashSet(Arrays.asList(sArray));
  }
}
