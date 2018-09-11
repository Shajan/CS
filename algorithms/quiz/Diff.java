import java.util.*;

// Find difference of two maps
class Diff<K,V> {
  public static <K,V> List<K> diffFast(Map<K,V> a, Map<K,V> b) {
    List<K> d = new ArrayList<K>();

    for (Map.Entry<K,V> entry : a.entrySet()) {
      K key = entry.getKey();
      if (!b.containsKey(key)) {
        d.add(key);
      } else {
        V value = b.get(key);
        if (value != entry.getValue())
          d.add(key);
        b.remove(key); // mutates list
      }
    }

    for (K key : b.keySet()) {
      d.add(key);
    }

    return d;
  }

  public static <K,V> List<K> diffSlow(Map<K,V> a, Map<K,V> b) {
    List<K> d = new ArrayList<K>();
    diff(a, b, d);
    diff(b, a, d);
    return d;
  }

  private static <K,V> void diff(Map<K,V> a, Map<K,V> b, List<K> diff) {
    for (Map.Entry<K,V> entry : a.entrySet()) {
      K key = entry.getKey();
      if (!b.containsKey(key)) {
        diff.add(key);
      } else {
        V value = b.get(key);
        if (value != entry.getValue())
          diff.add(key);
      }
    }
  }

  public static void test(Map<Integer,Integer> a, Map<Integer,Integer> b) {
    List<Integer> d1 = diffFast(a, b);
    List<Integer> d2 = diffSlow(a, b);

    if (d1.containsAll(d2) && d2.containsAll(d1))
      System.out.println("Success!");
    else
      System.out.println("Error!");
  }

  private static Map<Integer,Integer> squareMap(int n) {
    Map<Integer,Integer> map = new HashMap<Integer,Integer>();
    for (int i=0; i<n; ++i) {
      map.put(i, i*i);
    }
    return map;
  }

  public static void main(String args[]) {
    test(squareMap(4), squareMap(4));
/*
    test(squareMap(3), squareMap(4));
    test(squareMap(4), squareMap(3));
    test(squareMap(1), squareMap(5));

    Map<Integer,Integer> a = squareMap(5);
    a.put(3,1);
    test(a, squareMap(5));
    test(squareMap(5), a);
*/
  }
}
