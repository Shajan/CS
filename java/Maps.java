import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

class Maps {
  public static void main(String args[]) {
/*
    System.out.println("HashMap");
    Map<Integer,String> hm = new HashMap<>();
    populate(hm);
    print(hm);

    System.out.println("TreeMap");
    Map<Integer,String> tm = new TreeMap<>();
    populate(tm);
    print(tm);

    System.out.println("LinkedHashMap");
    Map<Integer,String> lm = new LinkedHashMap<>();
    populate(lm);
    print(lm);

    System.out.println("HashMap Iterator");
    print(hm);
    copy_semantics();
*/
    System.out.println("Map Keyset Mutations");
    map_keyset_mutations();
  }

  public static void populate(Map<Integer,String> m) {
    System.out.println("Insert Order (1,3,5,7,9,2,4,6,8)");
    m.put(1, "one");
    m.put(3, "three");
    m.put(5, "five");
    m.put(7, "seven");
    m.put(9, "nine");
    m.put(2, "two");
    m.put(4, "four");
    m.put(6, "six");
    m.put(8, "eight");
  }

  public static void print(Map<Integer,String> m) {
    for (Map.Entry entry : m.entrySet())
      System.out.println(entry.getKey() + " --> " + entry.getValue());
  }

  public static void iterate(Map<Integer,String> m) {
    Iterator iter = m.entrySet().iterator();
    while (iter.hasNext()) {
      Map.Entry entry = (Map.Entry) iter.next();
      System.out.println(entry.getKey() + " --> " + entry.getValue());
    }
  }

  public static void copy_semantics() {
    Map<Integer,String> hm = new HashMap<Integer,String>();
    populate(hm);

    // Check if key set is mutable, vs. a clone
    Set<Integer> keys = hm.keySet();
    try {
      keys.add(100);
      System.out.println("Error! Map.keySet() is mutable");
    } catch (UnsupportedOperationException ex) {
      System.out.println("Map.keySet() is unmutable");
    }

    // Check if value is mutable, vs. a clone
    Collection<String> values = hm.values();
    try {
      values.add("baz");
      System.out.println("Error! Map.value() is mutable");
    } catch (UnsupportedOperationException ex) {
      System.out.println("Map.value() is unmutable");
    }
  }

  public static void map_keyset_mutations() {
    Map<Integer, String> map = new HashMap<>();
    populate(map);

    Set<Integer> set = new HashSet<>();
    set.add(1);
    set.add(3);
    set.add(5);
    set.add(100);

    // Entries in the backing map can be removed using keySet
    // All set remove operations work
    // However entries cannot be added
    //
    // Only keep entries that match the set (1, 3, 5)
    map.keySet().retainAll(set);

    iterate(map); // prints 1, 3, 5
  }
}
