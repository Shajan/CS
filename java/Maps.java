import java.util.Map;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.LinkedHashMap;
import java.util.Iterator;

class Maps {
  public static void main(String args[]) {
    Map<String,String> hm = new HashMap<String,String>();
    Map<String,String> tm = new TreeMap<String,String>();
    Map<String,String> lm = new LinkedHashMap<String,String>();
    populate(hm);
    populate(tm);
    populate(lm);

    System.out.println("HashMap");
    print(hm);
    System.out.println("TreeMap");
    print(tm);
    System.out.println("LinkedHashMap");
    print(lm);

    System.out.println("HashMap Iterator");
    print(hm);
  }

  public static void populate(Map<String,String> m) {
    m.put("1", "one");
    m.put("2", "two");
    m.put("3", "three");
    m.put("4", "four");
    m.put("5", "five");
    m.put("6", "six");
    m.put("9", "nine");
    m.put("7", "seven");
    m.put("8", "eight");
  }

  public static void print(Map<String,String> m) {
    for (Map.Entry entry : m.entrySet())
      System.out.println(entry.getKey() + " --> " + entry.getValue());
  }

  public static void iterate(Map<String,String> m) {
    Iterator iter = m.entrySet().iterator();
    while (iter.hasNext()) {
      Map.Entry entry = (Map.Entry) iter.next();
      System.out.println(entry.getKey() + " --> " + entry.getValue());
    }
  }
}
