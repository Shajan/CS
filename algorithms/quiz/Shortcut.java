import java.util.*;

class Shortcut {

  public static void main(String[] args) {
    System.out.println("HashMap");
    test_maps(new HashMap<String, Integer>());
    System.out.println("LinkedHashMap");
    test_maps(new LinkedHashMap<String, Integer>());
    System.out.println("TreeMap");
    test_maps(new TreeMap<String, Integer>());

    stack_queue();
    sorting();
  }

  static void test_maps(Map<String, Integer> map) {
/*
╔══════════════╦═════════════════════╦═══════════════════╦══════════════════════╗
║   Property   ║       HashMap       ║      TreeMap      ║     LinkedHashMap    ║
╠══════════════╬═════════════════════╬═══════════════════╬══════════════════════╣
║              ║  no guarantee order ║ sorted according  ║                      ║
║   Order      ║ will remain constant║ to the natural    ║    insertion-order   ║
║              ║      over time      ║    ordering       ║                      ║
╠══════════════╬═════════════════════╬═══════════════════╬══════════════════════╣
║  Get/put     ║                     ║                   ║                      ║
║   remove     ║         O(1)        ║      O(log(n))    ║         O(1)         ║
║ containsKey  ║                     ║                   ║                      ║
╠══════════════╬═════════════════════╬═══════════════════╬══════════════════════╣
║              ║                     ║   NavigableMap    ║                      ║
║  Interfaces  ║         Map         ║       Map         ║         Map          ║
║              ║                     ║    SortedMap      ║                      ║
╠══════════════╬═════════════════════╬═══════════════════╬══════════════════════╣
║              ║                     ║                   ║                      ║
║     Null     ║       allowed       ║    only values    ║       allowed        ║
║ values/keys  ║                     ║                   ║                      ║
╠══════════════╬═════════════════════╦═══════════════════╦══════════════════════╣
║              ║                     ║                   ║                      ║
║Implementation║      buckets        ║   Red-Black Tree  ║    double-linked     ║
║              ║                     ║                   ║       buckets        ║
╚══════════════╩════════════════════════════════════════════════════════════════╝
*/
    //------------[Map]-----------------------
    String[] a = {"foo", "bar", "abc"};
    int i = 0;
    for (String s : a) {
      map.put(s, i++);
      System.out.print(s + ", ");
    }
    System.out.println("");
    
    //------------[Map, Set]-----------------------
    System.out.println("keySet()");
    Set<String> keys = map.keySet();
    for (String key : keys)
      System.out.print(key + ", ");
    System.out.println("");

    System.out.println("entrySet()");
    Set<Map.Entry<String, Integer>> eSet = map.entrySet();
    for (Map.Entry<String, Integer> e : eSet)
      System.out.print(e.getKey() + "->" + e.getValue() + ", ");
    System.out.println("");
    
    //------------[Iterator]-----------------------
    System.out.println("entrySet().iterator()");
    Iterator<Map.Entry<String, Integer>> iter = eSet.iterator();
    while (iter.hasNext())
      System.out.print(iter.next() + ", ");
    System.out.println("");

    //------------[Map, Collection]-----------------------
    System.out.println("values()");
    Collection<Integer> values = map.values();
    for (Integer value : values)
      System.out.print(value + ", ");
    System.out.println("");
  }

  static void stack_queue() {
    Stack<Integer> stack = new Stack<Integer>();
    stack.push(1);
    stack.push(2);
    stack.push(3);
    System.out.println("Stack.push 1, 2, 3");
    
    System.out.println(String.format("Stack.pop %s", stack.pop()));
    for (int i : stack)
      System.out.println(i);

    Queue<String> queue = new LinkedList<String>();
    queue.add("1");
    queue.add("2");
    queue.add("3");
    System.out.println("Queue.add 1, 2, 3");
    
    System.out.println(String.format("Queue.remove %s", queue.remove()));
    for (String s : queue)
      System.out.println(s);
  }

  static void sorting() {
    class Foo {
      int val;
      Foo(int v) { val = v; }
    }

    Comparator<Foo> comparator = new Comparator<Foo>() {
      public int compare(Foo o1, Foo o2) {
        return o1.val - o2.val;
      }
    };

    HashSet<Foo> hSet = new HashSet<Foo>();
    hSet.add(new Foo(100));
    hSet.add(new Foo(20));
    hSet.add(new Foo(150));

    SortedSet<Foo> sSet = new TreeSet<Foo>(comparator);
    sSet.addAll(hSet);

    for (Foo foo : sSet)
      System.out.print(foo.val + ", ");
    System.out.println("");
  }
}
