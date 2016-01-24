import java.util.*;

    String[] a = {"foo", "bar", "baz"};
    HashMap<String, Integer> h = new HashMap<String, Integer>();
    int i = 0;
    for (String s : a) {
      h.put(s, i++);
    }
    
    Set<Map.Entry<String, Integer>> s = h.entrySet();
    
    for (Map.Entry<String, Integer> e : s) {
      System.out.println(e.getKey());
      System.out.println(e.getValue());
    }
    
    Iterator<Map.Entry<String, Integer>> iter = s.iterator();
    
    while (iter.hasNext()) {
      System.out.println(iter.next());
    }

Comparator<Foo> comparator = new Comparator<Foo>() {
  public int compare(Foo o1, Foo o2) {
    ...
  }
}
SortedSet<Foo> keys = new TreeSet<Foo>(comparator);
keys.addAll(map.keySet());


    Stack<Integer> stack = new Stack<Integer>();
    stack.push(17);
    stack.push(100);
    stack.push(107);
    
    System.out.println(String.format("Pop %s", stack.pop()));
    for (int i : stack)
      System.out.println(i);
    
    
    Queue<String> queue = new LinkedList<String>();
    queue.add("0");
    queue.add("1");
    queue.add("2");
    
    System.out.println(String.format("Remove %s", queue.remove()));
    for (String s : queue)
      System.out.println(s);