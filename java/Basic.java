import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

class Basic {
  public static void main(String args[]) {
    String a = "abc";
    String b = "abc";
    System.out.println("a == b : " + a == b); // false (object identity)
    System.out.println("a.equals(b) : " + a.equals(b)); // true

    System.out.println("0123.substring(1): " + "0123".substring(1)); // prints "123"

/*
    StringBuilder sb = new StringBuilder();

    // ArrayList
    ArrayList<String> al = new ArrayList<String>();
    for (String s: args)
      al.add(s);

    for (String s: al)
      sb.append(s).append(",");

    if (sb.length() > 1)
      sb.setLength(sb.length() - 1);

    System.out.println("From ArrayList " + sb); 

    // Vector
    Vector<String> v = new Vector<String>(args.length);

    for (String s: args)
      v.add(s);

    sb.setLength(0);
    for (String s: v)
     sb.append(s).append(",");

    if (sb.length() > 1)
      sb.setLength(sb.length() - 1);

    System.out.println("From Vector " + sb); 
    System.out.println("v[0] " + v.get(0)); 

    // Arrays and sort
    int[] a = {1, 5, 15, 14, 3, 25};
    Arrays.sort(a);
    for (int i : a)
      System.out.println(i);

    // Hash
    Map<Integer,String> h = new HashMap<Integer,String>();
    for (int i=0; i<args.length; ++i)
      h.put(i, args[i]);

    for (Map.Entry<Integer,String> e: h.entrySet())
      System.out.println(e.getKey() + " -> " + e.getValue()); 

    if (h.containsKey(0))
      System.out.println("0 --> " + h.get(0)); 

    System.out.println("10 --> " + h.get(10));

    // Set
    Set<String> s = new HashSet<String>();
    s.add("1");
    s.add("2");
    System.out.println("1 --> " + s.contains("1"));
    System.out.println("10 --> " + s.contains("10"));
*/
  }
}
