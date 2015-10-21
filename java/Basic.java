import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import java.util.Vector;

class Basic {
  public static void main(String args[]) {
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

    // Hash
    Map<Integer,String> h = new HashMap<Integer,String>();
    for (int i=0; i<args.length; ++i)
      h.put(i, args[i]);

    for (Map.Entry<Integer,String> e: h.entrySet())
      System.out.println(e.getKey() + " -> " + e.getValue()); 

    // Arrays and sort
    int[] a = {1, 5, 15, 14, 3, 25};
    Arrays.sort(a);
    for (int i : a)
      System.out.println(i);
  }
}
