import java.util.*;

class Sort {
  public static void main(String[] args) {
    tsort(randomArray(10));
    //benchmark(10000000, 10);
  }

  public static int[] randomArray(int size) {
    Random rand = new Random();
    int[] a = new int[size];
    for (int i=0; i<size; ++i) {
      a[i] = rand.nextInt(100);
    }
    return a;
  }

  public static void benchmark(int size, int samples) {
    int[] base = randomArray(size);

    for (int i=0; i<samples; ++i) {
      long start = System.currentTimeMillis();

      int[] a = base.clone();
      Arrays.sort(a);
      System.out.println(String.format("Array.sort ms : %d", System.currentTimeMillis() - start));
      if (!isSorted(a))
        System.out.println("Not sorted!");

      a = base.clone();
      start = System.currentTimeMillis();
      qsort(a);
      System.out.println(String.format("qsort ms : %d", System.currentTimeMillis() - start));
      if (!isSorted(a))
        System.out.println("Not sorted!");
    }
  }

  public static void print(int[] a) {
    System.out.println(a);
  }

  public static void print(int[] a, int start, int end) {
    for (int i=start; i<=end; ++i) {
      System.out.print(a[i] + ", ");
    }
    System.out.println("");
  }

  public static boolean isSorted(int[] a) {
    for (int i=1; i<a.length; ++i) {
      if (a[i-1] > a[i])
        return false;
    }
    return true;
  }

  public static void qsort(int[] a) {
    qsort(a, 0, a.length - 1);
  }

  public static void swap(int[] a, int i, int j) {
    int tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }

  public static void qsort(int[] a, int first, int last) {
    int pivot = a[first + (last - first)/2];
    int i = first;
    int j = last;

    while (i <= j) {
      while (a[i] < pivot) ++i;
      while (a[j] > pivot) --j;
      if (i <= j) {
        swap(a, i, j);
        ++i;
        --j;
      }
    }

    if (first < j) 
      qsort(a, first, j);
    if (i < last) 
      qsort(a, i, last);
  }

  public static void tsort(int[] a) {
    Map<Integer, Set<Integer>> graph = new HashMap<Integer, Set<Integer>>();
    for (int i=0; i<a.length-1; ++i) {
      add_relation(graph, a[i], a[i+1]);
    }
    print(a, 0, a.length-1);
    System.out.println(graph.toString());
  }

  public static void add_relation(Map<Integer, Set<Integer>> graph, int a, int b) {
    if (a == b)
      return;

    int from = Math.min(a, b);
    int to = Math.max(a, b);

    Set<Integer> s = null; 
    if (graph.containsKey(from)) {
      s = graph.get(from);
    } else {
      s = new HashSet<Integer>();
      graph.put(from, s);
    }

    s.add(to);
  }
}
