import java.util.Arrays;
import java.util.Random;

class Sort {
  public static void main(String[] args) {
    benchmark(10000000, 10);
  }

  public static void benchmark(int size, int samples) {
    Random rand = new Random();
    int[] base = new int[size];
    for (int i=0; i<size; ++i) {
      base[i] = rand.nextInt();
    }

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
}
