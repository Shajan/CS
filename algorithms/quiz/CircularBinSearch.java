import java.util.*;

class CircularBinSearch {
  public static void main(String[] args) {
    int[] a = {5,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100};

    for (int i=0; i<a.length; ++i) {
      rotate(a, i);
      System.out.println(print(a, findOffsetOfSmallest(a)));
      rotate(a, a.length-i);
    }
  }

  private static String print(int a[], int n) {
    if (a.length == 0)
      return String.format("%d:", n);
    StringBuilder sb = new StringBuilder();
    sb.append(n).append("->").append(a[n]).append(":[");
    for (int i=0; i<a.length-1; ++i) {
      sb.append(a[i]).append(",");
    }
    sb.append(a[a.length-1]).append("]");
    return sb.toString();
  }

  private static void rotate(int[] a, int n) {
    int[] b = new int[a.length];

    for (int i=0; i<a.length; ++i)
      b[i] = a[i];

    for (int i=0; i<a.length; ++i)
      a[i] = b[(i+n)%(a.length)];
  }

  private static int findOffsetOfSmallest(int[] a) {
    int mid=0, start=0, end=a.length-1;

    while (start <= end) {
      mid = (start + end)/2;
      if (a[mid] < a[start])
        start = mid + 1;
      else
        end = mid - 1;
    }

    return mid;
  }

/*
  private static void test(int[] a, int num) {
    int idx = binsearch(a, num);
    if (idx == -1)
      System.out.println(String.format("Not found %d", num));
    else
      System.out.println(String.format("found %d %d", num, a[idx]));
  }

  private static int adj(int[] a, int offset, int mid) {
    int len = a.length;
    return (mid + offset) % len;
  }

  private static int binsearch(int[] a, int num, int offset) {
    int start = 0, end = a.length - 1;

    while (end >= start) {
      int mid = (start + end)/2;
      if (a[adj(a, offset, mid)] == num)
        return mid;
      if (a[adj(a, offset, mid)] > num)
        end = mid - 1;
      else
        start = mid + 1;
    }
    return -1;
  }
*/
}
