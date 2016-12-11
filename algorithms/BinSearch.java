import java.util.*;

class BinSearch {
  public static void main(String[] args) {
    int[] a = {10,20,40,50,70,80,100};
    for (int num: a) {
      test(a, num - 1);
      test(a, num);
      test(a, num + 1);
    }
  }

  private static void test(int[] a, int num) {
    int idx = binsearch(a, num);
    if (idx == -1)
      System.out.println(String.format("Not found %d", num));
    else
      System.out.println(String.format("found %d %d", num, a[idx]));
  }

  private static int binsearch(int[] a, int num) {
    int start = 0, end = a.length - 1;

    while (end >= start) {
      int mid = (start + end)/2;
      if (a[mid] == num)
        return mid;
      if (a[mid] > num)
        end = mid - 1;
      else
        start = mid + 1;
    }
    return -1;
  }
}
