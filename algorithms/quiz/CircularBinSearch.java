import java.util.*;

class CircularBinSearch {
  public static void main(String[] args) {
    //testFindSmallestRotatedList();
    //testBinSearch();
    testFindInRotatedList();
  }

  private static void testFindInRotatedList() {
    int[] a = {5,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100};

    for (int i=0; i<a.length; ++i) {
      rotate(a, i);
      verify(a, 4, findInRotatedList(a, 4));
      verify(a, 5, findInRotatedList(a, 5));
      verify(a, 6, findInRotatedList(a, 6));

      verify(a, 34, findInRotatedList(a, 34));
      verify(a, 35, findInRotatedList(a, 35));
      verify(a, 36, findInRotatedList(a, 36));

      verify(a, 99, findInRotatedList(a, 99));
      verify(a, 100, findInRotatedList(a, 100));
      verify(a, 101, findInRotatedList(a, 101));
      rotate(a, a.length-i);
    }
  }

  private static void testFindInRotatedList(int[] a, int num) {
    int idx = findInRotatedList(a, num);
    if (idx == -1)
      System.out.println(String.format("Not found %d", num));
    else
      System.out.println(String.format("found %d %d", num, a[idx]));
  }

  private static int findInRotatedList(int[] a, int num) {
    int start = 0, end = a.length - 1;

    while (end >= start) {
      if (a[start] < a[end])
        return binsearch(a, start, end, num);

      int mid = (start + end) / 2;
      if (a[mid] == num)
        return mid;

      if (a[mid] < a[end]) {
        if ((num > a[mid]) && (num <= a[end]))
          return binsearch(a, mid + 1, end, num);
        else
          end = mid - 1;
      } else {
        if ((num < a[mid]) && (num >= a[start]))
          return binsearch(a, start, mid - 1, num);
        else
          start = mid + 1;
      }
    }
    return -1;
  }

  private static int binsearch(int[] a, int start, int end, int num) {
    while (start <= end) {
      int mid = (start + end) / 2;
      if (a[mid] == num)
        return mid;
      if (a[mid] > num)
        end = mid - 1;
      else
        start = mid + 1;
    }
    return -1;
  }

  private static void testBinSearch() {
    int[] a = {5,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100};
    testBinSearch(a, 1);
    testBinSearch(a, 5);
    testBinSearch(a, 6);
    testBinSearch(a, 9);
    testBinSearch(a, 10);
    testBinSearch(a, 11);
    testBinSearch(a, 99);
    testBinSearch(a, 100);
    testBinSearch(a, 101);
  }

  private static void testBinSearch(int[] a, int num) {
    int idx = binsearch(a, num);
    if (idx == -1)
      System.out.println(String.format("Not found %d", num));
    else
      System.out.println(String.format("found %d %d", num, a[idx]));
  }

  private static int binsearch(int[] a, int num) {
    int start = 0, end = a.length - 1;
    while (end >= start) {
      int mid = (start + end) / 2;
      if (a[mid] == num)
        return mid;
      if (a[mid] > num)
        end = mid - 1;
      else
        start = mid + 1;
    }
    return -1;
  }

  private static void testFindSmallestRotatedList() {
    int[] a = {5,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100};
    for (int i=0; i<a.length; ++i) {
      rotate(a, i);
      System.out.println(print(a, findOffsetOfSmallest(a)));
      rotate(a, a.length-i);
    }
  }

  private static int findOffsetOfSmallest(int[] a) {
    int start=0, end=a.length-1;

    while (start < end) {
      int mid = (start + end) / 2;
      if (a[mid] > a[end])
        start = mid + 1;
      else
        end = mid;
    }

    return end;
  }

  private static void rotate(int[] a, int n) {
    int[] b = new int[a.length];

    for (int i=0; i<a.length; ++i)
      b[i] = a[i];

    for (int i=0; i<a.length; ++i)
      a[i] = b[(i+n)%(a.length)];
  }

  private static void verify(int a[], int n, int idx) {
    int expectIdx = -1;
    for (int i=0; i<a.length; ++i) {
      if (a[i] == n) {
        expectIdx = i;
        break;
      }
    }

    if (expectIdx != idx) {
      System.out.print("fail ");
    } else {
      System.out.print("pass ");
    }
    System.out.println(print(a, n, idx));
  }

  private static String print(int a[], int n, int idx) {
    if (a.length == 0)
      return String.format("%d:", idx);
    StringBuilder sb = new StringBuilder();
    sb.append(n).append("?");
    if (idx != -1)
      sb.append(idx).append("->").append(a[idx]);
    else
      sb.append("*->X");
    sb.append(":[");
    for (int i=0; i<a.length-1; ++i) {
      sb.append(a[i]).append(",");
    }
    sb.append(a[a.length-1]).append("]");
    return sb.toString();
  }

  private static String print(int a[], int idx) {
    if (a.length == 0)
      return String.format("%d:", idx);
    StringBuilder sb = new StringBuilder();
    sb.append(idx).append("->").append(a[idx]).append(":[");
    for (int i=0; i<a.length-1; ++i) {
      sb.append(a[i]).append(",");
    }
    sb.append(a[a.length-1]).append("]");
    return sb.toString();
  }

  private static String print(int a[]) {
    StringBuilder sb = new StringBuilder();
    sb.append("[");
    for (int i=0; i<a.length-1; ++i) {
      sb.append(a[i]).append(",");
    }
    sb.append(a[a.length-1]).append("]");
    return sb.toString();
  }
}
