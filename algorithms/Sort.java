
class Sort {
  public static void main(String[] args) {
    //int[] a = new int[]{1, 2, 3};
    //int[] a = new int[]{3, 2, 1};
    //int[] a = new int[]{50, 15, 49, 40};
    int[] a = new int[]{10, 5, 63, 84, 63, 5, 49, 77, 50, 7, 40, 98, 15};
    qsort(a, 0, a.length - 1);
    print(a, 0, a.length - 1);
  }

  public static void print(int[] a, int start, int end) {
    for (int i=start; i<=end; ++i) {
      System.out.print(a[i] + ", ");
    }
    System.out.println("");
  }

  public static void swap(int[] a, int i, int j) {
    System.out.println(a[i] + " <--> " + a[j]);
    int tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }

  public static void qsort(int[] a, int first, int last) {
    print(a, first, last);
    if (first >= last)
      return;

    int pivot = a[first];
    int i = first + 1;
    int j = last;

    while (i < j) {
      while (i < last && a[i] < pivot) ++i;
      while (j > first && a[j] > pivot) --j;
      if (i < j) {
        swap(a, i, j);
        ++i;
        --j;
      }
    }

    print(a, first, last);
    int pos = i;

    if (a[pos] > pivot)
      --pos;

    if (pos > first && pivot > a[pos])
      swap(a, first, pos);

    qsort(a, first, pos-1);
    qsort(a, pos+1, last);
  }
}
