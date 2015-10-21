class Heap {
  public static void main(String[] args) {
    assert parent(1) == 0;
    assert parent(2) == 0;
    assert parent(3) == 1;
    assert parent(4) == 1;

    assert left(0) == 1;
    assert right(0) == 2;
    assert left(1) == 3;
    assert right(1) == 4;

    //int[] a = new int[]{10, 5, 63, 84, 63, 5, 49, 77, 50, 7, 40, 98, 15};
    int[] a = new int[]{10, 6, 63, 84, 65, 5, 49, 77, 50, 7, 40, 98, 15};

    Heap h = new Heap(a);
    h.print();
    h.sort();
    h.print();
  }

  private int[] a;

  public Heap(int[] a) {
    this.a = a;
    for (int i=1; i<a.length; ++i)
      bubbleup(i); 
  }

  private static boolean isRoot(int idx) { return (idx == 0); }
  private static int parent(int idx) { return (idx - 1) / 2; }
  private static int left(int idx) { return (idx + 1) * 2 - 1; }
  private static int right(int idx) { return left(idx) + 1; }

  private boolean isValid(int idx) { return (idx >= 0) && (idx < a.length); }
  private boolean hasLeft(int idx) { return isValid(left(idx)); }
  private boolean hasRight(int idx) { return isValid(right(idx)); }

  private void swap(int i, int j) {
    int tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }

  private void bubbleup(int idx) {
    if (isRoot(idx))
      return;

    if (a[parent(idx)] > a[idx]) {
      swap(idx, parent(idx));
      bubbleup(parent(idx));
    }
  }

  private void bubbledown(int idx, int last) {
    int candidate = -1;
    if ((left(idx) < last) && (a[idx] > a[left(idx)])) {
      candidate = left(idx);
    }

    if ((right(idx) < last) && (a[idx] > a[right(idx)])) {
      if (candidate == -1)
        candidate = right(idx);
       else if (a[right(idx)] > a[left(idx)])
        candidate = left(idx);
       else 
        candidate = right(idx);
    }

    if (candidate != -1) {
      swap(idx, candidate);
      bubbledown(candidate, last);
    }
  }

  public void sort() {
    for (int i=a.length-1; i>0; --i) {
      swap(0, i);
      bubbledown(0, i);
    }
  }

  public void print() {
    for (int i=0; i<a.length-1; ++i)
      System.out.print(a[i] + ", ");
    if (a.length > 0)
      System.out.println(a[a.length-1]);
    print("", 0, true);
  }

  private void print(String prefix, int node, boolean isLastSibling) {
    System.out.println(prefix + (isLastSibling ? "*---" : "+---") + "[" + a[node] + "]");

    String paddNext = prefix + (isLastSibling ? "    " : "|   ");
    String paddCurrent = paddNext + "|"; 

    if (hasLeft(node)) {
      System.out.println(paddCurrent);
      print(paddNext, left(node), hasRight(node));
    }
    if (hasRight(node)) {
      System.out.println(paddCurrent);
      print(paddNext, right(node), true);
    }
  }
}
