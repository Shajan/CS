class MinMax {
  public static void main(String args[]) {
    System.out.println(max(10,10,10));
  }

  static int max(int a, int b, int c) {
    if (a > b) {
      if (a > c)
        return a;
      else
        return c;
    } else {
      if (b > c)
        return b;
      else
        return c;
    }
  }

  static int max(int a, int b) {
    if (a > b) {
      return a;
    } else {
      return b;
    }
  }

  static int min(int a, int b) {
    if (a < b) {
      return a;
    } else {
      return b;
    }
  }
}
