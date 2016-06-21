import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

class Randomize {
  public static void main(String args[]) {
    List<Integer> a = Arrays.asList(100, 20, 30, 150, 200, 175, 30, 210);
    Collections.sort(a, new NumberComparator());
    for (int i : a) {
      System.out.println(i);
    }

    Collections.shuffle(a, new Random(System.nanoTime()));
    System.out.println("After randomizing");
    for (int i : a) {
      System.out.println(i);
    }
  }

  static class NumberComparator implements Comparator<Integer> {
    @Override
    public int compare(Integer i1, Integer i2) {
      // Natural sort order: larger value is greater
      return i1 - i2;
    }
  }
}
