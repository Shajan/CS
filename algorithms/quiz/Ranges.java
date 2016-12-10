import java.util.*;

class Ranges {
  //static class Range implements Comparator<Range> {
  static class Range {
    public int start;
    public int end;
    public Range(int start, int end) { this.start = start; this.end = end; }

    @Override
    public String toString() { return String.format("[%d-%d]", start, end); }
  }

  static class OrderByStart implements Comparator<Range> {
    @Override
    public int compare(Range a, Range b) { return a.start - b.start; }
  }

  static class OrderByEnd implements Comparator<Range> {
    @Override
    public int compare(Range a, Range b) { return a.end - b.end; }
  }

  private static void print(Range[] ranges) { for (Range r : ranges) { System.out.println(r); } }
  private static void print(Collection<Range> ranges) { for (Range r : ranges) { System.out.println(r); } }

  public static void main(String args[]) {
    Range[] ranges = new Range[] {
      new Range(70, 80), new Range(10, 20), new Range(50, 60),
      new Range(55, 73), new Range(47, 57), new Range(12, 60),
      new Range(75, 85), new Range(15, 20), new Range(55, 70)
    };

/* Java 8
    Arrays.sort(ranges, new Comparator<Range>() {
      @Override
      public int compare(Range a, Range b) { return a.start - b.start; }
    });
*/
    Arrays.sort(ranges, new OrderByStart());
    print(ranges);

    PriorityQueue<Range> heap = new PriorityQueue<Range>(ranges.length, new OrderByEnd()); 

    heap.add(ranges[0]);

    for (int i=1; i<ranges.length; ++i) {
      Range r = ranges[i];
      Range endingSoon = heap.remove(); 
      Range nextEntry = new Range(0,0);

      if (endingSoon.end > r.start) {
        heap.add(endingSoon);
        nextEntry.start = r.start; nextEntry.end = r.end;
      } else {
        System.out.println(String.format("Merging %s,%s", endingSoon, r));
        nextEntry.start = endingSoon.start; nextEntry.end = r.end;
      }
      heap.add(nextEntry);
    }

    System.out.println(heap.size());

    while (heap.size() != 0)
      System.out.println(heap.poll());
  }
}
