import java.util.*;

/*
 * Longest sequence
 * Longest sequence ok to have other things in between
 */
class LongestSequence {
  public static void main(String[] args) {
    testContigious();
    testNonContigious();
  }

  static void testContigious() {
    test(longestContigious(new int[]{1}), 1);
    test(longestContigious(new int[]{2,3}), 2);
    test(longestContigious(new int[]{1,2,3,4}), 4);
    test(longestContigious(new int[]{4,2,3,5}), 2);
    test(longestContigious(new int[]{4,0,3,4}), 2);
  }

  static void testNonContigious() {
    test(longestSequenceSlow(new int[]{1}), 1);
    test(longestSequenceSlow(new int[]{2,3}), 2);
    test(longestSequenceSlow(new int[]{1,3,2}), 2);
    test(longestSequenceSlow(new int[]{0,3,4}), 2);
    test(longestSequenceSlow(new int[]{-1,4,4,5,3,4,25,3,2,6,75,6}), 3);

    test(longestSequence(new int[]{1}), 1);
    test(longestSequence(new int[]{2,3}), 2);
    test(longestSequence(new int[]{1,3,2}), 2);
    test(longestSequence(new int[]{0,3,4}), 2);
    test(longestSequence(new int[]{-1,4,4,5,3,4,25,3,2,6,75,6}), 3);
    test(longestSequence(new int[]{-1,4,4,5,3,4,25,3,2,6,75,6,7}), 4);
  }

  static void test(int result, int expected) {
    if (result != expected) {
      System.out.println(String.format("\nFound : %d, Expected : %d", result, expected));
    } else {
      System.out.print(".");
    }
  }

  static int longestContigious(int[] a) {
    int last = a[0]; // needs to have atleast 1 item
    int longest = 1;
    int currentRun = 1;

    for (int i=1; i<a.length; ++i) {
      if (a[i] == last + 1) {
        ++currentRun;
        if (currentRun > longest)
          longest = currentRun;
      } else {
        currentRun = 1;
      }
      last = a[i];
    }

    return longest;
  }

  static int longestSequenceSlow(int[] a) {
    return _longestSequenceReccursive(a[0]+1, a, 1, 1);
  }

  static int _longestSequenceReccursive(int next, int[] a, int start, int seqLenSoFar) {
    int ret = seqLenSoFar;
    for (int i=start; i<a.length; ++i) {
      int seqLen = 0;
      if (a[i] == next)  {
        seqLen = _longestSequenceReccursive(a[i]+1, a, i+1, seqLenSoFar+1);
      } else {
        seqLen = _longestSequenceReccursive(a[i]+1, a, i+1, 1);
      }
      if (seqLen > ret)
        ret = seqLen;
    }

    //System.out.println(String.format("next:%d,len:%d,start:%d,ret:%d", next, seqLenSoFar, start, ret));
    return ret;
  }

  /*
   * Store current best sequence for each index (initialize with 0)
   * Scan from left to right
   *   for each number, increment current best sequenece for if value+1 is present (with higher index)
   *
   * Optimization, do an intial scan and create hash table of indexes for each unique value
   */
  static int longestSequence(int[] a) {
    int globalBest = 0;
    int[] best = new int[a.length];
    Map<Integer, SortedSet<Integer>> map = new HashMap<>();

    // Create a lookup table from value --> set of locations
    for (int i=0; i<a.length; ++i) {
      if (!map.containsKey(a[i])) {
        map.put(a[i], new TreeSet<Integer>());
      }
      map.get(a[i]).add(i);
    }
   
    for (int i=0; i<a.length; ++i) {
      int next = a[i] + 1;
      if (map.containsKey(next)) {
        // list of locations > current index
        SortedSet<Integer> locations = map.get(next).tailSet(i);
        for (int idx : locations) {
          if (best[i] + 1 > best[idx]) {
            best[idx] = best[i] + 1;
            if (best[idx] > globalBest)
              globalBest = best[idx];
          }
        }
      }
    }

    // assuming input has atleast 1 item
    return globalBest+1;
  }
}
