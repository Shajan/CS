/**
 * Find longest substring
 * ("abcd", "xbcy") ==> "bc"
 */

class LongestCommonSubstring {
  public static void main(String[] args) {
    test("", "", "");
    test("a", "a", "a");
    test("a", "b", "");
    test("aaa", "aaa", "aaa");
    test("abc", "abc", "abc");
    test("xabcyabcdz", "pabcqqabzz", "abc");
  }

  private static void test(String a, String b, String expect) {
    String result = lcs(a, b);
    System.out.println(String.format("(%s, %s) => (%s) [success:%s]", a, b, result,
      (result.equals(expect)) ? "true" : String.format("false (expect:%s)", expect)));
  }

  private static String lcs(String a, String b) {
    /***
     * DP
     *  Cols being chars of string a
     *  Rows represent chars of string b
     *
     * Traverse row by row to find a match a(i) == b(j)
     *   set DP[i,j] to DP[i-1,j-1]+1
     *
     * Reduce memory requirement by only having two rows for DP (current and previous)
     */

    int rows = a.length();
    int cols = b.length();

    // Use two arrays to save space for a full metrics
    int[] previousRow = new int[cols];
    int[] currentRow = new int[cols];

    String longest = ""; // Longest so far

    for (int i=0; i<rows; ++i) {
      char r = a.charAt(i);
      for (int j=0; j<cols; ++j) {
        if (r == b.charAt(j)) {
          // Match!
          int matchLength = 1;
          if (j != 0) {
            matchLength += previousRow[j-1];
          }
          currentRow[j] = matchLength; 
          if (matchLength > longest.length()) {
            // Fond a new candidate
            longest = a.substring(i - matchLength + 1, i + 1);
          }
        }
        // Clear out previous array so that it can be used for next round
        if (j != 0) {
          previousRow[j-1] = 0;
        }
      }

      // Reuse previous row, make it current.
      // It is already zero-ed out upto the last item, which won't be read
      int[] tmpRow = previousRow;
      previousRow = currentRow;
      currentRow = tmpRow;
    }

    return longest;
  }
}
