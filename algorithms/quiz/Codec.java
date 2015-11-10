// Run-length coding:
// "a4b4c4" -> "aaaabbbbcccc"
// "a" -> "a1"
// "1" -> "11"

class Test {
  public static void main(String[] args) {
    System.out.println(encode("aa"));
    System.out.println(decode("a2"));
    System.out.println(encode("aabbbb"));
    System.out.println(decode("a2b4"));
    assert (decode(encode("aaabbc")).equals("aaabbc")); 
  }

  static String decode(String in) {
    assert (in != null && in.length() > 0);
    StringBuilder sb = new StringBuilder();
    char lastChar = in.charAt(0);
    int i = 1;
    while (i < in.length()) {
      int count = 0;
      StringBuilder sNum = new StringBuilder();
      while (i < in.length() && Character.isDigit(in.charAt(i))) {
        sNum.append(in.charAt(i));
        ++i;
      }
      if (sNum.length() > 0) {
        count = Integer.parseInt(sNum.toString()); // throws
        for (int k=0; k<count; ++k) {
          sb.append(lastChar);
        }
      }
      if (i < in.length()) {
        lastChar = in.charAt(i);
        ++i;
      }
    }
    return sb.toString();
  }
  
  static String encode(String in) {
    assert (in != null && in.length() > 0);
    StringBuilder sb = new StringBuilder();
    int count = 0;
    char lastChar = 0;
    for (int i=0; i<in.length(); ++i) {
      if (i == 0) {
        count = 1;
        lastChar = in.charAt(i);
      } else {
        if (lastChar == in.charAt(i)) {
          count++;
        } else {
          sb.append(lastChar);
          sb.append(count);
          count = 1;
          lastChar = in.charAt(i);
        }
      }
    }
    sb.append(lastChar);
    sb.append(count);
    return sb.toString();
  }
}
