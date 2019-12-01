import java.util.*;

// Increment a base 10 number
class Increment {

  private static char c2n(char c) { return (char)(c - '0'); }
  private static char n2c(char c) { return (char)(c + '0'); }

  private static String incr(String num) throws Exception {
    // String will be in reverse order
    StringBuilder sb = new StringBuilder();

    // Flag that shows no more increment is 
    // necessary, just copy rest of the string
    boolean done = false;

    char cArray[] = num.toCharArray();

    // Go thru one char at a time right to left
    for (int i=cArray.length - 1; i>=0; --i) {
      char c = cArray[i];
      if (!done) {
        char n = c2n(c);
        if (n > 9 || n < 0)
          throw new Exception("Invalid input " + num);

        if (n == 9) {
          n = 0;
        } else {
          ++n;
          done = true;
        }

        c = n2c(n);
      }
      sb.append(c);
    }

    // Add a leading '1'
    if (!done) {
      sb.append('1');
    }
    return sb.reverse().toString();
  }

  public static void main(String args[]) throws Exception {
    unit_test_conversion();
    unit_test_increment();
  }

  public static void unit_test_conversion() throws Exception {
    if (c2n('0') != 0) throw new Exception("Fail c2n 0");
    if (c2n('1') != 1) throw new Exception("Fail c2n 1");
    if (c2n('9') != 9) throw new Exception("Fail c2n 9");
    if (n2c((char)0) != '0') throw new Exception("Fail n2c 0");
    if (n2c((char)1) != '1') throw new Exception("Fail n2c 1");
    if (n2c((char)9) != '9') throw new Exception("Fail n2c 9");
  }

  public static void unit_test_increment() throws Exception {
    if (!incr("0").equals("1")) throw new Exception("Fail 0");
    if (!incr("9").equals("10")) throw new Exception("Fail 9");
    if (!incr("15").equals("16")) throw new Exception("Fail 15");
    if (!incr("95").equals("96")) throw new Exception("Fail 95");
    if (!incr("99").equals("100")) throw new Exception("Fail 99");
    if (!incr("199").equals("200")) throw new Exception("Fail 199");
    if (!incr("156799").equals("156800")) throw new Exception("Fail 156799");
  }
}
