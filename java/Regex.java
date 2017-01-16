import java.util.regex.*;

class Regex {
  public static void main(String args[]) {
/*
    Pattern p = Pattern.compile(args[0]);
    Matcher m = p.matcher(args[1]);
    System.out.println(String.format("regex %s, input %s, matches %s", args[0], args[1], m.matches()));
*/

    Pattern p = Pattern.compile("^.*[0-9]*.*$");
    Matcher m = p.matcher("abc123xyz456");
    System.out.println(m.matches()); // prints true

    p = Pattern.compile("-?[0-9]+");
    m = p.matcher("abc123xyz456xxx");

    while (m.find()) {
      System.out.println(m.group());
    }
  }
}
