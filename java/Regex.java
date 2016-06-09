import java.util.regex.*;

class Regex {
  public static void main(String args[]) {
    Pattern p = Pattern.compile(args[0]);
    Matcher m = p.matcher(args[1]);
    System.out.println(String.format("regex %s, input %s, matches %s", args[0], args[1], m.matches()));
  }
}
