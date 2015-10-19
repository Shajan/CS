import java.util.ArrayList;

class Strings {
  public static void main(String[] args) {
    escape("abc def");
    inline_escape("abc def  ".toCharArray());
  }

  public static void escape(String s) {
    System.out.println(s);
    char[] ca = s.toCharArray();
    StringBuilder sb = new StringBuilder();
    for (int i=0; i<ca.length; ++i) {
      if (ca[i] == ' ')
        sb.append("%20");
      else
        sb.append(ca[i]);
    }
    System.out.println(sb);
  }

  public static void inline_escape(char[] c) {
    System.out.println(new String(c));
    int i = c.length-1;
    int j = c.length-1;

    while (c[i] == ' ' && i>0)
      --i;

    for (; i>=0; --i) {
      if (c[i] == ' ') {
        c[j--] = '0';
        c[j--] = '2';
        c[j--] = '%';
      } else {
        c[j--] = c[i];
      }
    }
    System.out.println(new String(c));
  }
}
