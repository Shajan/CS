import javax.inject.*;

class DPI {
  public static void main(String args[]) {
    Op inc = new Incr();
    Op dec = new Decr();
    System.out.println("Incr(1) " + inc.op(1));
    System.out.println("Decr(1) " + dec.op(1));
  }

  interface Op {
    public int op(int i);
  }

  @Named("increment")
  static class Incr implements Op {
    @Override
    public int op(int i) { return ++i; }
  }

  @Named("decrement")
  static class Decr implements Op {
    @Override
    public int op(int i) { return --i; }
  }
}
