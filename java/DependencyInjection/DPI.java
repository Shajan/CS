import javax.inject.*;

/*
 * @Named annotation is used to resololve ambiguity
 *
 * No good reference, listing somewhat useful ones
 *   http://antoniogoncalves.org/2011/01/12/bootstrapping-cdi-in-several-environments/
 */

class DPI {
  public static void main(String args[]) {
    Op inc = new Incr();
    Op dec = new Decr();
    System.out.println("Incr(1) " + inc.op(1));
    System.out.println("Decr(1) " + dec.op(1));

    // injection by ctor
    //@Named("square")
    Op sq = new Sq();
    Processor p = new Processor(new Incr());

    System.out.println("Op(1) " + p.byName(1));
    System.out.println("Op(1) " + p.bySetter(1));
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

  @Named("square")
  static class Sq implements Op {
    @Override
    public int op(int i) { return i*i; }
  }

  static class Processor {
    @Inject
    private Op injectByCtor;
 
    @Inject
    @Named("increment")
    private Op injectByName;

    private Op injectBySetter;

    @Inject
    public Processor(Op op) { injectByCtor = op; }

    @Inject
    public void injectUsingMethod(@Named("decrement") Op op) { injectBySetter = op; }

    public int byName(int i) { return injectByName.op(1); }
    public int bySetter(int i) { return injectBySetter.op(1); }
  }
}
