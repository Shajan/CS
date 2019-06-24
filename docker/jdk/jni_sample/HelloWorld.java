class HelloWorld {
  static {
    try {
      System.load(new java.io.File( "." ).getCanonicalPath() + "/HelloWorld.so");
    } catch (java.io.IOException e) {
      System.out.println(e);
    } 
  } 

  private native void print();

  public static void main(String[] args) {
    new HelloWorld().print();
  }
}
