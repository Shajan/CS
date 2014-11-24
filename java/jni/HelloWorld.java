class HelloWorld {
  static {
    // Set JVM env to load correctly, for now loading with full path
    //System.loadLibrary("HelloWorld.so");
    System.load("/Users/sdasan/src/CS/java/jni/HelloWorld.so");
  }

  private native void print();

  public static void main(String[] args) {
    new HelloWorld().print();
  }
}
