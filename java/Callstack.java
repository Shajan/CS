import java.io.StringWriter;
import java.io.PrintWriter;

class Callstack {
  public static void main(String args[]) {
    f(10);
  }

  public static void f(int i) {
    System.out.println("i = " + i);
    try {
      if ((i != 0) && (i % 5) == 0)
        printCallStack();
      else if ((i != 0) && ((i % 7) == 0))
        throw new Exception("blah");
      if (i > 0)
        f(i-1);
    } catch (Exception e) {
       System.out.println("................Exception.......");
       printException(e);
       System.out.println("................Excepiton string.......");
       System.out.println(getStackTrace(e));
       System.out.println("................Excepiton end.......");
    }
  }

  public static void printCallStack() {
    System.out.println("................dumpStack.......");
    Thread.dumpStack();
    System.out.println("................getCallStack.......");
    System.out.println(getCallStack("\n", 1));
    System.out.println("................CallStack end.......");
  }

  public static String getCallStack(String delimitter, int framesToSkip) {
    StackTraceElement[] frames = Thread.currentThread().getStackTrace();
    StringBuffer sb = new StringBuffer();

    // First frame is java getStackTrace(), second one is this method, skip both
    for (int i=2 + framesToSkip; i<frames.length - 1; ++i) {
      sb.append(frames[i]).append(delimitter);
    }
    sb.append(frames[frames.length - 1]);
    return sb.toString();
  }

  public static void printException(Exception e) {
    e.printStackTrace(System.out);
  }

  public static String getStackTrace(Exception e) {
    StringWriter writer = new StringWriter();
    PrintWriter printWriter = new PrintWriter(writer);
    e.printStackTrace(printWriter);
    printWriter.flush();
    return writer.toString();
  }
}

