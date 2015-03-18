import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.util.Random;

class ipc {
  private static int sBuffSize = 1024*1024;
  private static int sDataMultiplier = 10;
  private static int sIterations = 1;
  private static String fileName = "data";
  private static boolean sDebug = false;

  public static void main(String args[]) {
    try {
      if (args.length == 0) {
        Process p = Runtime.getRuntime().exec("java ipc client");
        logInputStream(p.getErrorStream(), true);
        logInputStream(p.getInputStream(), false);
        server(p.getOutputStream());
        p.destroy();
      } else if (args.length == 1) {
        if (args[0].equals("client")) {
          client();
        } else {
          errorLog("Unknown option " + args[0]);
        }
      }
    } catch (Exception e) {
      logException(e);
    }
  }

  static class timed {
    long start;
    String tag;

    timed(String name) {
      start = System.nanoTime();
      tag = name;
    }

    void end() {
      long duration = System.nanoTime() - start;
      ipc.log(tag + " " + duration/(1000*1000) + " ms");
    }
  }

  static void write(OutputStream os, byte[] ba) throws IOException {
    for (int i=0; i<sDataMultiplier; ++i)
      os.write(ba);
  }

  static void read(InputStream is, byte[] ba) throws IOException {
    for (int i=0; i<sDataMultiplier; ++i) {
      int len = ba.length;
      int bytes = 0;
      int pos = 0;

      while (bytes != -1 && pos < len) {
        bytes = is.read(ba, pos, len - pos);
        if (bytes != -1) {
          pos += bytes;
          debugLog("Bytes read : " + bytes);
        }
      }
      if (pos != len)
        errorLog("Read: Unexpected EOF");
    }
  }

  static void server(final OutputStream os) throws IOException {
    debugLog("Start server..");
    Random random = new Random(System.currentTimeMillis());
    final byte[] ba = new byte[sBuffSize];
    random.nextBytes(ba);

    for (int i=0; i<sIterations; ++i)
      new timed("stdio.write"){{ write(os, ba); }}.end();
    debugLog("End server..");
  }

  static void client() throws IOException {
    debugLog("Start client..");
    final byte[] ba = new byte[sBuffSize];
    do {
      new timed("stdio.read"){{ read(System.in, ba); }}.end();
    } while (true);
  }

  static void errorLog(String s) {
    System.err.println(s);
  }

  static void log(String s) {
    System.out.println(s);
  }

  static void debugLog(String s) {
    if (sDebug)
      log(s);
  }

  static void logInputStream(final InputStream is, final boolean fError) throws IOException {
    new Thread() {
      @Override
      public void run() {
        try {
          BufferedReader br = new BufferedReader(new InputStreamReader(is));
          String line;
          while ((line = br.readLine()) != null) {
            if (fError)
              errorLog(line);
            else
              log(line);
          }
        } catch (IOException e) {
        } catch (Exception e) {
          logException(e);
        }
      }
    }.start();
  }

  static void logException(Exception e) {
    StringWriter sw = new StringWriter();
    PrintWriter pw = new PrintWriter(sw);
    e.printStackTrace(pw);
    errorLog(e + " " + sw.toString());
  }
}

