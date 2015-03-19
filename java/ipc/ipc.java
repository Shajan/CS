import java.io.*;
import java.util.Random;

class ipc {
  private static int sBuffSize = 1024*1024;
  private static int sDataMultiplier = 1024;
  private static int sIterations = 10;
  private static boolean sDebug = false;

  public static void main(String args[]) {
    try {
      if (args.length == 0) {
        Process p = Runtime.getRuntime().exec("java ipc client");
        logInputStream(p.getErrorStream(), true);
        logInputStream(p.getInputStream(), false);
        new stdio().server(p);
        p.destroy();
      } else if (args.length == 1) {
        if (args[0].equals("client")) {
          new stdio().client();
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

  interface IClientServer {
    void server(Process clientProcess) throws IOException;
    void client() throws IOException;
  }

  static class stdio implements IClientServer {
    @Override
    public void server(Process clientProcess) throws IOException {
      ipc.debugLog("Start server..");
      final OutputStream os = clientProcess.getOutputStream();
      Random random = new Random(System.currentTimeMillis());
      final byte[] ba = new byte[sBuffSize];
      random.nextBytes(ba);
  
      for (int i=0; i<sIterations; ++i)
        new ipc.timed("stdio.write"){{ write(os, ba); }}.end();
      ipc.debugLog("End server..");
    }

    @Override
    public void client() throws IOException {
      ipc.debugLog("Start client..");
      final byte[] ba = new byte[sBuffSize];
      do {
        new ipc.timed("stdio.read"){{ read(System.in, ba); }}.end();
      } while (true);
    }

    private void write(OutputStream os, byte[] ba) throws IOException {
      for (int i=0; i<ipc.sDataMultiplier; ++i)
        os.write(ba);
    }

    private void read(InputStream is, byte[] ba) throws IOException {
      for (int i=0; i<ipc.sDataMultiplier; ++i) {
        int len = ba.length;
        int bytes = 0;
        int pos = 0;
  
        while (bytes != -1 && pos < len) {
          bytes = is.read(ba, pos, len - pos);
          if (bytes != -1) {
            pos += bytes;
            ipc.debugLog("Bytes read : " + bytes);
          }
        }
        if (pos != len)
          ipc.errorLog("Read: Unexpected EOF");
      }
    }
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
