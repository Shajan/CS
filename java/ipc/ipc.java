/*
 * Benchmark different ipc transports - stdio, shared memory
 *
 * NOTE:
 *   Self contained single source file.
 *   Commandline determines mode (client vs. server).
 *   Server will launch the client.
 *
 * Author Shajan Dasan (sdasan@gmail.com)
 */
import java.io.*;
import java.nio.ByteBuffer;;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;

class ipc {
  private static int sBufferSize = 1024*1024;// Static buffer size
  private static int sDataMultiplier = 1024; // sBufferSize x sDataMultiplier gives the total payload
  private static int sIterations = 10;       // Number of times to send total payload
  private static String sFileName = "data";  // Used as backing file for shared memory
  private static boolean sDebug = false;
  private static byte[] sBuffer = null;      // Static buffer for read/write

  public static void main(String args[]) {
    ITransport transport = new stdio();
    boolean client = false;

    try {
      for (int i=0; i<args.length; ++i) {
        switch (args[i]) {
          case "-client" : client = true; break;
          case "-stdio" : transport = new stdio(); break;
          case "-memmap" : transport = new memmap(); break;
          case "-debug" : sDebug = true; break;
          case "-b" : sBufferSize = Integer.parseInt(args[++i]); break;
          case "-m" : sDataMultiplier = Integer.parseInt(args[++i]); break;
          case "-i" : sIterations = Integer.parseInt(args[++i]); break;
          case "-f" : sFileName = args[++i]; break;
          default : throw new IllegalArgumentException("Unknown option : " + args[i]);
        }
      }
    } catch (Exception e) {
      log("Usage : ipc [-client] [-stdio|-memmap] [-debug] [-f fileName] [-b bufferSize] [-m dataMultiplier] [-i iterations]");
      logException(e);
      return;
    }

    try {
      sBuffer = new byte[sBufferSize];
      if (client) {
        transport.client();
      } else { 
        // Launch client process with -client flag, in additon to whatever was passed in
        StringBuilder cmdLine = new StringBuilder("java ipc -client");
        for (int i=0; i<args.length; ++i) {
          cmdLine.append(" ");
          cmdLine.append(args[i]);
        }
        ipc.debugLog("Start client..");
        Process p = Runtime.getRuntime().exec(cmdLine.toString());
        logInputStream(p.getErrorStream(), true);
        logInputStream(p.getInputStream(), false);
        Random random = new Random(System.currentTimeMillis());
        random.nextBytes(sBuffer);
        ipc.debugLog("Start server..");
        transport.server(p);
        ipc.debugLog("End server..");
        p.destroy();
        ipc.debugLog("Kill client..");
      }
    } catch (Exception e) {
      logException(e);
    }
  }

  interface ITransport {
    void server(Process clientProcess) throws IOException;
    void client() throws IOException;
  }

  static class memmap implements ITransport {
    @Override
    public void server(Process clientProcess) throws IOException {
      FileChannel fc = new RandomAccessFile(new File(ipc.sFileName), "rw").getChannel();
      final MappedByteBuffer mem = fc.map(FileChannel.MapMode.READ_WRITE, 0, sBufferSize * sDataMultiplier);
      for (int i=0; i<sIterations; ++i)
        new ipc.timed("memmap.write"){{ write(mem.duplicate()); }}.end();
    }

    @Override
    public void client() throws IOException {
      FileChannel fc = new RandomAccessFile(new File(ipc.sFileName), "r").getChannel();
      final MappedByteBuffer mem = fc.map(FileChannel.MapMode.READ_ONLY, 0, sBufferSize * sDataMultiplier);
      do {
        new ipc.timed("memmap.read"){{ read(mem.duplicate()); }}.end();
      } while (true);
    }

    private void write(ByteBuffer bb) throws IOException {
      for (int i=0; i<ipc.sDataMultiplier; ++i)
        bb.put(ipc.sBuffer);
    }

    private void read(ByteBuffer bb) throws IOException {
      for (int i=0; i<ipc.sDataMultiplier; ++i)
        bb.get(ipc.sBuffer);
    }
  }

  static class stdio implements ITransport {
    @Override
    public void server(Process clientProcess) throws IOException {
      final OutputStream os = clientProcess.getOutputStream();
      for (int i=0; i<sIterations; ++i)
        new ipc.timed("stdio.write"){{ write(os); }}.end();
    }

    @Override
    public void client() throws IOException {
      do {
        new ipc.timed("stdio.read"){{ read(System.in); }}.end();
      } while (true);
    }

    private void write(OutputStream os) throws IOException {
      for (int i=0; i<ipc.sDataMultiplier; ++i)
        os.write(sBuffer);
    }

    private void read(InputStream is) throws IOException {
      for (int i=0; i<ipc.sDataMultiplier; ++i) {
        int len = sBuffer.length;
        int bytes = 0;
        int pos = 0;
  
        while (bytes != -1 && pos < len) {
          bytes = is.read(sBuffer, pos, len - pos);
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
