/*
 * Benchmark different ipc transports - stdio, shared memory, socket
 *
 * NOTE:
 *   Self contained single source file.
 *   Commandline determines mode (client vs. server).
 *   Server will launch the client.
 *
 * Shajan Dasan (sdasan@gmail.com)
 */
import java.io.*;
import java.net.*;
import java.nio.*;
import java.nio.channels.*;
import java.util.Random;

class ipc {
  private static int sBufferSize = 1024*1024;// Static buffer size
  private static int sDataMultiplier = 1024; // sBufferSize x sDataMultiplier gives the total payload
  private static int sIterations = 10;       // Number of times to send total payload
  private static String sFileName = "data";  // Used as backing file for shared memory
  private static int sPort = 8080;           // TCP server port for socket
  private static byte[] sBuffer;             // Static buffer for read/write
  private static ByteBuffer sByteBuffer;     // Static ByteBuffer for read/write
  private static Process sClientProcess;     // Client process launched by server, only valid in server mode
  private static boolean fNio = false;       // Force using nio (channel & ByteBuffer) for stdio
  private static boolean sDebug = false;

  public static void main(String args[]) {
    ITransport transport = new stdio();
    boolean client = false;
    boolean direct = false;

    try {
      for (int i=0; i<args.length; ++i) {
        switch (args[i]) {
          case "-client" : client = true; break;
          case "-stdio" : transport = new stdio(); break;
          case "-memmap" : transport = new memmap(); break;
          case "-socket" : transport = new socket(); break;
          case "-direct" : direct = true; break;
          case "-nio" : fNio = true; break;
          case "-debug" : sDebug = true; break;
          case "-b" : sBufferSize = Integer.parseInt(args[++i]); break;
          case "-m" : sDataMultiplier = Integer.parseInt(args[++i]); break;
          case "-i" : sIterations = Integer.parseInt(args[++i]); break;
          case "-p" : sPort = Integer.parseInt(args[++i]); break;
          case "-f" : sFileName = args[++i]; break;
          default : throw new IllegalArgumentException("Unknown option : " + args[i]);
        }
      }
    } catch (Exception e) {
      log("Usage : ipc [-client] [-stdio|-memmap] [-debug] [-direct] [-nio] [-p port] [-f fileName] [-b bufferSize] [-m dataMultiplier] [-i iterations]");
      logException(e);
      return;
    }

    try {
      sBuffer = new byte[sBufferSize];
      if (direct)
        sByteBuffer = ByteBuffer.allocateDirect(sBufferSize);
      else
        sByteBuffer = ByteBuffer.allocate(sBufferSize);
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
        sClientProcess = Runtime.getRuntime().exec(cmdLine.toString());
        logInputStream(sClientProcess.getErrorStream(), true);
        logInputStream(sClientProcess.getInputStream(), false);
        Random random = new Random(System.currentTimeMillis());
        random.nextBytes(sBuffer);
        ipc.debugLog("Start server..");
        transport.server();
        ipc.debugLog("End server..");
        sClientProcess.destroy();
        ipc.debugLog("Kill client..");
      }
    } catch (Exception e) {
      logException(e);
    }
  }

  interface ITransport {
    void server() throws IOException;
    void client() throws IOException;
  }

  static class stdio implements ITransport {
    @Override
    public void server() throws IOException {
      final OutputStream os = sClientProcess.getOutputStream();
      final WritableByteChannel channel = Channels.newChannel(os);
      if (fNio) {
        for (int i=0; i<sIterations; ++i)
          new ipc.timed("stdio.write"){{ write(channel); }}.end();
      } else {
        // Duplicating for loop to reduce perf variations, useful when comparing with other transports
        for (int i=0; i<sIterations; ++i)
          new ipc.timed("stdio.write"){{ write(os); }}.end();
      }
    }

    @Override
    public void client() throws IOException {
      final ReadableByteChannel channel = Channels.newChannel(System.in);
      if (fNio) {
        for (int i=0; i<sIterations; ++i)
          new ipc.timed("stdio.read"){{ read(channel); }}.end();
      } else {
        // Duplicating for loop to reduce perf variations, useful when comparing with other transports
        for (int i=0; i<sIterations; ++i)
          new ipc.timed("stdio.read"){{ read(System.in); }}.end();
      }
    }

    private void write(WritableByteChannel channel) throws IOException {
      for (int i=0; i<ipc.sDataMultiplier; ++i)
        channel.write(ipc.getByteBuffer());
    }

    private void read(ReadableByteChannel channel) throws IOException {
      for (int i=0; i<ipc.sDataMultiplier; ++i)
        channel.read(ipc.getByteBuffer());
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

  static class memmap implements ITransport {
    @Override
    public void server() throws IOException {
      File f = new File(ipc.sFileName);
      FileChannel fc = new RandomAccessFile(f, "rw").getChannel();
      final MappedByteBuffer mem = fc.map(FileChannel.MapMode.READ_WRITE, 0, sBufferSize * sDataMultiplier);
      for (int i=0; i<sIterations; ++i)
        new ipc.timed("memmap.write"){{ write(mem.duplicate()); }}.end();
      fc.close();
    }

    @Override
    public void client() throws IOException {
      // TODO: Cross process synchronize read/write, ok for now as we are interested only in raw throuput
      File f = new File(ipc.sFileName);
      FileChannel fc = new RandomAccessFile(f, "r").getChannel();
      final MappedByteBuffer mem = fc.map(FileChannel.MapMode.READ_ONLY, 0, sBufferSize * sDataMultiplier);
      for (int i=0; i<sIterations; ++i)
        new ipc.timed("memmap.read"){{ read(mem.duplicate()); }}.end();
      fc.close();
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

  static class socket implements ITransport {
    @Override
    public void server() throws IOException {
      final SocketChannel sc = SocketChannel.open();
      sc.connect(new InetSocketAddress("localhost", ipc.sPort));
      for (int i=0; i<sIterations; ++i)
        new ipc.timed("socket.write"){{ write(sc); }}.end();
      sc.close();
    }

    @Override
    public void client() throws IOException {
      final SocketChannel sc = SocketChannel.open();
      sc.connect(new InetSocketAddress("localhost", ipc.sPort));
      for (int i=0; i<sIterations; ++i)
        new ipc.timed("socket.read"){{ read(sc); }}.end();
      sc.close();
    }

    private void write(SocketChannel sc) throws IOException {
      for (int i=0; i<ipc.sDataMultiplier; ++i)
        sc.write(ipc.getByteBuffer());
    }

    private void read(SocketChannel sc) throws IOException {
      for (int i=0; i<ipc.sDataMultiplier; ++i)
        sc.read(ipc.getByteBuffer());
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

  static ByteBuffer getByteBuffer() {
    sByteBuffer.clear();
    return sByteBuffer;
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
