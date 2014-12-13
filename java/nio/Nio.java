import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.util.Random;

class Nio {
  private static int sBuffSize = 1024;
  private static int sDataSize = 1024*1024*1024;
  private static int sIterations = 10;
  private static String fileName = "data";
  private static byte[] byteArray = new byte[sBuffSize];
  private static ByteBuffer bb = ByteBuffer.wrap(byteArray);
  private static Random sRandom = new Random(System.currentTimeMillis());

  public static void main(String args[]) throws Exception {
    sRandom.nextBytes(byteArray);
    bb = ByteBuffer.wrap(byteArray);
    for (int i=0; i<sIterations; ++i)
      new timed("nio.array.write"){{ writeFile(fileName, sDataSize); }}.end();
    for (int i=0; i<sIterations; ++i)
      new timed("nio.array.read"){{ readFile(fileName); }}.end();

    bb = ByteBuffer.allocate(sBuffSize);
    bb.put(byteArray);
    for (int i=0; i<sIterations; ++i)
      new timed("nio.allocate.write"){{ writeFile(fileName, sDataSize); }}.end();
    for (int i=0; i<sIterations; ++i)
      new timed("nio.allocate.read"){{ readFile(fileName); }}.end();

    bb = ByteBuffer.allocateDirect(sBuffSize);
    bb.put(byteArray);
    for (int i=0; i<sIterations; ++i)
      new timed("nio.direct.write"){{ writeFile(fileName, sDataSize); }}.end();
    for (int i=0; i<sIterations; ++i)
      new timed("nio.direct.read"){{ readFile(fileName); }}.end();
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
      System.out.println(tag + " " + duration/(1000*1000) + " ms");
    }
  }

  static void writeFile(String fileName, int len) throws Exception {
    FileOutputStream out = new FileOutputStream(fileName);
    FileChannel outChannel = out.getChannel();

    for (int i=0; i<len; i+=sBuffSize) {
      bb.clear();
      outChannel.write(bb);
    }
    // Ignore the remainint, len will be aproximate
  }

  static void readFile(String fileName) throws Exception {
    FileInputStream in = new FileInputStream(fileName);
    FileChannel inChannel = in.getChannel();
    int bytesRead;
    do {
      bb.clear();
      bytesRead = inChannel.read(bb);
    } while (bytesRead != -1);
    inChannel.close();
    in.close();
  }
}

/*
  Results on Mac with SSD drive, all numbers in milliseconds (max, median, min)

  Not much difference in writes
  nio.array.write    8029 5419 4141
  nio.allocate.write 6954 5557 4018
  nio.direct.write   6751 5023 4223

  Direct appears to do well on reads
  nio.array.read     1388 1002  975
  nio.allocate.read  1002  991  973
  nio.direct.read     909  847  825
*/
