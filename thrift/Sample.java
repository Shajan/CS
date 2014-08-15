// File I/O
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.io.IOException;

// Thrift libraries
import org.apache.thrift.TException;
import org.apache.thrift.TDeserializer;
import org.apache.thrift.TSerializer;
import org.apache.thrift.protocol.TJSONProtocol;
import org.apache.thrift.transport.TFileTransport;

// Thrift generated class
import serializer.KeyVal;

public class Sample {
  public static void main(String[] args) {
    boolean read=true;
    String fileName = "data.bin";
    if (args.length > 0 && args[0].equals("write")) {
      read = false;
    }

    KeyVal kv = new KeyVal();
    if (read) {
      read(fileName, kv);
      System.out.println("key(" + kv.key + "), value(" +  kv.val + ")");
    } else {
      kv.key = "Name";
      kv.val = "Shajan Dasan";
      write(fileName, kv);
    }
  }

  public static void read(String fileName, KeyVal kv) {
    try {
      byte[] bytes = Files.readAllBytes(Paths.get(fileName));
      TDeserializer deserializer = new TDeserializer(new TJSONProtocol.Factory());
      deserializer.deserialize(kv, bytes);
    } catch (IOException|TException e) {
      System.out.println("Exception " + e);
    }
  }

  public static void write(String fileName, KeyVal kv) {
    try {
      TSerializer serializer = new TSerializer(new TJSONProtocol.Factory());
      byte[] bytes = serializer.serialize(kv);
      Files.write(Paths.get(fileName), bytes);
    } catch (IOException|TException e) {
      System.out.println("Exception " + e);
    }
  }
}
