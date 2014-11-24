import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;

import static java.nio.file.Files.readAllBytes;
import static java.nio.file.Paths.get;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

public class StaticServer {
  public static void main(String[] args) throws Exception {
    new StaticServer().serve(args[0]);
  }

  public void serve(String fileName) throws IOException {
    HttpServer server = HttpServer.create(new InetSocketAddress(8000), 0);
    server.createContext("/static", new MyHandler(fileName));
    server.setExecutor(null); // creates a default executor
    server.start();
  }

  class MyHandler implements HttpHandler {
    String staticFile;
    MyHandler(String fileName) {
      staticFile = fileName;
    }

    public void handle(HttpExchange t) throws IOException {
      String response = readFile(staticFile);
      t.sendResponseHeaders(200, response.length());
      OutputStream os = t.getResponseBody();
      os.write(response.getBytes(staticFile));
      os.close();
    }
  }

  static String readFile(String path) throws IOException {
    return new String(readAllBytes(get(path)));
  }
}
