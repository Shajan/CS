// git clone https://github.com/twitter/finagle/
// cd finagle
// ./sbt "project finagle-http" console
// Copy paste these in the REPL console
import com.twitter.finagle.Service
import com.twitter.finagle.http.Http
import com.twitter.util.Future
import com.twitter.util.TimeConversions._
import org.jboss.netty.handler.codec.http._
import java.net.{SocketAddress, InetSocketAddress}
import com.twitter.finagle.builder.{Server, ServerBuilder, ClientBuilder}

// Define our service: OK response for root, 404 for other paths
val rootService = new Service[HttpRequest, HttpResponse] {
  def apply(request: HttpRequest) = {
    val r = request.getUri match {
      case "/" => new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK)
      case _ => new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.NOT_FOUND)
    }
    Future.value(r)
  }
}

// Serve our service on a port
val address: SocketAddress = new InetSocketAddress(10000)
val server: Server = ServerBuilder().
  codec(Http()).
  bindTo(address).
  name("HttpServer"). // For debugging
  build(rootService)

// Now call the server
val client = ClientBuilder().codec(Http()).hosts(Seq(address)).hostConnectionLimit(1).build()
client(new DefaultHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/")).get(1.second)
client(new DefaultHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/foo")).get(1.second)

