import com.twitter.finagle._
import com.twitter.finagle.builder.{Server, ServerBuilder, ClientBuilder}
import org.jboss.netty.buffer.ChannelBuffers.copiedBuffer
import org.jboss.netty.handler.codec.http._
import org.jboss.netty.handler.codec.http.HttpResponseStatus._
import org.jboss.netty.handler.codec.http.HttpVersion.HTTP_1_1
import org.jboss.netty.util.CharsetUtil.UTF_8
import java.net.SocketAddress
import java.net.InetSocketAddress
import com.twitter.finagle.http.Http
import com.twitter.util.Future
import com.twitter.util.TimeConversions._
 
object ClientServer {
  // git clone https://github.com/twitter/finagle/
  // cd finagle
  // ./sbt "project finagle-http" console
  // :load <path-to-this-file>
  // ClientServer.run
  // :quit
  def run() {
    val service = new Service[HttpRequest, HttpResponse] {
      def apply(request: HttpRequest) = {
        val response = new DefaultHttpResponse(HTTP_1_1, OK);
        response.setContent(copiedBuffer("Hello World!", UTF_8));
        Future.value(response)
      }
    }
    val address: SocketAddress = new InetSocketAddress(10000)
    val server = ServerBuilder().codec(Http()).bindTo(address).name("HttpServer").build(service)
 
    val client = ClientBuilder().codec(Http()).hosts(Seq(address)).hostConnectionLimit(1).build()
    val future = client(new DefaultHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/"))
    val resolved = future.get(1.second)
    println(resolved.get().getContent().toString(UTF_8)) 
  }
}
