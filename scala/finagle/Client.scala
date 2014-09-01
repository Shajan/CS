// git clone https://github.com/twitter/finagle/
// cd finagle
// ./sbt "project finagle-http" console
// Copy paste these in the REPL console
import org.jboss.netty.handler.codec.http.{DefaultHttpRequest, HttpRequest, HttpResponse, HttpVersion, HttpMethod}
import com.twitter.finagle.Service
import com.twitter.finagle.builder.ClientBuilder
import com.twitter.finagle.http.Http

val client: Service[HttpRequest, HttpResponse] = ClientBuilder().
  codec(Http()).
  hosts("twitter.com:80").
  hostConnectionLimit(1).
  build()

val req = new DefaultHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/")

val f = client(req) // Client, send the request

f onSuccess { res =>
  println("got response", res)
} onFailure { exc =>
  println("failed :-(", exc)
}

