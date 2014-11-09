import scala.annotation.tailrec
import java.io._

object ClientServer {
  val nClients = 2
  val nServers = 8 
  def main(args: Array[String]) {
    if (args.length == 0) {
      val servers = launchServers(nServers)
      sys.addShutdownHook({for (server <- servers) server.destroy}) // works with ^C, but not with kill -9
      echoClient(nClients, servers)
    } else {
      args(0) match {
        case "server" => echoServer
        case _ => usage
      }
    }
  }

  def usage = println("scala ClientServer [client|server]")

  def exit = System.exit(0)

  def launchServers(count: Int): Seq[Process] = {
    var servers = Seq[Process]() 
    0 to count foreach { _ =>
      servers :+ Seq(new ProcessBuilder("scala", "ClientServer", "server").start)
    }
    servers
  }

  def echoServer = {
    @tailrec def serve(): Unit = {
      val line = readLine()
      line match {
        case "exit" => exit
        case _ => println("%s".format(line))
      }
      serve
    }
    serve
  }

  def echoClient(count: Int, p: Seq[Process]): Unit = {
    0 to count foreach { i => callServer(p(i)) }
  }

  def callServer(p: Process) = {
    val is = p.getInputStream
    val out = p.getOutputStream

    val t:Thread = new Thread(new Runnable {
      def run() {
        println(scala.io.Source.fromInputStream(is).mkString(""))
      }
    })

    t.start

    val ps: PrintStream = new PrintStream(out);
    ps.print("hello\n")
    ps.print("exit\n")
    ps.flush

    t.join
  }

}
