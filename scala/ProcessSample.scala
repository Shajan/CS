import scala.annotation.tailrec
import java.io._

object ProcessSample {
  def main(args: Array[String]) {
    if (args.length == 1) {
      echoClient(launchServer)
      exit
    } else if (args.length == 2) {
      echoServer
      exit
    }
    //simple
    //advanced
    //exit
    //error
  }

  def simple = {
  import scala.sys.process._
    "ls".!
    println("...........")
    println(Seq("ls", "/tmp").!!)
    println("...........")
  }

  def advanced = {
  import scala.sys.process._
    println("...........")
    // pipe
    "ls"  #| "wc"
    println("...........")
  }

  def launchServer = {
    new ProcessBuilder("scala", "ProcessSample", "a", "b").start()
  }

  def echoServer() = {
    @tailrec def serve(): Unit = {
      val line = readLine()
      line match {
        case "exit" => exit
        case _ => println("%s".format(line))
      }
      serve()
    }
    serve()
  }

  def echoClient(p: Process) = {
    sys.addShutdownHook(p.destroy) // works with ^C, but not with kill -9
    val is = p.getInputStream
    val out = p.getOutputStream

    val t:Thread = new Thread(new Runnable {
      def run() {
        println(scala.io.Source.fromInputStream(is).mkString(""))
      }
    })

    t.start

    val ps: PrintStream = new PrintStream(out)
    ps.print("hello\n")
    ps.print("exit\n")
    ps.flush

    t.join
  }

  def exit = {
    System.exit(0)
  }

  def error = {
    println("Error: should never get here!!")
  }
}
