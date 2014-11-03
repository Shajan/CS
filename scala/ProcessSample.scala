import scala.sys.process._

object ProcessSample {
  def main(args: Array[String]) {
    //simple
    advanced
    exit
    error
  }

  def simple = {
    "ls".!
    println("...........")
    println(Seq("ls", "/tmp").!!)
    println("...........")
  }

  def advanced = {
    println("...........")
    // pipe
    "ls"  #| "wc"
    println("...........")
  }

  def exit = {
    System.exit(0)
  }

  def error = {
    println("Error: should never get here!!")
  }
}
