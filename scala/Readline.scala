import scala.io.Source

object Readline {
  def main(args: Array[String]) {
    for (line <- Source.fromFile(args(0)).getLines())
      println(line)
  }
}
