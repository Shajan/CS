import scala.util.matching.Regex

object RegEx {
  def main(args: Array[String]) {
    val pattern = new Regex("(S|s)cala")
    val str = "Scala is scalable and cool"
    println((pattern findAllIn str).mkString(",")) // Scala,scala
    println(pattern replaceFirstIn(str, "Java")) // Java is scalable and cool

    val p = "Sleep ([0-9]*)$".r
    "Sleep 100" match {
      case p(n) => println(n.toInt) // prints 100
      case _ => println("no match")
    }

    val simple = "^[/]?(__.*/)?foo/.*".r
    args(0) match {
      case simple(_) => println("match")
      case _ => println("unable to find root folder in: " + args(0) )
    }
  }
}

