object Vararg {
  def main(args: Array[String]): Unit = {
    process()
    process("a", "b", "c")

    val seq = Seq("ABC", "DEF")
    process(seq:_*)

    val array = Array("X", "Y", "Z")
    process(array:_*)
  }

  // * at the end means any number of String
  def process(str: String*): Unit = {
    str.foreach(println)
    println("....")
  }
}
