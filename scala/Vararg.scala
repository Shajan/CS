// https://www.lorrin.org/blog/2011/10/04/scalas-missing-splat-operator/

object Vararg {
  def main(args: Array[String]): Unit = {
    process()
    process("a", "b", "c")

    val seq = Seq("ABC", "DEF")
    process(seq: _*)
    println("....")

    val array = Array("X", "Y", "Z")
    process(array: _*)
    println("....")

    val list = List("l1", "l2")
    process(list: _*)
    println("....")

    // Don't know why anyone does this, adding examnple here
    // just in case one comes across this code and needs to understand
    // what it does.
    val args = (10, 5)
    val result = mul _ tupled args 
    println(result)
  }

  // Unclear why this is useful, just take a sequence instead?
  // * at the end means any number of String
  def process(str: String*): Unit = {
    str.foreach(println)
  }

  def mul(a: Int, b: Int): Int = { a * b }
}
