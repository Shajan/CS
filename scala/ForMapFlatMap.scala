object ForMapFlatMap {
  def main(args: Array[String]): Unit = {
    val ln = List(1, 2, 3)
    val ls = List("A", "B", "C")

    // converts to ln flatMap { n => ls map { s => s + n } }
    val f = for {
      n <- ln
      s <- ls
    } yield s + n
    println(f.mkString(",")) // A1,B1,C1,A2,B2,C2,A3,B3,C3

    val m = ln map { n => ls map { s => s + n } }
    println(m.mkString(",")) // List(A1, B1, C1),List(A2, B2, C2),List(A3, B3, C3)

    val fm = ln flatMap { n => ls map { s => s + n } }
    println(fm.mkString(",")) // A1,B1,C1,A2,B2,C2,A3,B3,C3
  }
}
