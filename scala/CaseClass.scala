class Regular(val name: String, val id: Int)

// Implements boilerplate toString, ==, copy, apply, unapply..
case class Case(name: String, id: Int)

object CaseClass {
  def main(args: Array[String]): Unit = {
    val p1 = new Regular("regular", 1)
    val p2 = new Case("case", 1)
    println("regular %s, case %s".format(p1, p2)) // regular Regular@2913f35d, case Case(case,1)

    val p3 = new Regular("regular", 1)
    val p4 = new Case("case", 1)

    if (p1 == p3) println("Huh?")
    if (p2 != p4) println("Huh?")
  }
}

