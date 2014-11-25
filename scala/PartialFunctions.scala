object PartialFunctions {
  def main(args: Array[String]) {
    UsingIsDefinedAt
    UsingCase 
    Collect
    Lists
    Composition
  }

  def UsingIsDefinedAt = {
    val oneByX = new PartialFunction[Int, Int] {
      def apply(x: Int) = 1/x
      def isDefinedAt(x: Int) = x != 0
    }
    println(oneByX.isDefinedAt(1)) // true
    println(oneByX.isDefinedAt(0)) // false
    println(oneByX(1)) // 1
    try {
      oneByX(0) 
    } catch {
      case e: Throwable => println(e) // ArithmeticException: / by zero
    }
  }

  def UsingCase = {
    val oneByX: PartialFunction[Int, Int] = { case x: Int if x != 0 => 1 / x }
    println(oneByX.isDefinedAt(1)) // true
    println(oneByX.isDefinedAt(0)) // false
    println(oneByX(1)) // 1
    try {
      oneByX(0) 
    } catch {
      case e: Throwable => println(e) // scala.MatchError: 0 ..
    }
  }

  def Collect = {
    val inc: PartialFunction[Any, Int] = { case i: Int => i + 1 }
    println(inc.isDefinedAt(1)) // true
    println(inc.isDefinedAt("foo")) // false
    println(inc(1)) // 2

    // collect method takes a partial function
    val lInt = List(1, "foo", 2) collect inc
    println(lInt) // List(2, 3)
  }

  def Lists = {
    // Any instance of Seq, Set, Map is a partial function
    val l = List("a", "b", "c")
    println(l.isDefinedAt(0)) // true
    println(l.isDefinedAt(3)) // false

    // if (isDefined l(1)) l1... l(2), l(10)
    val l1 = Seq(1, 2, 10) collect l
    println(l1) // List(b, c)

    println(l.lift(0)) // Some(a)
    println(l.lift(10)) // None
  }

  def Composition = {
    val one: PartialFunction[Int, String] = { case 1 => "one" }
    val two: PartialFunction[Int, String] = { case 2 => "two" }
    val three: PartialFunction[Int, String] = { case 3 => "three" }
    val unknown: PartialFunction[Int, String] = { case _ => "unknown" }

    val partial = one orElse two orElse three orElse unknown

    println(partial(1)) // one
    println(partial(7)) // unknown
  }
}
