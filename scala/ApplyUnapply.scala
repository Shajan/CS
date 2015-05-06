object ApplyUnapply {
  def main(args: Array[String]): Unit = {
    val f1 = Foo(10)
    f1 match {
      case Foo(i) => println(i)
      case _ => println("Huh?")
    }
    val f2: Foo = null 
    f2 match {
      case Foo(i) => println("Huh?")
      case _ => println("ok")
    }
  }
}

// Private constructor
class Foo private (val i: Int)

// Singleton companion object
object Foo {
  // Factory
  def apply(i: Int): Foo = new Foo(i)

  // Decomposition
  def unapply(foo: Foo): Option[Int] = foo match {
    case f:Foo =>
      println("unapply ok!")
      Some(f.i)
    case null =>
      println("unapply no match")
      None
  }
}
