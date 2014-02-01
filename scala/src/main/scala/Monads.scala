object Monads {
/*
  val f: A => B
  val g: B => C
  //val h = { a => g(f(a)) }
  val h = f andThen g

  val f: A => Option[B]
  val g: B => Option[C]
  val h = 
*/

  // Example
  //   def stringLength(s: String): Option[Int] = Some(s.length)
  //   def half(n: Int): Option[Int] = if (n % 2 == 0) Some(n/2) else None
  //   val composed = { a: String => stringLength(a) flatMap half }
  //   composed("abc")
  //   composed("abcd")
  //
  //   val quarter = { a: Int => half(a) flatMap half }
  //   quarter(8)
  //
  //   val eights = { a: Int => half(a) flatMap half flatMap half }
  //   eights(8)
  sealed trait Option[+A] {
    def map[B](f: A => B) : Option[B]
    def flatMap[B](f: A => Option[B]) : Option[B]
  }
  case class Some[+A](value: A) extends Option[A] {
    def map[B](f: A => B) : Option[B] = Some(f(value))
    def flatMap[B](f: A => Option[B]) : Option[B] = f(value)
  }
  case object None extends Option[Nothing] {
    def map[B](f: Nothing => B) : Option[B] = None
    def flatMap[B](f: Nothing => Option[B]) : Option[B] = None
  }

  def test1 = {
    // returns List(ax, ay, az, bx, by, bz, cx, cy, cz)
    for { // returns List(ax, ay, az, bx, by, bz, cx, cy, cz)
      a <- List("a", "b", "c")
      z <- List("x", "y", "z")
    } yield a + z
  }

  def test2 = {
    // returns List(ax, ay, az, bx, by, bz, cx, cy, cz) - same as above
    List("a", "b", "c") flatMap { a => List("x", "y", "z") map { z => a + z } }
  }

  def test3 = {
    // returns List(List(ax, ay, az), List(bx, by, bz), List(cx, cy, cz))
    List("a", "b", "c") map { a => List("x", "y", "z") map { z => a + z } }
  }

  // Deffered or lazy computation.. 
  class Future[+A](val run: () => A) {
    def map[B](f: A => B): Future[B] =
      new Future(() => f(run()))
    def flatMap[B](f: A => Future[B]): Future[B] =
      // (run andThen f)() // runs immediately
      new Future(() => f(run()).run())
  }

  // Synchronous execution
  def slowIncrement1(n: Int): Int = {
    Thread.sleep(1000)
    n + 1
  }
  // Deffered execution
  def slowIncrement2(n: Int): Future[Int] = {
    new Future({() => Thread.sleep(1000); n + 1}) 
  }
}
