object Implicit {
  def main(args: Array[String]) {
    ImplicitMethod.test
    ImplicitClass.test
    ImplicitParams.test
    ImplicitPipe.test
  }
}

// Implicit method
object ImplicitMethod {
import scala.language.implicitConversions
  implicit def num(s: String): Int = { Integer.parseInt(s) }

  def test() {
    val n: Int = "10"
    println("\"10\" => " + n) // prints 10
  }
}

// Implicit class
object ImplicitClass {
  def test() {
    print("10 square", 10 square)
    print("4 fatorial", 4 factorial)
  }

  // Only one argument for implict classes
  implicit class Helper(x: Int) {

    def square: Int = x*x

    def factorial: Int = { 
      var y: Int = 1
      for (i <- 2 to x)
        y *= i
      y
    }
  }

  def print(txt: String, result: Int): Unit = {
    println("%s => %d".format(txt, result))
  }
}

// Implicit paramter
object ImplicitParams {
  abstract class Monoid[A] {
    def add(x: A, y: A): A
    def unit: A
  }

  implicit val stringMonoid: Monoid[String] = new Monoid[String] {
    def add(x: String, y: String): String = x concat y
    def unit: String = ""
  }
  
  implicit val intMonoid: Monoid[Int] = new Monoid[Int] {
    def add(x: Int, y: Int): Int = x + y
    def unit: Int = 0
  }
  
  // Select m based on 'A' stringMonoid or intMonoid
  def sum[A](xs: List[A])(implicit m: Monoid[A]): A = {
    if (xs.isEmpty) m.unit
    else m.add(xs.head, sum(xs.tail))
  }

  def test() {
    println(sum(List(1, 2, 3)))       // uses IntMonoid implicitly
    println(sum(List("a", "b", "c"))) // uses StringMonoid implicitly
  }
}

// https://www.scala-academy.com/tutorials/scala-pipe-operator-tutorial
// A common pattern
implicit class Pipe[T](val v: T) extends AnyVal {
  def |>[U] (f: T => U) = f(v)

  // Additional suggestions:
  def $$[U](f: T => U): T = {f(v); v}
  def #!(str: String = ""): T = {println(str + v); v}
}

object ImplicitPipe {
  import math
  def rate(): Double = { 1.1 }
  def test(): Unit = {
    val a = rate() + math.log(rate()) // what if rate() changes between the two calls?
    val b = { val r = rate(); rate + math.log(rate) }

    // Double converted to Pipe[Double]
    // then apply method |>, which takes a function that takes Double
    val c = rate().|>(r => r + math.log(rate)) 
  }
}
