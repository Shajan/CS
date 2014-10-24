// Monads is a pattern, not a specific type
// http://www.codecommit.com/blog/ruby/monads-are-not-metaphors

object Monads {
  def main(args: Array[String]) {
     simple()
     advanced()
  }

/*
 * Any time you start with something which you pull apart and use to compute a new something
 * of that same type, you have a monad. It’s really as simple as that.
 */
  case class Thing[+A](value: A) {
    def bind[B](f: A => Thing[B]) = f(value)
  }

/*
 * We also have this fancy bind function, which digs inside our Thing and allows a function
 * which we supply to use that value to create a new Thing. Scala calls this function “flatMap“.
 */

  def simple() {
    def foo(i: Int) = Thing(i + 1)
    val a = Thing(1)
    val b = a bind foo
    withoutMonad()
    usingMonad()
  }

  // A mock database get funciton
  def db(row: Int, col: String) = if (row%2 == 0) "foo" else null

  def withoutMonad() = {
    def firstName(rowId: Int): String = db(rowId, "fName") 
    def lastName(rowId: Int): String = db(rowId, "lName")
    def fullName(rowId: Int): String = {
      val fname = firstName(rowId)
      if (fname != null) {
        val lname = lastName(rowId)
        if (lname != null) fname + " " + lname else null
      } else null
    }
  }

  sealed trait Option[+A] {
    def bind[B](f: A => Option[B]): Option[B]
  }
 
  case class Some[+A](value: A) extends Option[A] {
    def bind[B](f: A => Option[B]) = f(value)
  }
 
  // Nothing is kinda special in scala it is a subtype of everything
  case object None extends Option[Nothing] {
    def bind[B](f: Nothing => Option[B]) = None
  }

  def optiondb(row: Int, col: String) = {
    val s = db(row, col)
    if (s == null) None else Some(s)
  }

  def usingMonad() = {
    def firstName(rowId: Int): Option[String] = optiondb(rowId, "fName")
    def lastName(rowId: Int): Option[String] = optiondb(rowId, "lName")
    def fullName(rowId: Int): Option[String] = {
      firstName(rowId) bind { fname =>
        lastName(rowId) bind { lname =>
          Some(fname + " " + lname)
        }
      }
    }
  }

import scala.language.higherKinds

  trait Monad[M[_]] {
    def unit[A](a: A): M[A]
    def bind[A, B](m: M[A])(f: A => M[B]): M[B]
  }
 
  implicit object ThingMonad extends Monad[Thing] {
    def unit[A](a: A) = Thing(a)
    def bind[A, B](thing: Thing[A])(f: A => Thing[B]) = thing bind f
  }
 
  implicit object OptionMonad extends Monad[Option] {
    def unit[A](a: A) = Some(a) // <--- if a is null?
    def bind[A, B](opt: Option[A])(f: A => Option[B]) = opt bind f
  }
 
  // Convert any List[M[A]] to M[List[A]]
  def sequence[M[_], A](ms: List[M[A]])(implicit tc: Monad[M]) = {
    ms.foldRight(tc.unit(List[A]())) { (m, acc) =>
      tc.bind(m) { a =>
        tc.bind(acc) { tail => 
          tc.unit(a :: tail) 
        } 
      }
    }
  }

  def advanced() = {
    val lThings = List(Thing(1), Thing(2), Thing(3))
    val tList = Monads.sequence(lThings)
    println(tList) // Thing(List(1, 2, 3))
    val lOptions: List[Option[Int]] = List(Some(1), Some(2), Some(3))
    val oList = Monads.sequence(lOptions)
    println(oList) // Some(List(1, 2, 3))
  }
}
