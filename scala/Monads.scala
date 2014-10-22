// Monads is a pattern, not a specific type
// http://www.codecommit.com/blog/ruby/monads-are-not-metaphors

object Monads {
  def main(args: Array[String]) {
     simple()
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
}
