import java.io.Closeable

object FunctionSyntax {

  // NOTE: map is already present in the List class
  // List(1,2,3) map { _ * 2 }
  //
  // usage : map(List(1,2,3)) { a => a * 2 }
  def map[A, B](list: List[A])(f: A => B): List[B] =
    list match {
      case head :: tail => f(head) :: map(tail) { f }
      case Nil => Nil
    }

  // do something, then finally close
  // usage : withClosable(new FileInputStream("foo")) { f => .. }
  def withCloseable[A <: Closeable](closeable: A)(f: A => Unit) = {
    try {
      f(closeable)
    } finally {
      closeable.close()
    }
  }

  // NOTE: filter is already present in the List class
  // List(1,2,3) filter { _ % 2  == 0 }
  //
  // usage : filter(List(1,2,3)) { a => a % 2 == 0 }
  // usage : filter(List(1,2,3)) { _ % 2 == 0 } 
  // '_' is the first argument to the function
  // usage : filter(List(1,2,3)) { println("."); a => a % 2 == 0 }
  //     bug.. println gets called only once, it's not part of the function
  //     defined by the function expression 'a => a % 2 == 0'
  def filter[A](list: List[A])(f: A => Boolean): List[A] =
    list match {
      case head :: tail =>
        if (f(head))
          head :: filter(tail)(f)
        else
          filter(tail)(f)
      case Nil => Nil
    }

  // Take a list of A, with an accumulator
  // foldLeft(List(1,2,3), 0) { (b, a) => b + a }
  // foldLeft(List(1,2,3), 0) { _ + _ }
  def foldLeft[A, B](list: List[A], acc: B)(f: (B, A) => B): B =
    list match {
      case head :: tail => foldLeft(tail, f(acc, head)) { f }
      case Nil => acc
    }

  // usage mapUsingFoldLeft(List(1,2,3)) { _ * 2 }
  def mapUsingFoldLeft[A, B](list: List[A])(f: A => B): List[B] =
    foldLeft(list, List.empty[B]) { (acc, elem) => f(elem) :: acc }

  // NOTE: groupBy is already present in the List class
  // example: 1 to 10 groupBy { _ % 3 }
  //
}

