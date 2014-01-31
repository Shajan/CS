object Functional {

  // example : toStringAll(List(1, "hello", ('a', List(2, 3, 4), 5)))
  def toStringAll(list: List[Any]): List[String] =
    list match {
      // case x :: y matches atleast one element x in the list y can be Nil
      // => a :: b '::' is the concatination operation
      //   method name is '::', any method name ending with ':' is right associated
      //   equivalent to b.::(a) where '::' is the function name
      //   rhs 'b' is the object whose method is called
      //
      // Nil is the empty list, a singleton object
      case head :: tail => head.toString :: toStringAll(tail)
      case Nil => Nil
    }

  // a function that transforms A to B
  trait Function[A, B] {
    def apply(a: A): B
  }

  def transformHugeStack[A, B](list: List[A], f: Function[A, B]): List[B] =
    list match {
      case head :: tail => f(head) :: transformHugeStack(tail, f)
                        // not tail recursive rhs is the object on which '::' is called
      case Nil => Nil
    }

  def transform[A, B](list: List[A], f: Function[A, B]): List[B] = {
    def go(list: List[A], f: Function[A, B], acc: List[B]): List[B] =
      list match {
        case head :: tail => go(tail, f, f(head) :: acc) // tail reccursion by scala compiler
        case Nil => acc.reverse
      }
    go(list, f, List.empty)
  }
}

