object Functional {
  def toStringAll(list: List[Any]): List[String] =
    list match {
      case head :: tail => head.toString :: toStringAll(tail)
      case Nil => Nil
    }

  def transformHugeStack[A, B](list: List[A], f: Function[A, B]): List[B] =
    list match {
      case head :: tail => f(head) :: transform(tail, f)
      case Nil => Nil
    }

  def transform[A, B](list: List[A], f: Function[A, B]): List[B] = {
    def go(list: List[A], f: Function[A, B], acc: List[B]): List[B] =
      list match {
        case head :: tail => go(tail, f, f(head) :: acc) // simulate tail reccursion by scala compiler
        case Nil => acc.reverse
      }
    go(list, f, List.empty)
  }
}

