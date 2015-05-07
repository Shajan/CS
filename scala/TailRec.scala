import scala.annotation.tailrec

object TailRec {
  def fact(n: Int): Int = {
    @tailrec
    def factHelper(acc: Int, n: Int): Int = {
      if (n <= 0) acc
      else factHelper(acc * n, n - 1)
    }
    factHelper(1, n)
  }

/* The following won't work
  @tailrec
  def fact(n: Int): Int = {
    if (n <= 0) 1
    else n * fact(n-1)
    // can't tail reccurse, as 'n * XX' still needs to be computed
    // Use an accumulator to hold the result of that computation instead
  }
 */

  def reverse[T](l: List[T]): List[T] = {
    @tailrec
    def reverse(acc: List[T], rem: List[T]): List[T] = rem match {
      case Nil => acc
      case head :: tail => reverse(head :: acc, tail)
    }
    reverse(Nil, l)
  }

  def main(args: Array[String]): Unit = {
    println(fact(10))

    val l1 = List(1,2,3,4,5,6,7)
    val l2 = reverse(l1)

    println(l1.mkString(","))
    println(l2.mkString(","))
  }
}

