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
  def main(args: Array[String]): Unit = {
    println(fact(10))
  }
}

