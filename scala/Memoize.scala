class Memoize[A, B](f: A => B) {
  import scala.collection.mutable
  private[this] val cache = mutable.Map.empty[A, B]
 
  def apply(x: A): B = {
    if (cache.contains(x)) {
      cache(x)
    } else {
      val y = f(x)
      cache += ((x, y))
      y
    }
  }
}
 
object Memoize {
  def apply[A, B](f: A => B) = new Memoize(f)

  //Uncomment to see how slow this will be
  //val memoizedFact: Int => Int = fact
  val memoizedFib = Memoize(fib)
  def fib(n: Int): Int = {
    if (n < 2) 1
    else memoizedFib(n - 1) + memoizedFib(n - 2)
  }

  def main(args: Array[String]): Unit = {
    println("Fibanocci %s : %d".format(47, memoizedFib(47)))
  }
}
