object Currying {
  def main(args: Array[String]) {
    val nums = List(1, 2, 3, 4, 5, 6, 7, 8)
    println(filter(nums, modN(2))) // List(2, 4, 6, 8)
    println(filter(nums, modN(3))) // List(3, 6)
  }

  def modN(n: Int)(x: Int) = ((x % n) == 0)

  def filter(l: List[Int], f: Int => Boolean): List[Int] = {
    if (l.isEmpty) l
    else if (f(l.head)) l.head :: filter(l.tail, f)
    else filter(l.tail, f)
  }
}
