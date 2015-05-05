object ScalaSchool {
  def main(args: Array[String]) {
    //range()
    //for_loop()
    //array()
    list()
  }

  def range() = {
    println("0 until 10 : %s".format(0 until 10))
    println("0 to 10 : %s".format(0 to 10))
    println("0 until 10 by 2 : %s".format(0 until 10 by 2))
/* Output
    0 until 10 : Range(0, 1, 2, 3, 4, 5, 6, 7, 8, 9) // immutable.Range
    0 to 10 : Range(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) // immutable.Range.Inclusive
    0 until 10 by 2 : Range(0, 2, 4, 6, 8) // Range
*/
  }

  def for_loop() = {
    // iterates over Range(0,1,2), (0 until 3) is a range
    for (i <- 0 until 3) {
      println(i)
    }
  }

  def array() = {
    val a = new Array[Int](10)
    a(0) = 1 // paranthesis instead of square
    println(a(0))
    println(a.size)

    val b = new collection.mutable.ArrayBuffer[Int]
    b.append(1,2,3)
    val a1 = b.toArray
    println(a1.size)
  }

  def list() = {
    // linked list with head and tail
    val l1 = List(1, 2, 3)
    println(l1.head) // 1
    println(l1.tail) // List(2, 3)
    println(l1(1))   // 2

    // Cons operator "::", right associated
    // Invoking cons method on the right side of operator, argument left side of operator
    val l2 = "A" :: Nil
    println(l2) // List(A)
    val l3 = "A" :: "B" :: "C" :: Nil
    println(l3) // List(A, B, C)
    val l4 = "A" :: ("B" :: ("C" :: Nil)) // equivalent to l3
    println(l4) // List(A, B, C)
  }
}
