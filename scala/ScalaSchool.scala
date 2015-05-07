object ScalaSchool {
  def main(args: Array[String]) {
    //range()
    //for_loop() // iterate - foreach, do something
    //for_comprehension() // yield result, map/flatMap to another type of same collection
    //array()
    //list()
    //pattern()
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
    for (i <- 0 until 3) println(i)

    val l = List("A", "B", "C")
    for (i <- l) println(i)
    // Equivalent to
    0 until 3 foreach { println(_) }
    l.foreach { println(_) }

    // for and options
    val si:Option[String] = Some("XXXX")
    val none:Option[Int] = None
    for (s <- si) println(s) // Prints XXXX 
    for (s <- none) println(s) // Does not print anything
    // Equivalent to 
    si.foreach { println(_) }
    none.foreach { println(_) }
  }

  def for_comprehension() = {
    // for comprehension - just like map with syntatic sugar
    // Transform each element of the source collection to create
    // a new collection.
    val l = List(1, 2, 3, 4)
    val lsq = for (i <- l) yield i*i 
    println(lsq.mkString(",")) // 1,4,9,16

    val crossproduct = for {
      a <- List("A", "B", "C") // Collection type of first genrator is used for result
      b <- List("X", "Y", "Z")
      concat = "%s.%s".format(a, b)
    } yield concat

    println(crossproduct.mkString(",")) // A.X,A.Y,A.Z,B.X,B.Y,B.Z,C.X,C.Y,C.Z

    // Range will be converted to a Vector
    val even = for (i <- 0 to 10 if (i % 2 == 0)) yield i
    println(even.mkString(",")) // 0,2,4,6,8,10

    // Conditions
    val large = for {
      i <- 0 until 10
      j <- 0 until 10
      sum = i + j
      if (sum) > 15
    } yield sum
    println(large.mkString(",")) // 16,16,17,16,17,18

    // for comprehension and options
    val si:Option[String] = Some("XXXX")
    val none:Option[Int] = None
    println(for (s <- si) yield s) // Some("XXXX")
    println(for (s <- none) yield s) // Does not yield anything
    // Equivalent to 
    println(si.map { x => x }) // Some("XXXX")
    println(none.map { x => x }) // Does not print anything
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

  def pattern() = {
    val a = Array(1, 2, 3, 4, 5) 
    a match {
      case Array(x, y, _*) => println("%d, %d".format(x, y))
      case _ => println("Huh?")
    }
  }
}
