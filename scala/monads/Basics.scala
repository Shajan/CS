object Basics {
  def main(args: Array[String]) {
     //mapFlatMap()
     forComprehension()
  }

  def mapFlatMap() = {
    val someTwo = Some(2)
    println(someTwo.map( i => i*i ))                    // Some(4)
    println(someTwo.map( i => "Value : %d".format(i) )) // Some(Value : 2)

    println(someTwo.map( i => Some(i) ))                // Some(Some(2))
    println(someTwo.map( i => Some(i) ).flatten)        // Some(2)
    println(someTwo.flatMap( i => Some(i) ))            // Some(2)
  }

  def optInt: Option[Int] = {
    import scala.util.Random._
    //if (nextBoolean()) Some(nextInt()%100) else None
    if (nextBoolean()) Some(2) else None
  }

  def forComprehension() = {
    // nested maps
    val a = optInt.flatMap(i => optInt.map(j => i + j))
    println(a)  // Some(4) or None

    // simpler to use for comprehension
    val b = for {
      i <- optInt  // return type is determined by the first one
      j <- optInt
    } yield i + j
    println(b) // Some(4) or None
  }
}
