object Basics {
  def main(args: Array[String]) {
     basics()
  }

  def basics() = {
    val someTwo = Some(2)
    println(someTwo.map( i => i*i ))                    // Some(4)
    println(someTwo.map( i => "Value : %d".format(i) )) // Some(Value : 2)

    println(someTwo.map( i => Some(i) ))                // Some(Some(2))
    println(someTwo.map( i => Some(i) ).flatten)        // Some(2)
    println(someTwo.flatMap( i => Some(i) ))            // Some(2)
  }
}
