object CallByName {
  def callByValue(x: Int) = { x + x }

  // Evertime x is evaluated, the fucntion that returns value of x
  // is evaluated
  def callByName(x: => Int) = { x + x }

  def main(args: Array[String]) {
    println("callByValue(10) = " + callByValue(10)) // 20
    println("callByName(10) = " + callByName(10)) // 20

    println("callByValue({10}) = " + callByValue({10})) // 20
    println("callByName({10}) = " + callByName({10})) // 20

    var x = 0
    println("callByValue({x += 10}) = " + callByValue({x += 10; x})) // 20
    println("x = " + x) // 10

    x = 0
    println("callByName({x += 10}) = " + callByName({x += 10; x})) // 30
    println("x = " + x) // 20
  }
}
