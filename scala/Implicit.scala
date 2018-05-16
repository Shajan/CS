object Implicit {
  def main(args: Array[String]) {
    print("10 square", 10 square)
    print("4 fatorial", 4 factorial)
  }

  // Only one argument for implict classes
  implicit class Helper(x: Int) {

    def square: Int = { x*x }

    def factorial: Int = { 
      var y: Int = 1
      for (i <- 2 to x)
        y *= i
      y
    }
  }

  def print(txt: String, result: Int): Unit = {
    println("%s => %d".format(txt, result))
  }
}
