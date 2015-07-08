import scala.util.{Try, Success, Failure}

object ScalaTry {
  def main(args: Array[String]) {
    basics()
  }

  def getInt(): Try[Int] = Try {
    import scala.util.Random._
    if (nextBoolean()) 1 else throw new Exception("original")
  }

  def basics() = {
    val result: Try[Int] = getInt()
    println("result : %s".format(result))
    println("getOrElse : %d".format(result.getOrElse(0)))

    result match {
      case Success(i) => println("success")   // result.get will return int
      case Failure(e) => println("exception") // result.get will throw
    }

    // Maps Success to another Success, flatMap works simillarly
    val plus10Try: Try[Int] = result map { i => i + 10 }
    println("plus10 : %s".format(plus10Try))

    // handle takes a partial function that return another value or transform exception
    val fiveOnFailure: Try[Int] = result.recover {
      case e: Exception => 5
    }
    println("fiveOnFailure : %s".format(fiveOnFailure))

    val anotherExceptionOnFailure: Try[Int] = result.recover {
      case e: Exception => throw new Exception("transformed") 
    }
    println("anotherExceptionOnFailure : %s".format(anotherExceptionOnFailure))
  }
}
