// http://en.wikipedia.org/wiki/Penney's_game

import scala.collection.mutable.ListBuffer
import scala.language.implicitConversions
import scala.language.reflectiveCalls
import util.Random

object PennysGame {

  // Add operator 'times' for int that takes function and calls it that many times
  implicit def intWithTimes(n: Int) = new {
    def times(f: => Unit) = 1 to n foreach {_ => f}
  }

  def log(l: ListBuffer[Boolean]) = {
    def convert(b: Boolean) = b match {
      case true => "H"
      case false => "T" 
    }
    val str = for { b <- l } yield convert(b)
    println(str.mkString)
  } 

  def main(args: Array[String]) {
    val random = new Random(System.currentTimeMillis)
    val l = new ListBuffer[Boolean]
    5.times { l += random.nextBoolean() }
    log(l)
  }
}
