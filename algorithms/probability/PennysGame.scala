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

  def convert(b: Boolean) = b match {
    case true => "H"
    case false => "T" 
  }

  def mkString(l: List[Boolean]) = {
    val result = for { b <- l } yield convert(b)
    result.mkString
  }

  def mkString(l: ListBuffer[Boolean]):String = {
    mkString(l.toList)
  }

  def firstIndexOf(l: ListBuffer[Boolean], p: List[Boolean]) = {
    val str = mkString(l)
    val pattern = mkString(p)
//    println("l:" + str + ",p:" + pattern + ",i:" + str.indexOf(pattern))
    val i = str.indexOf(pattern)
    if (i == -1) println(pattern + " Not found in " + str)
    i
  }

  def main(args: Array[String]) {
    val random = new Random(System.currentTimeMillis)

    val patterns = List(
              List(true, true, true),
              List(false, true, true),
              List(false, false, false),
              List(true, false, false))

    val lbase = new ListBuffer[Boolean]
    val flips = 100000
    val window = 200
    val samples = flips - window

    flips.times { lbase += random.nextBoolean() }

    for (p <- patterns) {
      var sum = 0
      val ll = lbase.sliding(window)
      for (l <- ll) {
        sum = sum + firstIndexOf(l, p)
      }
      println(mkString(p) + ":" + sum/samples)
    }
  }
}
