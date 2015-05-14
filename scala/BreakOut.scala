import scala.collection._

// Reference
//
// http://stackoverflow.com/questions/1715681/scala-2-8-breakout/1716558#1716558
// breakout gives the compiler a suggestion as to which Builder to choose implicitly
// (essentially it allows the compiler to choose which factory it thinks fits the situation best.)

object BreakOut {
  def main(args: Array[String]): Unit = {
    val l = List(1, 2, 3)
    println(l)   //List(1, 2, 3)

    val imp = l.map(_ + 1)
    println(imp) //List(2, 3, 4)

    val b = l.map(_ + 1)(breakOut)
    println(b)   //Vector(2, 3, 4)

    val arr: Array[Int] = l.map(_ + 1)(breakOut)
    println(arr)   //Array

    val set: Set[Int] = l.map(_ + 1)(breakOut)
    println(set)   //Set(2, 3, 4)

    val map: Map[Int, String] = l.map { i => i -> i.toString } (breakOut)
    println(map)   //Map(1 -> "1", 2 -> "2", 3 -> "3")
  }
}
