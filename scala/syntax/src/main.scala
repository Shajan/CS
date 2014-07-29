object main extends App {
/*
  if (args.length > 0)
    println("Hello " + args(0))
*/
  ArrayOperations.Run()
//  Precondition.Run()
}

object ArrayOperations {
  val a = Array(1, 2, 4, 6, 8, 10)
  def Run() = {
    a.foreach(println(_))
    b = a.foreach(_ > 5)
    b.foreach(println(_))
  }   
}

object Precondition {
  def Run() = {
    println(Divide(1, 1))
    //println(Divide(1, 0))
  }
  def Divide(i:Int, j:Int):Double = {
    require(j != 0)
    i/j
  }
}