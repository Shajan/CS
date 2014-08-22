
object Sample {
  def main(args: Array[String]) {
    ArraySample.Run(args)
    //Precondition.Run(args)
  }
}

object Seperator {
  def line(s : String) = {
    println("........... " + s + " ...........")
  }
}

sealed trait Executable {
  def Run(args: Array[String])
}

object ArraySample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("Array")
    val a = Array(10, 20, 30, 40, 50)
    println(a.deep.mkString(","))
    //basic(a)
    arrayType()
    //advanced(a)
  }

  def basic(a: Array[Int]) = {
    Seperator.line("foreach(x => println(x))")
    a.foreach(x => println(x))

    Seperator.line("foreach(println _)")
    a.foreach(println _)

    Seperator.line("for (x <- a) println(x)")
    for (x <- a) println(x)

    Seperator.line("a.filter(_>25)")
    val b = a.filter(_>25)
    b.foreach(println _)
  }

  def arrayType() = {
    ////////////////////////////////////////////////////////////////////////////////////////////
    // Arrays are java Arrays '==' operatior compares reference and can't be overridden
    //
    // Can be wrapped to collection.mutable.WrappedArray which is a Sequence and can be compared
    // for content equality 
    ////////////////////////////////////////////////////////////////////////////////////////////
    print("Array(1, 2, 3) == Array(1, 2, 3) : ")
    println(Array(1, 2, 3) == Array(1, 2, 3))

    print("Array(1, 2, 3) sameElements Array(1, 2, 3) : ")
    println(Array(1, 2, 3) sameElements Array(1, 2, 3))

    print("(Array(1, 2, 3) : Seq[Int]) == (Array(1, 2, 3) : Seq[Int]) : ")
    println((Array(1, 2, 3) : Seq[Int]) == (Array(1, 2, 3) : Seq[Int]))
  }

  def advanced(a: Array[Int]) = {
    ////////////////////////////////////////////////////////////////////////////////////////////
    // Reduce : Collection with element type A to element of type A
    //
    // def reduceLeft[B >: A](f: (B, A) => B): B
    ////////////////////////////////////////////////////////////////////////////////////////////

    Seperator.line("a.reduceLeft(_+_)")
    val sumL = a.reduceLeft(_+_)
    println("sumL = " + sumL)

    Seperator.line("a.reduceRight(_+_)")
    val sumR = a.reduceRight(_+_)
    println("sumR = " + sumR)

    // convert array of int to array of string
    val c = a.map( _.toString )

    Seperator.line("c.reduceLeft(t, e) => if (t.length > e.length) t else e")
    val longerL = c.reduceLeft((t, e) => if (t.length > e.length) t else e)
    println("longerL = " + longerL)

    Seperator.line("c.reduceRight(e, t) => if (t.length > e.length) t else e")
    val longerR = c.reduceRight((e, t) => if (t.length > e.length) t else e)
    println("longerR = " + longerR)

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Fold : Collection with element type A to any type B
    //
    // def foldLeft[B](z: B)(f: (B, A) => B): B
    ////////////////////////////////////////////////////////////////////////////////////////////

    Seperator.line("a.foldLeft(\"\")((t, e) => if (t.length > e.toString.length) t else e.toString)")
    val foldL = a.foldLeft("")((t, e) => if (t.length > e.toString.length) t else e.toString)
    println("foldL = " + foldL)

    Seperator.line("a.foldRight(\"\")((e, t) => if (t.length > e.toString.length) t else e.toString)")
    val foldR = a.foldRight("")((e, t) => if (t.length > e.toString.length) t else e.toString)
    println("foldR = " + foldR)
  }
}

object Precondition extends Executable {
  def Run(args: Array[String]) = {
    println(Divide(1, 1))
    println(Divide(1, 0))
  }
  def Divide(i:Int, j:Int):Double = {
    require(j != 0)
    i/j
  }
}

