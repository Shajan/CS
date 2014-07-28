
object Sample {
  def main(args: Array[String]) {
    MyArray.Run(args)
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

object MyArray extends Executable {
  def Run(args: Array[String]) = {
    val a = Array(10, 20, 30, 40, 50)
    //basic(a)
    advanced(a)
  }

  def basic(a: Array[Int]) = {
    Seperator.line("Array")

    a.foreach(x => println(x))
    Seperator.line("foreach(x => println(x))")

    a.foreach(println _)
    Seperator.line("foreach(println _)")

    for (x <- a) println(x)
    Seperator.line("for (x <- a) println(x)")

    val b = a.filter(_>25)
    b.foreach(println _)
    Seperator.line("a.filter(_>25)")
  }

  def advanced(a: Array[Int]) = {
    ////////////////////////////////////////////////////////////////////////////////////////////
    // Reduce : Collection with element type A to element of type A
    //
    // def reduceLeft [B >: A] (f: (B, A) => B): B
    ////////////////////////////////////////////////////////////////////////////////////////////

    val sumL = a.reduceLeft(_+_)
    println("sumL = " + sumL)
    Seperator.line("a.reduceLeft(_+_)")

    val sumR = a.reduceRight(_+_)
    println("sumR = " + sumR)
    Seperator.line("a.reduceRight(_+_)")

    // convert array of int to array of string
    val c = a.map( _.toString )

    val longerL = c.reduceLeft((t, e) => if (t.length > e.length) t else e)
    println("longerL = " + longerL)
    Seperator.line("c.reduceLeft(t, e) => if (t.length > e.length) t else e")

    val longerR = c.reduceRight((e, t) => if (t.length > e.length) t else e)
    println("longerR = " + longerR)
    Seperator.line("c.reduceLeft(e, t) => if (t.length > e.length) t else e")

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Fold : Collection with element type A to any type B
    //
    // def foldLeft[B](z: B)(f: (B, A) => B): B
    ////////////////////////////////////////////////////////////////////////////////////////////

    val foldL = a.foldLeft("")((t, e) => if (t.length > e.toString.length) t else e.toString)
    println("foldL = " + foldL)
    Seperator.line("a.foldLeft(\"\")((t, e) => if (t.length > e.toString.length) t else e.toString)")

    val foldR = a.foldRight("")((e, t) => if (t.length > e.toString.length) t else e.toString)
    println("foldR = " + foldR)
    Seperator.line("a.foldRight(\"\")((e, t) => if (t.length > e.toString.length) t else e.toString)")
  }
}
