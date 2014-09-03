// Imports for feature
import scala.concurrent._
import scala.concurrent.duration._
import ExecutionContext.Implicits.global

object Sample {
  def main(args: Array[String]) {
    CmdLineSample.Run(args)
    //FutureSample.Run(args)
    //FlatMapSample.Run(args)
    //OptionSample.Run(args)
    //ControlSample.Run(args)
    //ArraySample.Run(args)
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

object CmdLineSample extends Executable {
  def parse(args: List[String], K: Set[String], KV: Set[String],
            parsed: Map[String, String]): Map[String, String] = {
    args match {
      case Nil => parsed // Return parsed
      case key :: value :: tail if KV.contains(key) => parse(tail, K, KV, parsed ++ Map(key -> value))
      case value :: tail if K.contains(value) => parse(tail, K, KV, parsed ++ Map(value -> value))
      case _ => {
        println("Unknown option: " + args.mkString(","))
        Map() 
      }
    }
  }

  def Run(args: Array[String]) = {
    Seperator.line("CmdLine")
    val cmd = parse(args.toList, Set("Help", "--?"), Set("-n", "-age"), Map())
    for ((k,v) <- cmd) { println(k + "=" + v) }
  }
}

object FutureSample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("Future")
    Basic()
  }

  def Basic() = {
    println("1.Defining f")
    val f = future {
      println("....Start f....")
      Thread.sleep(10)
      println("....End   f....")
      "f"
    }
    println("2.Defining f.onSuccess")
    // Async call
    f.onSuccess { case s => println(".... onSuccess: " + s) }

    println("3.Defining f.foreach")
    // see http://stackoverflow.com/questions/2173373/scala-foreach-strange-behaviour
    // Process value
    f.foreach(s => (println(".... foreach Result: " + s )))

    // Create a new feature by applying a function to success results
    println("4.Defining f.map")
    val fMap = f.map(x => {
      println("....Start fMap....")
      Thread.sleep(10)
      println("....End   fMap....")
      "(m)" + _
    })

    // Create a new feature by applying a function to success results
    // return new feature
    println("5.Defining f.flatMap")
    val fFlatMap = f.flatMap(x => {
      future {
        println("....Start fFlatMap....")
        Thread.sleep(10)
        println("....End   fFlatMap....")
        "(fm)" + _
      }
    })

    // Create a new feature by filtering value
    println("6.Defining f.filter")
    val fFilter = f.filter(x => {
      println("....Start filter....")
      Thread.sleep(10)
      println("....End   filter....")
      true
    })

    println("7.Start Wait all")
    val r = for {
      r1 <- f
      r2 <- fMap
      r3 <- fFlatMap
    } yield (r1, r2, r3)
    val (a, b, c) = Await.result(r, Duration.Inf)
    //Await.ready(f, Duration.Inf)
    println("Last.End Wait All")
  }
}

object FlatMapSample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("FlatMap")
    val l = List(1,2,3,4,5)
    println("List: " + l.mkString(","))
    val m = Map(1->3, 2->6, 3->9)
    println("Map: " + m.mkString(","))
    basic(l, m)
  }
  def basic(l:List[Int], m:Map[Int,Int]) = {
    val by2 = l.map(_*2)
    println("l.map(_*2) : " + by2.mkString(",")) // 2,4,6,8,10

    def even(x:Int) = if ((x%2)==0) Some(x) else None
    val some = l.map( even _ )
    println("l.map( even _ ) : " + some.mkString(","))
    // None,Some(2),None,Some(4),None

    val lfm = l.flatMap( even _ )
    println("l.flatMap( even _) : " + lfm.mkString(",")) // 2,4

    val ll = l.map(x=>List(x-1, x, x+1))
    println("l.map(x=>List(x-1, x, x+1)): " + ll.mkString(","))
    // List(0, 1, 2),List(1, 2, 3),List(2, 3, 4),List(3, 4, 5),List(4, 5, 6)

    val fm = l.flatMap(x=>List(x-1, x, x+1))
    println("l.flatMap(x=>List(x-1, x, x+1)): " + fm.mkString(","))
    // 0,1,2,1,2,3,2,3,4,3,4,5,4,5,6


    val ml = m.toList
    println("m.toList: " + ml.mkString(",")) // (1,3),(2,6),(3,9)
    println("m.mapValues(_*2): " + m.mapValues(_*2).mkString(",")) // 1 -> 6,2 -> 12,3 -> 18
    println("m.mapValues : m.mapValues { even _ }: " + m.mapValues { even _ } )
    // Map(1 -> None, 2 -> Some(6), 3 -> None)

    def evenMap(k:Int, v:Int) = if ((k%2)==0) Some(k->v) else None
    // _1, _2 first, second fileds of a touple :
    // m.flatMap { e => f(e._1, e_2) }
    // m.flatMap { (k,v) => f(k,v) } Syntax not supported in scala
    println("m.flatMap { case (k,v) => evenMap(k,v) }: " + m.flatMap { case (k,v) => evenMap(k,v) })
    // Map(2 -> 6)

    // Using filter
    println("m.filter(e => even(e._2)) != None: " +  m.filter(e => even(e._2) != None))
    println("m.filter{ case (k,v)=> even(v) != None }: " + m.filter{ case (k,v)=> even(v) != None })
    println("m.filter{ case (k,v)=> even(v).isDefined }: " + m.filter{ case (k,v)=> even(v).isDefined })
    // All three above gives same result Map(1 -> 2, 2 -> 4, 3 -> 6)
    // Map(2 -> 6)
  }
}

object OptionSample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("Option")
    // lCase.get returns Option { Some[String] or None }
    val lCase = Map("A" -> "a", "B" -> "b")
    basic(lCase)
    //controlled(lCase)
    //default(lCase)
  }
  def basic(lCase: Map[String, String]) {
    val a = lCase.get("A")
    val x = lCase.get("X")
    println("lCase of A isEmpty? : " + a.isEmpty)
    println("lCase of A : " + a)
    println("lCase of X isEmpty? : " + x.isEmpty)
    println("lCase of X : " + x)
    print("a.foreach( println _ ) : ")
    a.foreach( println _ )
    print("x.foreach( println _ ) : ")
    x.foreach( println _ )
  }
  def controlled(lCase: Map[String, String]) {
    def convert(o: Option[String]) = o match {
      case Some(s) => s
      case None => "None"
    }
    println("lCase of A : " + convert(lCase.get("A")))
    println("lCase of X : " + convert(lCase.get("X")))
  }
  def default(lCase: Map[String, String]) {
    val a = lCase.get("A")
    val x = lCase.get("X")
    println("lCase of A : " + a.getOrElse("?"))
    println("lCase of X : " + x.getOrElse("?"))
  }
}

object ControlSample extends Executable {
  def Run(args: Array[String]) = {
    exceptions()
    loops()
  }

  def exceptions() = {
    Seperator.line("Exception")
    try {
      throw new RuntimeException("RuntimeException")
    } catch {
      case e: RuntimeException => println(e.toString)
    } finally {
      println("finally")
    }

    Seperator.line("Return values")

/*  Comment to avoid warning in finally

    def f() = try { 1 } finally { 2 }
    val a = f()  // a == 1
    if (a != 1) println("Error! a should be 1")

    def g() = try {
      throw new RuntimeException
    } catch {
      case e: RuntimeException => 2
    } finally {
      3
    }
    val b = g() // b == 2
    if (b != 2) println("Error! b should be 2")
*/

    // If we have return statement, return type need to be declared
    def h(): Int = try {
      throw new RuntimeException
    } catch {
      case e: RuntimeException => return 2
    } finally {
      return 3
    }
    val c = h() // b == 3
    if (c != 3) println("Error! c should be 3")
  }

  def loops() = {
    Seperator.line("for loops")
    for (i <- 1 to 4)
      print(i.toString + " ") // 1,2,3,4
    println
    for (i <- 1 until 4)
      print(i.toString + " ") // 1,2,3
    println

    Seperator.line("All Files")
    val files = (new java.io.File(".")).listFiles
    for (file <- files)
      print(file.length + " ")
    println

    Seperator.line("Scala Files")
    for (
      file <- files
      if file.isFile
      if file.getName.endsWith(".scala")
    ) println(file)  // only *.scala

    Seperator.line("Multiple generators inside ()")
    for (
      i <- 1 to 5 if (i % 2 == 1); // Need ';' here as for condition is in '(' ')'
      j <- 6 to 10 if (j % 2 == 0);
      k = i*j
    ) print(i.toString + "x" + j.toString + "->" + k.toString + " ")  // i odd, j even
    println

    Seperator.line("Multiple generators inside {}")
    for {
      i <- 1 to 5 if (i % 2 == 0) // ';' not required here as for condition is in '{' '}'
      j <- 6 to 10 if (j % 2 == 1)
      k = i*j
    } print(i.toString + "x" + j.toString + "->" + k.toString + " ")  // i even, j odd
    println

    Seperator.line("yield")
    val scalaFiles =
      for (file <- files if file.getName.endsWith(".scala"))
        yield file
    println(scalaFiles.mkString(", "))
  }
}

object ArraySample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("Array")
    val a = Array(10, 20, 30, 40, 50)
    println(a.mkString(","))
    //basic(a)
    //arrayType()
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

  ////////////////////////////////////////////////////////////////////////////////////////////
  // Arrays are java Arrays. '==' operatior compares reference and can't be overridden
  ////////////////////////////////////////////////////////////////////////////////////////////
  def arrayType() = {
    print("Array(1, 2, 3) == Array(1, 2, 3) : ")
    println(Array(1, 2, 3) == Array(1, 2, 3)) // false

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Can be wrapped to collection.mutable.WrappedArray which is a Sequence and can be compared
    // for content equality
    ////////////////////////////////////////////////////////////////////////////////////////////

    print("(Array(1, 2, 3) : Seq[Int]) == (Array(1, 2, 3) : Seq[Int]) : ")
    println((Array(1, 2, 3) : Seq[Int]) == (Array(1, 2, 3) : Seq[Int])) // true

    print("Array(1, 2, 3) sameElements Array(1, 2, 3) : ")
    println(Array(1, 2, 3) sameElements Array(1, 2, 3)) // true

    ////////////////////////////////////////////////////////////////////////////////////////////
    // sameElements won't be sufficient if array elelements only have default == defined
    // Use .deep to convert all Arrays to Seq[Int] before comparison
    // does not work when there are items other than Arrays containing Arrays
    ////////////////////////////////////////////////////////////////////////////////////////////

    print("Array(11, Array(21, 22)) sameElements Array(11, Array(21, 22)) : ")
    println(Array(11, Array(21, 22)) sameElements Array(11, Array(21, 22))) // false
    print("Array(11, Array(21, 22)).deep sameElements Array(11, Array(21, 22)).deep : ")
    println(Array(11, Array(21, 22)).deep sameElements Array(11, Array(21, 22)).deep) // true
    print("Array(11, Array(21), Seq(1, 2)).deep sameElements Array(11, Array(21), Seq(1, 2)).deep) : ")
    println(Array(11, Array(21), Seq(1, 2)).deep sameElements Array(11, Array(21), Seq(1, 2)).deep) // true
    print("Array(Seq(Array(1))).deep sameElements Array(Seq(Array(1))).deep) : ")
    println(Array(Seq(Array(1))).deep sameElements Array(Seq(Array(1))).deep) // false
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
    // z: Is the starting instance, example empty string
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

