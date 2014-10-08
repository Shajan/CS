import scala.concurrent._
import scala.concurrent.duration._
import ExecutionContext.Implicits.global

object Sample {
  def main(args: Array[String]) {
    //ClassSample.Run(args)
    //ClosureSample.Run(args)
    //FunctionSample.Run(args)
    FutureSample.Run(args)
    //FlatMapSample.Run(args)
    //OptionSample.Run(args)
    //ControlSample.Run(args)
    //ListSample.Run(args)
    //ArraySample.Run(args)
    //Precondition.Run(args)
    //CmdLineSample.Run(args)
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

object ClassSample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("Class")
    applyMethod()
  }
  def applyMethod() = {
    object A {
      def apply(i: Int) = new A(i)
    }
    class A(i: Int)
    val a = A(1)
  }
}

object ClosureSample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("Closure")
    basic()
    advanced()
  }
  def basic() = {
    var a = 10
    def foo(b: Int) = {
      if (a != b) 
        println("Error! a:"+a+",b:"+b) 
    }
    foo(10)
    a = 11
    foo(11)
  }
  def advanced() = {
    // Returns a function instance that increments by x
    // 'Remembers' x for later use
    def createIncrementer(x: Int) = (y: Int) => x + y

    // Create a function that increments by 1
    val incr1 = createIncrementer(1)

    // Create a function that increments by 10
    val incr10 = createIncrementer(10)

    if (incr1(10) != 11)
      println("Error! expected 11 :" + incr1(10))

    if (incr10(50) != 60)
      println("Error! expected 60 :" + incr10(50))
  }
}

object FunctionSample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("Function")
    PartiallyApplied()
  }
  def PartiallyApplied() = {
    val sum = (_: Int) + (_: Int) + (_: Int)
    val add = sum
    val add_three = sum(1, _: Int, 2)

    println("sum(1,2,3):" + sum(1,2,3))
    println("add(1,2,3):" + add(1,2,3))
    println("add_three = sum(1, _: Int, 2)")
    println("add_three(5):" + add_three(5))
  }
}

object FutureSample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("Future")
    Basic()
    //Timing()
  }

  def Basic() = {
    Seperator.line("Future:Basic")
    val f1 = future {
      Thread.sleep(100)
      "f1"
    }
    val r1 = Await.result(f1, 1 second)
    println(r1)

    val f2 = future {
      println("f2")
    }

    val f3 = future {
      Thread.sleep(100)
      throw new Exception("Future exception")
      "f3"
    }

    try {
      val r3 = Await.result(f3, 1 second)
      println("We never get here! " + r3)
    } catch {
      case e: Throwable => println(e)
    }

    val input = List(1,2,3,4,5)
    val tasks: Seq[Future[String]] = input.map { i =>
      future {
        println(i + ":executing")
        if (i%2 == 0) Thread.sleep(100)
        println(i + ":done-executing")
        i.toString + ":result"
      }
    }

    // Combine Sequence of Future Strings to Future of Sequence of Strings
    val aggregated: Future[Seq[String]] = Future.sequence(tasks)
    val result: Seq[String] = Await.result(aggregated, 1 second)
    result.foreach(println(_))
  }

  def Timing() = {
    println("1.Defining f")
    val f = future {
      println("....Start f....")
      Thread.sleep(100)
      println("....End f....")
      "f"
    }
    println("2.Defining f.onSuccess")
    // Async call
    f.onSuccess { case s => println(".... onSuccess: " + s) }

    println("3.Defining f.foreach")
    // see http://stackoverflow.com/questions/2173373/scala-foreach-strange-behaviour
    // Process value
    f.foreach(s => (println(".... f.foreach: " + s )))

    // Create a new feature by applying a function to success results
    println("4.Defining f.map")
    val fMap = f.map(x => {
      println("....In f.Map....")
      "(m)" + _
    })

    // Create a new feature by applying a function to success results
    // return new feature
    println("5.Defining f.flatMap")
    val fFlatMap = f.flatMap(x => {
      future {
        println("....Start f.FlatMap....")
        Thread.sleep(100)
        println("....End f.FlatMap....")
        "(fm)" + _
      }
    })

    // Create a new feature by filtering value
    println("6.Defining f.filter")
    val fFilter = f.filter(x => {
      println("....In f.filter....")
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
/* 
	1.Defining f
	2.Defining f.onSuccess
	....Start f....
	3.Defining f.foreach
	4.Defining f.map
	5.Defining f.flatMap
	6.Defining f.filter
	7.Start Wait all
	....End f....
	....In f.filter....
	....In f.Map....
	.... onSuccess: f
	....Start f.FlatMap....
	.... f.foreach : f
	....End f.FlatMap....
	Last.End Wait All
*/
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
    println("def even(x:Int) = if ((x%2)==0) Some(x) else None")

    val lm = l.map(even _)
    println("l.map(even _) : " + lm.mkString(","))
    // None,Some(2),None,Some(4),None

    val lfm = l.flatMap(even _)
    println("l.flatMap(even _) : " + lfm.mkString(",")) // 2,4

    val ll = l.map(x=>List(x-1, x, x+1))
    println("l.map(x=>List(x-1, x, x+1)): " + ll.mkString(","))
    // List(0, 1, 2),List(1, 2, 3),List(2, 3, 4),List(3, 4, 5),List(4, 5, 6)

    val fm = l.flatMap(x=>List(x-1, x, x+1))
    println("l.flatMap(x=>List(x-1, x, x+1)): " + fm.mkString(","))
    // 0,1,2,1,2,3,2,3,4,3,4,5,4,5,6

    val ml = m.toList
    println("m.toList: " + ml.mkString(",")) // (1,3),(2,6),(3,9)
    println("m.mapValues(_*2): " + m.mapValues(_*2).mkString(",")) // 1 -> 6,2 -> 12,3 -> 18
    println("m.mapValues {even _}: " + m.mapValues {even _} )
    // Map(1 -> None, 2 -> Some(6), 3 -> None)

    def evenKey(k:Int, v:Int) = if ((k%2)==0) Some(k->v) else None
    println("def evenKey(k:Int, v:Int) = if ((k%2)==0) Some(k->v) else None")

    // m.flatMap { (k,v) => f(k,v) } Syntax not supported in scala
    // _1, _2 first, second fileds of a touple
    println("m.flatMap { e => evenKey(e._1, e._2) }: " + m.flatMap { e => evenKey(e._1,e._2) })
    // Map(2 -> 6)

    // Equivalent to above
    println("m.flatMap { (k,v) => evenKey(k,v) }: " + m.flatMap { case (k,v) => evenKey(k,v) })
    // Map(2 -> 6)

    // Using filter
    println("m.filter(e => even(e._2)) != None: " +  m.filter(e => even(e._2) != None))
    println("m.filter{ case (k,v) => even(v) != None }: " + m.filter{ case (k,v) => even(v) != None })
    println("m.filter{ case (k,v) => even(v).isDefined }: " + m.filter{ case (k,v) => even(v).isDefined })
    // All three above gives same result Map(1 -> 2, 2 -> 4, 3 -> 6)
  }
}

object OptionSample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("Option")
    // lCase.get returns Option { Some[String] or None }
    val lCase = Map("A" -> "a", "B" -> "b")
    basic(lCase)
    controlled(lCase)
    default(lCase)
  }
  def basic(lCase: Map[String, String]) {
    val a = lCase.get("A")
    val x = lCase.get("X")
    println("lCase of A isEmpty? : " + a.isEmpty)
    println("lCase of A : " + a)  // Some(a)
    println("lCase of X isEmpty? : " + x.isEmpty)
    println("lCase of X : " + x)  // Nothing
    print("a.foreach( println _ ) : ")
    a.foreach( println _ )        // 'a'
    print("x.foreach( println _ ) : ")
    x.foreach( println _ )        // println won't be executed
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
    //exceptions()
    for_loops()
    //for_yield()
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

  def for_loops() = {
    Seperator.line("for loops")
/*
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
*/

    // Group by n adjacent elements in a list
    val l = List(1 to 10)
    println("l:", l.mkString(","))
//@TODO
  }

  def for_yield() = {
    Seperator.line("yield")

    val files = (new java.io.File(".")).listFiles
    val scalaFiles =
      for (file <- files if file.getName.endsWith(".scala"))
        yield file
    println(scalaFiles.mkString(", "))

    val seq = for {
      i <- 1 to 5 if (i % 2 == 0)
      j <- 6 to 10 if (j % 2 == 1)
      k = i*j
    } yield(i, j, k)

    println(seq.map(x => x._1.toString + " x " + x._2.toString + " = " + x._3.toString).mkString(", "))
    // 2 x 7 = 14, 2 x 9 = 18, 4 x 7 = 28, 4 x 9 = 36
  }
}

object ListSample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("List")
    //basic()
    //advanced()
    listBuffer()
  }
  def basic() = {
    // Group by at most n elements
    println(List(1,2,3,4,5,"six").grouped(4).toList) // List(List(1, 2, 3, 4), List(5, six))

    // Create pairs with list item and index
    println(List("a", "b", "c").zipWithIndex) // List((a,0), (b,1), (c,2))

    // Transpose row <--> column
    val abc123 = List(
                   List('a','b','c'),
                   List('1','2','3'))
    val a1b2c3 = abc123.transpose 
    println(abc123 + " -- transpose --> " + a1b2c3)
    // List(List(a, b, c), List(1, 2, 3)) -- transpose --> List(List(a, 1), List(b, 2), List(c, 3))
  }
  def advanced() = {
    // Reference http://aperiodic.net/phil/scala/s-99/
    def groupRuns[A](l: List[A]): List[List[A]] = {
      if (l.isEmpty) List(List())
      else {
        // span(p: (A) ⇒ Boolean): (List[A], List[A])
        // Splits list into a prefix/suffix pair according to a predicate.
        val (grouped, next) = l span { _ == l.head }
        if (next == Nil) List(grouped)
        else if (grouped == Nil) List(next)
        else grouped :: groupRuns(next)
      }
    }
    println(groupRuns(List(1, 1, 1, 2, 2, 5, 7, 7)))
    // List(List(1, 1, 1), List(2, 2), List(5), List(7, 7))

    def countRuns[A](l: List[A]): List[(Int, A)] = {
      if (l.isEmpty) Nil
      else {
        val (simillar, next) = l span { _ == l.head }
        (simillar.length, simillar.head) :: countRuns(next)
      }
    }
    println(countRuns(List(1, 1, 1, 2, 2, 5, 7, 7)))
    // List((3,1), (2,2), (1,5), (2,7))

    val minmax = List(10, 5, 9, 4, 7, 20, 3, 7, 16).
      foldLeft((1000, -1000))((t, a) =>
        ((if (a < t._1) a else t._1), (if (a > t._2) a else t._2)))
    println("min:" + minmax._1 + " ,max:" + minmax._2) // min:3 ,max:20
  }
  def listBuffer() = {
    import scala.collection.mutable.ListBuffer
    Seperator.line("ListBuffer")
    val l1 = ListBuffer(1, 2, 3)
    val l2 = ListBuffer(4, 5, 6)
    val l3 = l1 ++ l2

    println(l3) // ListBuffer(1, 2, 3, 4, 5, 6)
    println(l1) // ListBuffer(1, 2, 3)

    l3 += 7
    println(l3) // ListBuffer(1, 2, 3, 4, 5, 6, 7)

    val l5 = l3 :+ 8
    println(l3) // ListBuffer(1, 2, 3, 4, 5, 6, 7)
    println(l5) // ListBuffer(1, 2, 3, 4, 5, 6, 7, 8)

    val l4 = l3.dropRight(1)
    println(l3) // ListBuffer(1, 2, 3, 4, 5, 6, 7)
    println(l4) // ListBuffer(1, 2, 3, 4, 5, 6)

  }
}

object ArraySample extends Executable {
  def Run(args: Array[String]) = {
    Seperator.line("Array")
    val a = Array(10, 20, 30, 40, 50)
    println(a.mkString(","))
    basic(a)
    arrayType()
    advanced(a)
  }

  def basic(a: Array[Int]) = {
    println("foreach(x => println(x))")
    a.foreach(x => println(x))

    println("foreach(println _)")
    a.foreach(println _)

    println("foreach(println)")
    a.foreach(println)

    println("for (x <- a) println(x)")
    for (x <- a) println(x)

    println("a.filter(_>25)")
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
    // convert array of int to array of string
    val c = a.map( _.toString )

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
