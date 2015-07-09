import com.twitter.util._

object TwitterFuture {
  def main(args: Array[String]) {
    basic()
    map()
    flat_map()
    value_on_error()
    future_on_error()
    concurrent_failany()
    concurrent()
    callback()
  }

  def basic() = {
    val f1 = Future.value(1)
    val f2 = Future { 2 }
    val f3 = Future { new Exception("3") }
  }

  // Sequential Composition: map  A = B
  def map() = {
    val fp = FuturePool.unboundedPool { println("fp.start"); Thread.sleep(10000); println("fp.end"); 10 }

    // f is Future, statement inside map is executed synchronously after fp is done
    // don't block on map, use flatmap and return a future if you have to block
    val f = fp map { i => println("fp.map"); 1 + 3 } // "fp.map" prints after "fp.end"

    println(fp)
    println(f)
  }

  // Sequential Composition: flatMap  A = Future[B]
  def flat_map() = {
    val fp = FuturePool.unboundedPool {
      println("fp.start"); Thread.sleep(5000); println("fp.end"); 10 }
    val f = fp flatMap { 
      i => Future { println("flatMap.start"); Thread.sleep(5000); println("flatMap.end"); i + 3 }
    }
    println(fp)
    println(f)
  }

  // Set a value on excepiton
  def value_on_error() = {
    val f1 = Future.value(1)
    val f2 = f1 handle { case e:Exception => -1 } 
    println(f2) // Future 1

    val f3 = Future.exception(new Exception)
    val f4 = f3 handle { case e:Exception => -1 } 
    println(f4) // Future -1
  }

  // Set a Future on exception
  def future_on_error() = {
    val f1 = Future.value(1)
    val f2 = f1 rescue { case e:Exception => Future.value(2) } 
    println(f2) // Future 1

    val f3 = Future.exception(new Exception)
    val f4 = f3 handle { case e:Exception => Future.value(3) }  // This could be a retry
    println(f4) // Future 3 
  }

  def concurrent_failany() = {
    // Collect Seq(Future) --> Future[Seq]
    // Fail if any one fails

    val s:Seq[Future[Int]] = Seq(Future.value(1), Future.value(2), Future.value(3))
    val f:Future[Seq[Int]] = Future.collect(s)

    f map { s => println(s.mkString(",")) } // 1,2,3

    val s1:Seq[Future[Int]] = Seq(Future.value(1), Future.exception(new Exception), Future.value(3))
    val f1:Future[Seq[Int]] = Future.collect(s1)

    f1 map { s => println(s.mkString(",")) } // does not print anything

    // join returns a future of touple
    // Fail if any one fails
    val f2:Future[(Int,String,Int)] = Future.join(Future.value(1), Future.value("X"), Future.value(3))

    f2 map { s => println(s) } // (1,X,3)

  }

  // get the first one that failed or completed
  def concurrent() = {
    def sleep(sec: Int): Future[Int] = FuturePool.unboundedPool {
      println("start %d".format(sec))
      Thread.sleep(sec * 1000)
      if ((sec % 2) ==  1) {
        println("exception %d".format(sec))
        throw new Exception("Fail %d".format(sec))
      }
      println("end %d".format(sec))
      sec * 1000
    }

    val f:Seq[Future[Int]] = Seq(10,5,7,1,9).map(sleep(_))

    // Signature select[A](fs: Seq[A]): Future[(Try[A], Seq[Future[A]])]
    // note syntax: case(x,y) for tuple can't use (x,y) => need case
    val result = Future.select(f) map { case(tryresult, remaining) =>
      remaining.map(_.raise(new FutureCancelledException)) // This line appears to not work.
      tryresult() // apply the Try will yield the result or an exception 
    }

    println("Result : ".format(result)) // the first failed one.
  }

  // callback way of handling future, 
  // useful only for side-effects/logging, no returns
  // prefer map, flatmap etc.
  def callback() = {
    val f: Future[Int] = FuturePool.unboundedPool { Thread.sleep(1000); 1 }

    f onSuccess { res: Int =>
      println("f:Success " + res) // prints this
    } onFailure { ex: Throwable =>
      println("f:Failure " + ex)
    } ensure {
      println("f:Ensure") // prints this
    }

    val f1: Future[Int] = FuturePool.unboundedPool { throw new Exception }

    f1 onSuccess { res: Int =>
      println("f:Success " + res)
    } onFailure { ex: Throwable =>
      println("f:Failure " + ex) // prints this
    } ensure {
      println("f:Ensure") // prints this
    }
  }

  def timeout() = {
    import com.twitter.conversions.time._
    implicit val timer = new JavaTimer

    val f: Future[Int] = FuturePool.unboundedPool {
      println("start")
      Thread.sleep(10000)
      println("end huh?")
      1
    }

    // Within returns a new feature, does not cancell the original
    val f1 = f.within(2.seconds) onSuccess { res: Int =>
      println("f:Success " + res)
    } onFailure {
      case ex: TimeoutException =>
        println("f:Timeout " + ex) // prints this
        // Now cancel the original future
        f.raise(new Exception) // Not guaranteed to work
      case ex: Throwable => 
        println("f:Exception " + ex) // prints this
    }
  }
}
