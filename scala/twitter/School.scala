import com.twitter.util._ // requires twitter util jar

object School {
  def main(args: Array[String]) {
    val futurePool = new Pool(2) // two threads
    //setValue()
    //simple(futurePool.pool)
    //callbacks(futurePool.pool)
    //map()
    timeout()
    futurePool.shutdown() // Without this the process won't exit
  }

  def getInt(): Int = 1
  def getException(): Int = throw new Exception("getException")
  def get(): Int = {
    import scala.util.Random._
    if (nextBoolean()) getInt else getException
  }

  def delayedInt(seconds: Int): Int = {
    Thread.sleep(seconds * 1000)
    getInt()
  }

  def delayedException(seconds: Int): Int = {
    Thread.sleep(seconds * 1000)
    getException()
  }

  def delayed(seconds: Int): Int = {
    Thread.sleep(seconds * 1000)
    get()
  }

  // These are all blocking operations on the current thread
  def setValue() = {
    val fInt: Future[Int] = Future.value(1)
    println("fInt %s".format(fInt)) // Return(1)

    val fException: Future[Int] = Future.exception(new Exception("inline"))
    println("fException %s".format(fException)) // Throw(java.lang.Exception: inline) 

    val fEvalInt: Future[Int] = Future(getInt * 2)
    println("fEvalInt %s".format(fEvalInt)) // Return(2)

    val fEvalException: Future[Int] = Future(getException * 2)
    println("fEvalException %s".format(fEvalException)) // Throw(java.lang.Exception: getException) 

    val fSlowEval: Future[Int] = Future(delayedInt(3))
    println("fSlowEval %s".format(fSlowEval)) // Return(1), but after 3 seconds
  }

  // Helper class to manage pools
  // Alternate to just use FuturePool.unboundedPool
  class Pool(threads: Int) {
    // Future pool has a set of future jobs and an executor with a few threads to execute
    import java.util.concurrent.Executors
    import com.twitter.util.FuturePool
    private val executor = Executors.newFixedThreadPool(threads) // number of threads

    val pool = FuturePool(executor)
    def shutdown() = executor.shutdown() // return now, but will let in-progress tasks complete
    def shutdownNow() = executor.shutdownNow() // cancell tasks before returning
  }

  def simple(pool: FuturePool) = {
    // Executes in different thread, returns future in this thread
    val fRun: Future[Int] = pool { delayedInt(1) }
    println("fRun %s".format(fRun)) // Promise - subtype of Future

    delayedInt(2) // just to add delay, to show the results
    println("fRun %s".format(fRun)) // Return(1)

    //futurePool.shutdown() // Without this the program won't exit
  }

  def callbacks(pool: FuturePool) = {
    val fInt: Future[Int] = pool { delayedInt(1) }
    val fException: Future[Int] = pool { delayedException(1) }

    println("fInt %s".format(fInt)) // Promise - subtype of Future
    println("fException %s".format(fException)) // Promise - subtype of Future

    fInt.onSuccess {
      case i: Int => println("onSuccess : %d".format(i)) // 1
    }.onFailure {
      case e: Throwable => println("What? : %s".format(e))
    }.ensure {
      println("ensure") // ensure
    }

    fException.onSuccess {
      case i: Int => println("What? : %d".format(i))
    }.onFailure {
      case e: Throwable => println("onFailure : %s".format(e)) // Exception
    }.ensure {
      println("ensure") // ensure
    }
  }

  def map() = {
    val futurePool: Pool = new Pool(2)
    val fInt: Future[Int] = futurePool.pool { delayedInt(1) }
    println("fInt %s".format(fInt)) // Promise

    // Executes after fInt future is done in this thread
    // Returns a future
    val fMap = fInt.map { i => i + 1 }
    println("fMap %s".format(fMap)) // Promise

    // Use flatMap if computation need to happen in another future
    val fFlatMap = fInt.flatMap { i => futurePool.pool { i + 1 } }
    println("fFlatMap %s".format(fFlatMap)) // Promise

    fMap.onSuccess { case i: Int => println("fMap onSuccess : %d".format(i)) } // 2
    fFlatMap.onSuccess { case i: Int => println("fFlatMap onSuccess : %d".format(i)) } // 2
             .onFailure { case e: Throwable  => println("fFlatMap error : %s".format(e)) }

    fFlatMap.ensure {
      // If shutdown is called before the second future is done
      // fFlatMap will not get scheduled and get an exception
      futurePool.shutdown
    }
  }

  // requires finable and netty jars
  def timeout() = {
    import com.twitter.conversions.time._ // 1.seconds
    import com.twitter.finagle.util.DefaultTimer
    implicit val timer = DefaultTimer.twitter

    val futurePool: Pool = new Pool(2)
    val fSlow: Future[Int] = futurePool.pool { delayedInt(3) } // delay 10 seconds
    val fTimed: Future[Int] = fSlow.within(1.seconds) // A seperate future

    fTimed.onFailure {
      case _:TimeoutException =>
        println("timeout")
        fSlow.raise(new FutureCancelledException) // advisory, not guaranteed
      case _: Exception => println("exception from original future")
    }

    fSlow.onSuccess {
      i: Int => println("fSlow: OnSuccess %d".format(i)) // OnSuccess - cancell does not really work
    }.onFailure {
      e: Throwable => println("fSlow: OnFailure %s".format(e))
    }.ensure {
      futurePool.shutdown
    }
  }
}
