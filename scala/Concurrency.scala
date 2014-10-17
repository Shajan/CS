import java.util.concurrent.{Callable, Executors, ExecutorService}

// https://twitter.github.io/scala_school/concurrency.html

object Concurrency {
  def main(args: Array[String]) {
    simple()
    threadpool()
  }

  class MyRunnable(s: String = "") extends Runnable {
    def run(): Unit = {
      Thread.sleep(10) // milliseconds
      println("MyRunnable.run " + s + " " + Thread.currentThread.getName())
    }
  }

/*
  class MyResult
  class MyCallable() extends Callable[MyResult] {
    def call(): MyResult = {
      println("MyCallable.call")
      new MyResult()
    }
  }
*/
  def simple() = {
    val myRunnable = new MyRunnable()
    val myThread = new Thread(myRunnable)
    println("Starting thread")
    myThread.start
    println("Done Starting")
    // Output:
    //     Starting thread
    //     Done Starting
    //     MyRunnable.run Thread-0
  }

  def threadpool() = {
    val pool: ExecutorService = Executors.newFixedThreadPool(2) // 2 threads, unbounded queue
    try {
      1 to 5 foreach { i => pool.execute(new MyRunnable(i.toString)) }
    } finally {
      pool.shutdown()
    }
    // Output:
    //	MyRunnable.run 1 pool-1-thread-1
    //	MyRunnable.run 2 pool-1-thread-2
    //	MyRunnable.run 4 pool-1-thread-2
    //	MyRunnable.run 3 pool-1-thread-1
    //	MyRunnable.run 5 pool-1-thread-2
  }
}

