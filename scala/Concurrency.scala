import java.util.concurrent._
import java.util.concurrent.atomic.AtomicInteger

// https://twitter.github.io/scala_school/concurrency.html

object Concurrency {
  def main(args: Array[String]) {
    //simple()
    //threadpool()
    blockingQueue()
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

  // http://tutorials.jenkov.com/java-util-concurrent/blockingqueue.html
  def blockingQueue() = {
    //arrayBlockingQueue
    simpleThreadPool
  }

  abstract class Producer(val queue: BlockingQueue[Int]) extends Runnable { val c = new AtomicInteger }
  class BlockingProducer(queue: BlockingQueue[Int]) extends Producer(queue) { 
    override def run() { queue.put(c.getAndIncrement) } }
  class ThrowingProducer(queue: BlockingQueue[Int]) extends Producer(queue) { 
    override def run() { queue.add(c.getAndIncrement) } }

  abstract class Consumer(val queue: BlockingQueue[Int]) extends Runnable
  class BlockingConsumer(queue: BlockingQueue[Int]) extends Consumer(queue) { 
    override def run() { queue.take() } }
  class ThrowingConsumer(queue: BlockingQueue[Int]) extends Consumer(queue) { 
    override def run() { queue.remove() } }

  def arrayBlockingQueue = {
    // Array queue with capacity 1
    val abq = new ArrayBlockingQueue[Int](1)
    // Linked structure
    //val lbq = new LinkedBlockingQueue[Int](1)

    abq.put(1)
    // abq.put(2) // will hang
    // abq.add(2) // will throw
    println(abq.offer(2)) // prints false

/*
    val producer = new BlockingProducer(abq)
    val consumer = new BlockingConsumer(abq)
*/
  }

  def simpleThreadPool = {
    val jobQueueSize = 0
    val jobQueue: BlockingQueue[Runnable] = if (jobQueueSize == 0) new SynchronousQueue[Runnable]()
      else new LinkedBlockingQueue[Runnable](jobQueueSize)
    // 1 thread
    val pool: ExecutorService = new ThreadPoolExecutor(1, 1, Long.MaxValue, TimeUnit.MILLISECONDS, jobQueue)

    try {
      1 to 2 foreach { i => pool.execute(new MyRunnable(i.toString)) }
    } catch {
      case e: RejectedExecutionException => println("Rejected second job, as expected")
    } finally {
      pool.shutdown()
    }
  }
}

