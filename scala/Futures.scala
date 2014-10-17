import java.util.concurrent.{Callable, Executors, ExecutorService}

// https://twitter.github.io/scala_school/concurrency.html

object Concurrency {
  def main(args: Array[String]) {
  }

  class MyResult(s: String)

  class MyCallable(s: String = "") extends Callable[MyResult] {
    def call(): MyResult = {
      println("MyCallable.call " + s + " " + Thread.currentThread.getName())
      new MyResult(s)
    }
  }

  def simple() = {
  }
}

