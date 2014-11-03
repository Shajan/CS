object Shutdown {
  def main(args: Array[String]) {
    sys.addShutdownHook(println("a")) // works with ^C, but not with kill -9

    val t1:Thread = new Thread(new Runnable {
      def run() {
        Thread.sleep(100)
        sys.addShutdownHook(println("b"))
      }
    })

    val t2:Thread = new Thread(new Runnable {
      def run() {
        Thread.sleep(200)
      }
    })

    sys.addShutdownHook(println("c"))

    t1.start
    t2.start

    t1.join
    t2.join
  }
}
