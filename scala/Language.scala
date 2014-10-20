object Language {
  def main(args: Array[String]) {
    loops()
  }

  def loops() = {
    var i=3 
    loop {
      println("i = " + i)
      i -= 1
    } unless (i == 0)  // <-- same as loop({..}).unless({..})

    def loop(body: => Unit): LoopUnlessCond = {
      new LoopUnlessCond(body)
    }
    class LoopUnlessCond(body: => Unit) {
      def unless(cond: => Boolean) {
        body
        if (!cond) unless(cond)
      }
    }
  }
}
