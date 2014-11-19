object Function {
  def main(args: Array[String]) {
    foo { () => 1 }
    bar { 1 }
  }

  def foo[A](f: () => A): A = f()
  def bar[A](f: => A): A = f
}

