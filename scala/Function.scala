object Function {
  def main(args: Array[String]) {
    simple
    composition
  }

  def foo[A](f: () => A): A = f()
  def bar[A](f: => A): A = f

  def simple = {
    foo { () => 1 }
    bar { 1 }
  }

  def f(s: String) = "f(" + s + ")"
  def g(s: String) = "g(" + s + ")"

  def composition = {
    val fComposeG = f _ compose g _
    val fAndThenG = f _ andThen g _

    println(fComposeG("a"))  // f(g(a)) 
    println(fAndThenG("a"))  // g(f(a)) 
  }
}

