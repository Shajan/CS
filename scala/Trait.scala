// Knows how to count
trait Counter {
  private[this] var c: Int = 0
  def inc = { c = c + 1 }
  def count: Int = { c }
}

// Knows how to count
trait Logger {
  def log(s: String) = println(s)
}

abstract class Compute {
  def apply(i: Int): Int
}

class Square extends Compute {
  override def apply(i: Int) = i*i
}

class Negate extends Compute {
  override def apply(i: Int) = -i
}

// Can't extend two things, Counter is conveniently a trait
trait CountingFilter extends Compute with Counter {
  // Need to be abstract override
  abstract override def apply(i: Int): Int = {
    inc
    super.apply(i)
  }
}

trait LoggingFilter extends Compute with Logger {
  abstract override def apply(i: Int): Int = {
    log("before : %d".format(i))
    val r = super.apply(i)
    log("after : %d".format(r))
    r
  }
}

object Trait {
  def main(args: Array[String]): Unit = {
    // method apply is chained: LoggingFilter.apply then CountingFilter.apply
    val s = new Square with CountingFilter with LoggingFilter
    s(2)
    s(3)
    println("Count : %d".format(s.count))
    val n = new Negate with CountingFilter with LoggingFilter
    n(2)
    n(3)
    n(5)
    println("Count : %d".format(n.count))
  }
}
