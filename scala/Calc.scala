object Calc {
  def main(args: Array[String]): Unit = args match {
    case Array(s) => println(calculate(s))
    case _ => println("Usage postfix-expression")
  }

  object Num {
    def unapply(s: String): Option[Int] = try {
      Some(s.toInt)
    } catch {
      case _: Throwable => None
    }
  }

  sealed trait Operator {
    def apply(a: Int, b: Int): Int
  }

  object Operator {
    case object ADD extends Operator {
      def apply(a: Int, b: Int): Int = a + b
    }
    case object SUB extends Operator {
      def apply(a: Int, b: Int): Int = a - b
    }
    case object MUL extends Operator {
      def apply(a: Int, b: Int): Int = a * b
    }
    case object DIV extends Operator {
      def apply(a: Int, b: Int): Int = a / b
    }

    val map = Map( "+" -> ADD, "-" -> SUB, "*" -> MUL, "/" -> DIV)
    def unapply(s: String): Option[Operator] = map.get(s)
  }

  def calculate(s: String): Int = {
    // split string based on whitespace
    val tokens = s.split("\\s+")
    val stack = new collection.mutable.Stack[Int]()

    for (token <- tokens) token match {
      case Operator(op) => stack.push(op(stack.pop, stack.pop))
      case Num(n) => stack.push(n)
      case _ => throw new Exception("Unknown token : %s".format(token))
    }
    stack.pop
  }
}
