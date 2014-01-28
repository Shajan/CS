import collection.mutable

object Number {
  def unapply(token: String): Option[Int] =
    try {
      Some(token.toInt)
    } catch {
      case _: NumberFormatException => None
    }
}

abstract class Operator {
  def apply(lhs: Int, rhs: Int): Int
}
object Add extends Operator {
  def apply(lhs: Int, rhs: Int): Int = lhs + rhs
}
object Subtract extends Operator {
  def apply(lhs: Int, rhs: Int): Int = lhs - rhs
}
object Multiply extends Operator {
  def apply(lhs: Int, rhs: Int): Int = lhs * rhs
}
object Divide extends Operator {
  def apply(lhs: Int, rhs: Int): Int = lhs / rhs
}

object Operator {
  def unapply(token: String): Option[Operator] =
    token match {
      case "+" => Some(Add)
      case "-" => Some(Subtract)
      case "*" => Some(Multiply)
      case "/" => Some(Divide)
      case _   => None
    }
}

object Calculator {
  def calculate(expression: String): Int = {
    val tokens: Array[String] = expression.split(" ")

    val stack = mutable.Stack.empty[Int]

    for (token <- tokens) {
      token match {
        case Number(n) =>
          stack.push(n)
        case Operator(op) =>
          val rhs = stack.pop()
          val lhs = stack.pop()
          stack.push(op(lhs, rhs))
        case _ =>
          throw new IllegalArgumentException("garbage token: " + token)
      }
    }

    stack.pop()
  }

  def main(args: Array[String]): Unit =
    println(calculate(args(0)))

}
