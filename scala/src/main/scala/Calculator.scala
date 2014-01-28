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
class Add extends Operator {
  def apply(lhs: Int, rhs: Int): Int = lhs + rhs
}
class Subtract extends Operator {
  def apply(lhs: Int, rhs: Int): Int = lhs - rhs
}
class Multiply extends Operator {
  def apply(lhs: Int, rhs: Int): Int = lhs * rhs
}
class Divide extends Operator {
  def apply(lhs: Int, rhs: Int): Int = lhs / rhs
}

object Operator {
  def unapply(token: String): Option[Operator] =
    token match {
      case "+" => Some(new Add)
      case "-" => Some(new Subtract)
      case "*" => Some(new Multiply)
      case "/" => Some(new Divide)
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
