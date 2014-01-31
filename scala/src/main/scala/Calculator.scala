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

sealed abstract class Expression
case class NumberExpression(value: Int) extends Expression
case class OperationExpression(
  op: Operator,
  lhs: Expression,
  rhs: Expression
) extends Expression

object Calculator {
  // Create the tree of expressoin
  def parse(expression: String): Expression = {
    val tokens: Array[String] = expression.split(" ")
    val stack = mutable.Stack.empty[Expression]
    for (token <- tokens) {
      token match {
        case Number(n) =>
          stack.push(NumberExpression(n))
        case Operator(op) =>
          val rhs = stack.pop() // rhs is top of stack
          val lhs = stack.pop()
          stack.push(OperationExpression(op, lhs, rhs))
        case _ =>
          throw new IllegalArgumentException("garbage token: " + token)
      }
    }
    stack.pop()
  }

  // reccursively evaluate the expression tree
  def calculate(expression: Expression) : Int =
    expression match {
      case NumberExpression(value) => value
      case OperationExpression(op, lhs, rhs) =>
        op(calculate(lhs), calculate(rhs))
    }

  // usage : Calculator.main(Array("1 1 +"))
  def main(args: Array[String]): Unit =
    println(calculate(parse(args(0))))
}
