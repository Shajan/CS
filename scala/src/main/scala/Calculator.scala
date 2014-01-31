import collection.mutable

sealed trait Expression
case class NumberExpression(
  value: Int
) extends Expression
case class OperationExpression(
  //op: Operator,
  op: (Int, Int) => Int,
  lhs: Expression,
  rhs: Expression
) extends Expression

object Number {
  def unapply(token: String): Option[Int] =
    try {
      Some(token.toInt)
    } catch {
      case _: NumberFormatException => None
    }
}

/*
abstract class Operator          { def apply(lhs: Int, rhs: Int): Int }
object Add extends Operator      { def apply(lhs: Int, rhs: Int): Int = lhs + rhs }
object Subtract extends Operator { def apply(lhs: Int, rhs: Int): Int = lhs - rhs }
object Multiply extends Operator { def apply(lhs: Int, rhs: Int): Int = lhs * rhs }
object Divide extends Operator   { def apply(lhs: Int, rhs: Int): Int = lhs / rhs }

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
*/

object Operator {
  val operators = Map[String, (Int, Int) => Int](
      "+" -> { _ + _ },
      "-" -> { _ - _ },
      "*" -> { _ * _ },
      "/" -> { _ / _ }
    )

  def unapply(token: String): Option[(Int, Int) => Int] =
    operators.get(token)
}

object Calculator {
  // Using foldLeft
  def parse(expression: String): Expression = {
    def go(stack: List[Expression], token: String): List[Expression] =
      (stack, token) match {
        case (_, Number(n)) =>
          NumberExpression(n) :: stack
        case (rhs :: lhs :: tail, Operator(op)) =>
          OperationExpression(op, lhs, rhs) :: tail
        case _ =>
          throw new IllegalArgumentException("Not enough tokens")
      }
    val tokens = expression.split(" ")
    val stack = tokens.foldLeft(List.empty[Expression])(go)
    stack.head
  }

/* Procedural way with a for loop
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
*/

  // reccursively evaluate the expression tree
  def calculate(expression: Expression): Int =
    expression match {
      case NumberExpression(value) =>
        value
      case OperationExpression(op, lhs, rhs) =>
        op(calculate(lhs), calculate(rhs))
    }

  // usage : Calculator.main(Array("1 1 +"))
  def main(args: Array[String]): Unit =
    println(calculate(parse(args(0))))
}
