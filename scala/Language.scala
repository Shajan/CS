import scala.annotation.tailrec

object Language {
  def main(args: Array[String]) {
    //loops
    traits
    //abstractmember
  }

  def loops = {
    var i=3 
    loop {
      println("i = " + i)
      i -= 1
    } unless (i == 0)  // <-- same as loop({..}).unless({..})

    def loop(body: => Unit): LoopUnlessCond = {
      new LoopUnlessCond(body)
    }
    class LoopUnlessCond(body: => Unit) {
      @tailrec final def unless(cond: => Boolean) {
        body
        if (!cond) unless(cond)
      }
    }
  }

  class Ball {
    def properties(): List[String] = List()
    override def toString() = "It's a" +
      properties.mkString(" ", ", ", " ") +
      "ball"
  }

  trait Red extends Ball {
    override def properties() = super.properties ::: List("red")
  }

  trait Shiny extends Ball {
    override def properties() = super.properties ::: List("shiny")
  }

  def traits = {
    println(new Ball with Shiny with Red) // It's a shiny, red ball
    println(new Ball with Red with Shiny) // It's a red, shiny ball
  }

  def abstractmember = {
    abstract class A { def f:Unit }
    class B extends A { override def f:Unit = { println("B.f") } }
    class C extends A { override def f:Unit = { println("C.f") } }
    abstract class Base { val a:A }
    class Derived extends Base { val a:A = new B }
    val d = new Derived
    d.a.f
  }
}
