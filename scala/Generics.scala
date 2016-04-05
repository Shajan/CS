object Generics {
  def main(args: Array[String]) {
    new X[A].x(new ABC)
    new X[AB].x(new ABC)
    new X[ABC].x(new ABC)

    // A is not an AB, so 'new Y[A]' not allowed
    new Y[AB].y(new ABC)
    new Y[ABC].y(new ABC)

    new P[A].x(new A)
    new P[A].p(new A)
  }

  //def thisWillNotCompile(c: C) = { c.g(c.f()) }
  def thisWillCompile[T](c: C[T]) = { c.g(c.f()) }
}

class A { }
class AB extends A { }
class ABC extends AB { }

class X[T <: A] { def x(t: T) = { } }
class Y[T <: AB] { def y(t: T) = { } }

class P[T <: A] extends X[T] { def p(t: T) = { } }

class C[T](t: T) {
  def f(): T = t
  def g(t: T) = Unit
  def test(a: A[T]) = a.g(a.f())
}
