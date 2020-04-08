object Stream {
  def main(args: Array[String]): Unit = {
    streamIterator()
    println("------------------------")
    streamSeq()
  }

  def get(n: Int): Iterator[Int] = {
    val c:C = new C
    val iter = c.getIterator
    iter.takeWhile(_ < n) // Limit to first n items
  }

  def streamIterator(): Unit = {
    val iter: Iterator[Int] = get(5)
    // Nothing is materialized till this x.forach is called
    iter.foreach(i => println("iterated " + i))
  }
    
  def streamSeq(): Unit = {
    val iter: Iterator[Int] = get(5)
    val s = x.toSeq
    // Nothing is materialized till this x.forach is called
    s.foreach(i => println("iterated " + i))
  }
}

class C {
  var c: Int = 0
  def nxt(): Int = {
    c += 1;
    println("Materialized : " + c)
    c
  }

  def getIterator: Iterator[Int] = Iterator.continually { this.nxt }
}
