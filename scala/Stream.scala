object Stream {
  def main(args: Array[String]): Unit = {
    streamIterator()
    println("------------------------")
    streamSeq()
  }

  def streamIterator(): Unit = {
    val c:C = new C
    val iter = c.getIterator
    val x = iter.takeWhile(_ < 5) // Limit to first 4 items

    // Nothing is materialized till this x.forach is called
    x.foreach(i => println("iterated " + i))
  }
    
  def streamSeq(): Unit = {
    val c:C = new C
    val iter = c.getIterator
    val x = iter.takeWhile(_ < 5) // Limit to first 4 items

    // Nothing is materialized till this x.forach is called
    val s = x.toSeq

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
