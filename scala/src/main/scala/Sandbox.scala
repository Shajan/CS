import collection.mutable.Buffer

object Sandbox {

  ////////////////////////////////////////////////
  // Option can be Some or None
  //
  def safeStringToInt(s: String): Option[Int] = {
    try {
      Some(s.toInt)
    } catch {
      case _: NumberFormatException => None 
    }
  }

  ////////////////////////////////////////////////
  // return 0 on error
  //
  def safeStringToIntBad(s: String): Int = {
    try {
      s.toInt
    } catch {
      case _: NumberFormatException => 0
    }
  }

  def daysInMonth(n: Int): Int = 
    n match {
      case 1 | 3 | 5 | 7 | 8 | 10 | 12 => 31
      case 4 | 6 | 9 | 11 => 30
      case 2 => 28
      // default will get a match error exception
    }

  def monthName(n: Int): String = 
    n match {
      case 1 => "Jan"
      case 2 => "Feb"
      case 3 => "Mar"
      case _ => "Unknown"
    }

  def whatIsThis(x: Any): String = 
    x match {
      case 1 => "the number one"
      case n: Int => "some other int " + n // n is Int
      case List(a, b) =>
        "list of two elements " + a + b
      case List(a, _*) =>
        "list of one or more elements " + a
      case _ => "something else"
    }

  // Usage
  //  happyOrSad(Some("abc"))
  //  happyOrSad(Nothing)
  def happyOrSad(opt: Option[Any]): String =
    opt match {
      case Some(x: String) => "happy " + x
      case Some(_) => "happy something"
      case None => "sad"
    }

  def loops() = {

    val quietWords = List("Let's", "transform", "some", "collections")
    var noisyWords1 = Buffer.empty[String]
    for (i <- 0 to quietWords.size) // size is O(n) operation for List (singly linkedlist)
      noisyWords1 += quietWords(i).toUpperCase
    // Seq("Let's", "transform", "some", "collections") is also a linked list

    // Faster
    var noisyWords2 = Buffer.empty[String]
    for (word <- quietWords)
      noisyWords2 += word.toUpperCase

    // Best way
    val noisyWords = for (word <- quietWords) yield
      word.toUpperCase

  }

  def loopA() = {
    val salutations = for {
      hello <- List("hello", "greetings")
      world <- List("world", "interwebs")
    } yield "%s %s!".format(hello, world)
  }

  // Used for match Even(n)
  object Even {
    def unapply(n: Int): Option[Int] =
      if (n % 2 == 0) Some(n) else None
  }
}
