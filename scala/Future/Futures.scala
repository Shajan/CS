import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

// http://danielwestheide.com/blog/2013/01/09/the-neophytes-guide-to-scala-part-8-welcome-to-the-future.html
/*
 * object Future {
 *   def apply[T](body: => T)(implicit execctx: ExecutionContext): Future[T]
 * }
 */
object Futures {
  //@volatile var where_is_main_thread: String = ""
  var where_is_main_thread: String = ""
  def main(args: Array[String]) {
    reallySimple()
/*
    where_is_main_thread = "main"
    simple()
    where_is_main_thread = "done sample"
    Thread.sleep(100)
*/
  }

  def reallySimple() = {
    val f1 = Future { 1 } 
    println(f1.value) // prints None 
    f1 map { i => println(i) } // prints 1
  }

  def log(s: String) = {
    println("%s:%s:%s".format(s, Thread.currentThread.getName(), where_is_main_thread))
  }

  def simple() = {
    where_is_main_thread = "simple"
    val f1 = Future {
      log("f1 start")
      Thread.sleep(10)
      log("f1 end")
      1
    }
    where_is_main_thread = "defined f1"

    val f2 = Future {
      log("f2 start")
      Thread.sleep(10)
      log("f2 end")
      2
    }
    where_is_main_thread = "defined f2"

    f1.map { i => log(i.toString) }
    f2.map { i => log(i.toString) }

    where_is_main_thread = "defined map"
  }
}

