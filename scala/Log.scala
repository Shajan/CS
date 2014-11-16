import java.util.logging._

object Log {
  def main(args: Array[String]) {
    val t = new MyTest
    t.test
  }
}

class MyTest {
  val logger = Logger.getLogger(this.getClass.getName)
  def test = {
    logger.setLevel(Level.WARNING);  // Log warning or higher
    // In decreasing order of severity
    logger.finest("hello finest")
    logger.finer("hello finer")
    logger.fine("hello fine")
    logger.config("hello config")
    logger.warning("hello warning")
    logger.info("hello info")
    logger.severe("hello severe")
  }
}

