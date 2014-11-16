import java.util.logging._

object Log {
  def main(args: Array[String]) {
    Console.log
    NoConsole.log
  }
}

object Console {
  val logger = Logger.getLogger(this.getClass.getName)
  logger.setLevel(Level.WARNING) // Log warning or higher
  def log = {
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

object NoConsole {
/* 
//To globally remove handlers
val globalLogger = Logger.getLogger("global")
val handlers = globalLogger.getHandlers()
for (handler <- handlers)
    globalLogger.removeHandler(handler)
*/
  val logger = Logger.getLogger(this.getClass.getName)
  logger.setUseParentHandlers(false)
  logger.setLevel(Level.WARNING)

  // create a TXT formatter
  val formatter = new SimpleFormatter
  val filehandler = new FileHandler("/tmp/Logging.txt")
  filehandler.setFormatter(formatter)
  logger.addHandler(filehandler)

  def log = {
    logger.warning("hello warning")
    logger.info("hello info")
    logger.severe("hello severe")
  }
}

