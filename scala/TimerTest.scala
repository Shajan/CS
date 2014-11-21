import java.util.{Timer, TimerTask}

object TimerTest {
  def main(args: Array[String]) {
    val timer: Timer = new Timer()
     timer.schedule(new TimerTask() {
        override def run() = {
          println("In timer")
        }
     }, 10)

     timer.schedule(new TimerTask() {
        override def run() = {
          println("In second timer")
        }
     }, 10)

     println("Scheduled both, sleeping")
     Thread.sleep(100)
     timer.cancel()
     println("Done")
  }
}

