import java.io.File

object ListReccursive {
  def main(args: Array[String]) {
    recursiveListFiles(new File(args(0))).foreach(println)
  }

  def recursiveListFiles(f: File): Array[File] = {
    val these = f.listFiles
    these ++ these.filter(_.isDirectory).flatMap(recursiveListFiles)
  }
}

