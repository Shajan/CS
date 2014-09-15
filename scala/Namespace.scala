import java.io.{File => JavaFile}

object Namespace {
  def main(args: Array[String]) {
    recursiveListFiles(new JavaFile(args(0))).foreach(println)
  }

  def recursiveListFiles(f: JavaFile): Array[JavaFile] = {
    val these = f.listFiles
    these ++ these.filter(_.isDirectory).flatMap(recursiveListFiles)
  }
}

