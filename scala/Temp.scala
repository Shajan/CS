import scala.util.matching.Regex
import scala.collection.mutable.Map

object Temp {
  def main(args: Array[String]) {
    val m = Map[String, Seq[Int]]()
    val l1 = Seq(1, 2, 3)
    m += ("3" -> l1)
    m += ("0" -> Seq[Int]())
    tst("-", m)
    tst("0", m)
    tst("3", m)
  }

  def tst(s: String, m: Map[String, Seq[Int]]) = {
    m.get(s) match {
      case Some(seq) if (seq.size > 0) => println("found")
      case Some(seq) => println("empty")
      case None => println("not found")
    }
  }

  def tst1() {
    val jvmCmd: Seq[String] = Seq("java", "-Xms200M", "-Xmx200M", "-jar", "vireotool.jar")
    val formatArgs: Seq[String] = Seq("-inFormat=binary", "-outFormat=binary")
    val operationArgs: Seq[String] = Seq("-op=" + "tanscode")
    val ioArgs: Seq[String] = Seq("-input=" + "-", "-output=" + "-")

    val cmd = jvmCmd ++ formatArgs ++ operationArgs ++ ioArgs
    val r = "^-op=(.*)".r
    val op = find(r, cmd)
    val newCmd = removeOp(r, cmd)

    println(cmd)
    println(op)
    println(newCmd)
  }

  def find(r: Regex, cmd: Seq[String]): String = {
    val op = cmd.filter(c => c match { case r(op) => true; case _ => false })
    if (op.size == 0) throw new Exception("Undefined operation") 
    if (op.size > 1) throw new Exception("Ambigious operation")
    op.head
  }

  def removeOp(r: Regex, cmd: Seq[String]) = {
    val filtered = cmd.filter(c => c match { case r(op) => false; case _ => true })
    filtered
  }
}
