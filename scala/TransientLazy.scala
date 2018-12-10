import java.io._

object TransientLazy {
  def main(args: Array[String]) {
    testTransient()
  }

  def testTransient() = {
    val t1 = new TransientLazy("t1")
    println(".... no field access yet ...")
    // init : 0 <-- happens here
    println("1 " + t1.tField) // 1 t1-1-[XXX]
    println("2 " + t1.tField) // 2 t1-1-[XXX]

    // Serialize t1
    val oStream = new ByteArrayOutputStream
    new ObjectOutputStream(oStream).writeObject(t1)
    val bytes = oStream.toByteArray

    // Deserialize to t2
    // transient field will be re-computed
    val iStream = new ByteArrayInputStream(bytes)
    val t2 = new ObjectInputStream(iStream)
      .readObject.asInstanceOf[TransientLazy]
    // init : 1 <-- happens here
    println(".... no field access yet ...")
    println("1 " + t2.tField) // 1 t1-2-[YYY]
    println("2 " + t2.tField) // 2 t1-2-[YYY]
  }
}

// Transient fields are re-constructed and not part of serialized object
// These fields are re-constructed after deserialization
class TransientLazy(val state: String) extends Serializable {
  var count: Int = 0
  @transient lazy val tField : String = { // Initialization happens only once
    println("init : " + count)
    count += 1
    "%s-%d-[%d]".format(this.state, count, System.currentTimeMillis) 
  }
}
