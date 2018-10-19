package com.intel.analytics.zoo.examples.mlperf.recommendation

object NcfLogger {
  val header = ":::MLPv0.5.0 ncf"

  def info(message: String, mapping: Array[(String, String)]): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message: {" +
      f"${mapping.map(m => s""""${m._1}": ${m._2}""").mkString(", ")}}")
  }

  def info(message: String, value: Float): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message $value")
  }

  def info(message: String, value: String): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message: $value")
  }

  def info(message: String, value: Int): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message: $value")
  }

  def info(message: String, value: Boolean): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message: $value")
  }

  def info(message: String): Unit = {
    val ste = Thread.currentThread().getStackTrace()
    println(f"$header ${System.currentTimeMillis() / 1e3}%10.3f " +
      f"(${ste(2).getFileName}:${ste(2).getLineNumber}) $message")
  }

}
