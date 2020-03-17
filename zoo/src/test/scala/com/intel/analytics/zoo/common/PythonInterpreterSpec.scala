package com.intel.analytics.zoo.common

import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import jep.NDArray
import org.apache.spark.{SparkConf, SparkContext}

class PythonInterpreterSpec extends ZooSpecHelper{
  "interp" should "work" in {
    PythonInterpreter.exec("import numpy as np")
    (0 until 1).toParArray.foreach{i =>
      PythonInterpreter.exec("np.array([1, 2, 3])")
    }
    val sc = new SparkContext(new SparkConf().setAppName("app").setMaster("local[4]"))
    (0 to 10).foreach(i =>
      sc.parallelize(0 to 10, 1).mapPartitions(i => {
        println(Thread.currentThread())
        PythonInterpreter.exec("a = np.array([1, 2, 3])")
        i
      }).count()
    )
    println(PythonInterpreter.getValue[NDArray[_]]("a").getData())
  }

}
