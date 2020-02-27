package com.intel.analytics.zoo.examples.pytorch

import com.intel.analytics.zoo.common.NNContext.initNNContext
import com.intel.analytics.zoo.pipeline.estimator.python.PythonEstimator
import org.apache.spark.{SparkConf, SparkContext}

object Test {
  def main(args: Array[String]): Unit = {
    val sc =initNNContext("1234")
    new PythonEstimator[Float]().estimatorTest()
  }

}
