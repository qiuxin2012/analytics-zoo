/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.examples.vnni.bigdl

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.models.image.imageclassification.ImageClassifier
import org.apache.log4j.Logger
import scopt.OptionParser

case class PerfParams(model: String = "",
                      batchSize: Int = 32,
                      iteration: Int = 1000)

object Perf {

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.engineType", "mkldnn")

    val parser = new OptionParser[PerfParams]("ResNet50 Int8 Performance Test") {
      opt[String]('m', "model")
        .text("The path to the downloaded int8 model snapshot")
        .action((v, p) => p.copy(model = v))
        .required()
      opt[Int]('b', "batchSize")
        .text("The batch size of input data")
        .action((v, p) => p.copy(batchSize = v))
      opt[Int]('i', "iteration")
        .text("The number of iterations to run the performance test. " +
          "The result should be the average of each iteration time cost")
        .action((v, p) => p.copy(iteration = v))
    }

    parser.parse(args, PerfParams()).foreach { param =>
      val batchSize = param.batchSize
      Engine.init

      val model = ImageClassifier.loadModel[Float](param.model)
      model.setEvaluateStatus()

      val inputTensor = Tensor[Float](64, 3, 224, 224).rand(-1, 1)
      var i = 0
      while (i < 1000) {
        val start = System.nanoTime()
        model.forward(inputTensor)
        val end = System.nanoTime()

        println(s"elapsed ${(end - start) / 1e9}")
        i += 1
      }

    }
  }
}
