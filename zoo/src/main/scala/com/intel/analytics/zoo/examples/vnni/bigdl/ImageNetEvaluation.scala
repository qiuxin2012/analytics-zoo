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

import com.intel.analytics.bigdl.dataset.DistributedDataSet
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.image.imageclassification.ImageClassifier
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

case class ImageNetEvaluationParams(folder: String = "./",
                                    model: String = "",
                                    partitionNum: Int = 32)

object ImageNetEvaluation {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.bigdl.transform.vision").setLevel(Level.ERROR)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    System.setProperty("bigdl.engineType", "mkldnn")

    val parser = new OptionParser[ImageNetEvaluationParams]("ImageNet Int8 Inference Example") {
      opt[String]('f', "folder")
        .text("The folder path that contains ImageNet no-resize sequence files")
        .action((x, c) => c.copy(folder = x))
        .required()
      opt[String]('m', "model")
        .text("The path to the downloaded int8 model snapshot")
        .action((x, c) => c.copy(model = x))
        .required()
      opt[Int]("partitionNum")
        .text("The partition number of the dataset")
        .action((x, c) => c.copy(partitionNum = x))
    }
    parser.parse(args, ImageNetEvaluationParams()).map(param => {
      val sc = NNContext.initNNContext("ImageNet evaluation example with int8 quantized model")
      val images = ImageSet.readSequenceFiles(param.folder, sc, param.partitionNum)
//       If the actual partitionNum of sequence files is too large, then the
//       total batchSize we calculate (partitionNum * batchPerPartition) would be
//       too large for inference.
      // mkldnn runs a single model and single partition on a single node.
      if (images.rdd.partitions.length > param.partitionNum) {
        images.rdd = images.rdd.coalesce(param.partitionNum, shuffle = false)
      }
      val model = ImageClassifier.loadModel[Float](param.model)
      logger.info(s"Start evaluation on dataset under ${param.folder}...")
      val result = model.evaluateImageSet(images,
        Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
      result.foreach(r => println(s"${r._2} is ${r._1}"))
      logger.info("Evaluation finished.")


      val image = sc.range(1, 1000, 1).map(i =>
        (Tensor[Float](3, 224, 224).rand(), i.toString))

      val batchsize = 64
      val bcModel = ModelBroadcast[Float]().broadcast(sc, model)
      val res = image.mapPartitions{imageTensor =>
        val localModel = bcModel.value()
        println("clone clone clone")

        val inputTensor = Tensor[Float](batchsize, 3, 224, 224).rand(-1, 1)
        var i = 0
        while (i < 1000) {
          val start = System.nanoTime()
          localModel.forward(inputTensor)
          val end = System.nanoTime()

          println(s"elapsed ${(end - start) / 1e9}")
          i += 1
        }

        Iterator.single(1)
        //      val inputTensor = Tensor[Float](batchsize, 3, 224, 224)
        //      imageTensor.grouped(batchsize).flatMap{batch =>
        //        val size = batch.size
        //        (0 until size).foreach{i =>
        //          inputTensor.select(1, i + 1).copy(batch(i)._1)
        //        }
        //        val start = System.nanoTime()
        //        val output = localModel.forward(inputTensor).toTensor[Float]
        //        val end = System.nanoTime()
        //        logger.info(s"elapsed ${(end - start) / 1e9} s")
        //        (0 until size).map{i =>
        //          (batch(i)._2, output.valueAt(i + 1, 1),
        //            output.valueAt(i + 1, 2))
        //        }
        //      }
      }.collect()


      sc.stop()
    })
  }
}
