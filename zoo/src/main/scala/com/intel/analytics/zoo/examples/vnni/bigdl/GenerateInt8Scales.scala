/*
 * Copyright 2016 The BigDL Authors.
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

import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.example.mkldnn.int8.Utils.GenInt8ScalesParams
import com.intel.analytics.bigdl.models.resnet.ImageNetDataSet
import com.intel.analytics.bigdl.nn.{Graph, Module}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.tensorflow.Tensor

/**
 * GenerateInt8Scales will generate a model with scales information,
 * which will be used with mkldnn int8. You can pass a model trained from BigDL
 * and will genereate a model whose name is the same except including "quantized"
 */
object GenerateInt8Scales {
  val logger: Logger = Logger.getLogger(getClass)
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  import com.intel.analytics.bigdl.example.mkldnn.int8.Utils._
  import com.intel.analytics.bigdl.example.mkldnn.int8.GenerateInt8Scales._

  def main(args: Array[String]): Unit = {
    genInt8ScalesParser.parse(args, GenInt8ScalesParams()).foreach { param =>
      val conf = Engine.createSparkConf().setAppName("Quantize the model")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init

      val partitionNum = EngineRef.getCoreNumber()
      val batchsize = param.batchSize
      val images = CaffeInfDemo.read(sc, param.folder).map(v => Sample(v._1))
      val dataset = DataSet.rdd(images) -> SampleToMiniBatch(batchsize)
      val defPath = param.model + "/deploy_overlap.prototxt"
      val modelPath = param.model + "/bvlc.caffemodel"

      val model = CaffeInfDemo.caffe2zoo(Module.loadCaffeModel[Float](defPath, modelPath)).toGraph()
      genereateInt8Scales(model, param.model, dataset.toDistributed().data(false))
      saveQuantizedModel(model, param.model)
    }
  }
}
