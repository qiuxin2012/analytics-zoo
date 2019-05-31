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

import java.io.{BufferedOutputStream, FileOutputStream}

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.caffe.CaffeLoader
import com.intel.analytics.zoo.common.{NNContext, Utils}
import com.intel.analytics.zoo.examples.streaming.objectdetection.StreamingObjectDetection.{readFile, writeFile}
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.models.image.common.ImageModel
import com.intel.analytics.zoo.models.image.imageclassification.{ImageClassifier, LabelOutput}
import com.intel.analytics.zoo.feature.image
import org.apache.log4j.{Level, Logger}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.opencv.imgcodecs.Imgcodecs
import scopt.OptionParser


object CaffeDemo {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class Param(defPath: String = "", outputPath: String = "",
                   modelPath: String = "", inputPath: String = "")

  val parser = new OptionParser[Param]("Analytics Zoo Streaming Object Detection") {
    head("Analytics Zoo Streaming Object Detection")
    opt[String]('d', "defPath")
      .text("folder that used to store the streaming paths")
      .action((x, c) => c.copy(defPath = x))
    opt[String]('o', "outputPath")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputPath = x))
    opt[String]('i', "inputPath")
      .text("where you put the input txt")
      .action((x, c) => c.copy(inputPath = x))
      .required()
    opt[String]('m', "modelPath")
      .text("Analytics Zoo model path")
      .action((x, c) => c.copy(modelPath = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.ERROR)
    System.setProperty("bigdl.engineType", "mkldnn")
    val params = parser.parse(args, Param()).get
    val sc = NNContext.initNNContext("Caffe Test")

    // val model = CaffeLoader.loadCaffe[Float](modelPath = "", defPath = "")
    //  ._1.asInstanceOf[Graph[Float]]

    val model = ImageClassifier.loadCaffeModel[Float](params.defPath, params.modelPath)

    // val model = ImageClassifier.loadModel[Float](params.model)

    val imgPath = sc.textFile(params.inputPath)

    val images = ImageSet.rdd(imgPath.map(path => readFile(path)))
    images -> ImageBytesToMat(imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR)
//    val input = images ->
//      ImageBytesToMat(imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR) ->
//      ImageCenterCrop(224, 224) ->
//      ImageMatToTensor() ->
//      ImageSetToSample()


//    val test = model.predictImageSet(images).toDistributed().rdd.map(
//      x => (x.uri(), x.toString)
//    )
//    test.collect()
//    val p = images.rdd.collect()

    val result = model.predictImageSet(images)
      // .toDistributed().rdd.map(x => (x.uri(), x.toString)).collect()
//    val tp = result.toDistributed().rdd.collect()
    val r = result.toDistributed().rdd.
      map(f => (f.uri(), f[Tensor[Float]](ImageFeature.predict))).collect()

//    val labelOutput = LabelOutput(model.getConfig().labelMap, probAsOutput = false)
//    val results = labelOutput(result).toDistributed().rdd.collect()

    import java.io._
    val writer = new PrintWriter(new File("/home/litchy/tmp/health.txt"))
    r.foreach(x => writer.write(x._1 + " " +  x._2.toString.charAt(0) + "\n"))

    writer.close()


//    results.foreach(x => {
//      val cls = x("classes").asInstanceOf[Array[String]]
//      val probs = x("probs").asInstanceOf[Array[Float]]
//      val uri = x("uri").asInstanceOf[String]
//
//      val rstr = if (probs(0) < probs(1)) "1" else "0"
//      writer.write(uri + " " + probs(0).toString + "\n")
//      println(x.toString)
//
//    })
//    writer.close()


//      for (i <- 0 to 1) {
//        writer.write(cls(i) + probs(i) + "\n")
//      }

//    val res1 = model.predictImageSet(images)
//    val r1 = res1.toDistributed().rdd.collect()
//    r1.foreach(x => println(x.toString))


  }

}
