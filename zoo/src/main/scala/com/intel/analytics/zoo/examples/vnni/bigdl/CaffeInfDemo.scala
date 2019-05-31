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

import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.models.image.imageclassification.ImageClassifier
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser
import org.apache.spark.SparkContext

import scala.reflect.ClassTag

object CaffeInfDemo {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class Param(defPath: String = "", outputPath: String = "",
                   modelPath: String = "", inputPath: String = "",
                   batchSize: Int = 4)

  val parser = new OptionParser[Param]("Analytics Zoo Streaming Object Detection") {
    head("Analytics Zoo Streaming Object Detection")
    opt[String]('d', "defPath")
      .text("folder that used to store the streaming paths")
      .action((x, c) => c.copy(defPath = x))
      .required()
    opt[String]('o', "outputPath")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputPath = x))
      .required()
    opt[String]('i', "inputPath")
      .text("where you put the input txt")
      .action((x, c) => c.copy(inputPath = x))
      .required()
    opt[String]('m', "modelPath")
      .text("Analytics Zoo model path")
      .action((x, c) => c.copy(modelPath = x))
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
  }

  def main(args: Array[String]): Unit = {
    val params = parser.parse(args, Param()).get
//    val sc = NNContext.initNNContext("Caffe Test")
    val conf = Engine.createSparkConf()
      .setAppName("test on caffe model")
    val sc = new SparkContext(conf)
    Engine.init

//    val imgPath123 = sc.textFile(params.inputPath,
//      1).map(_.trim())
//      .filter(_.size > 0).cache()
//    imgPath123.count()
//    val imgPath = imgPath123.coalesce(1).cache()
//
//    imgPath.count()
//    val cachePathTime = System.nanoTime()
//    val image = imgPath.map { path =>
//      (ImageProcessing.preprocessImage(path), path)
//    }.cache()
//
//    image.count()
//    println(s"cache path cost ${(System.nanoTime() - cachePathTime) / 1e9}")

    val image = sc.range(1, 40000, 1, 1).map(i =>
      (Tensor[Float](3, 224, 224).rand(), i.toString)).mapPartitions{iter =>
      val batchsize = params.batchSize
      iter.grouped(batchsize).toParArray.map{ batch =>
        val inputTensor = Tensor[Float](batchsize, 3, 224, 224)
        val size = batch.size
        (0 until size).foreach { i =>
          inputTensor.select(1, i + 1).copy(batch(i)._1)
        }
        (inputTensor, batch.map(_._2))
      }.toIterator
    }
    image.cache().count()
    val batchsize = params.batchSize

//    val model = new InferenceModel()
//    println(EngineRef.getCoreNumber())
//    model.doLoadCaffe(params.defPath, params.modelPath)
//    val caffeModel = Module.loadCaffeModel[Float](params.defPath, params.modelPath)
//       .toGraph()
//    val t = Tensor[Float](batchsize, 3, 224, 224)
//    var i = 1
//    image.take(batchsize).foreach{v =>
//      t.select(1, i).copy(v._1)
//      i += 1
//    }
//    caffeModel.evaluate()
//    caffeModel.setInputDimMask(0)
//    caffeModel.setOutputDimMask(0)
//    caffeModel.setWeightDimMask(1)
//    caffeModel.forward(t)
//    KerasUtils.invokeMethod(caffeModel, "calcScales",
//      t)
//
//
//    val clazz = Class.forName("com.intel.analytics.bigdl.utils.intermediate.ConversionUtils")
//    val m = clazz.getMethods().filter(_.getName == "convert")
//      .filter(_.getParameterTypes.size == 4)(0)
//    val model = m.invoke(clazz, caffeModel, Boolean.box(true), ClassTag.Float,
//      TensorNumeric.NumericFloat).asInstanceOf[Module[Float]]
//     val model = Module.loadModule[Float](params.modelPath)
    val model = ImageClassifier.loadModel[Float](params.modelPath)
    model.setEvaluateStatus()

//    val inputTensor = Tensor[Float](64, 3, 224, 224).rand(-1, 1)
//    var i = 0
//    while (i < 1000) {
//      val start = System.nanoTime()
//      model.forward(inputTensor)
//      val end = System.nanoTime()
//
//      println(s"elapsed ${(end - start) / 1e9}")
//      i += 1
//    }


    val s = System.nanoTime()
    val bcModel = ModelBroadcast[Float]().broadcast(sc, model)
    val res = image.mapPartitions{imageTensor =>
      val localModel = bcModel.value()
      imageTensor.map{batch =>
        val inputTensor = batch._1
        val size = batch._2.size
        val start = System.nanoTime()
        val output = localModel.forward(inputTensor).toTensor[Float]
        val end = System.nanoTime()
        logger.info(s"elapsed ${(end - start) / 1e9} s")
        (0 until size).map{i =>
          (batch._2(i), output.valueAt(i + 1, 1),
            output.valueAt(i + 1, 2))
        }
      }
    }.collect()
//    val inputTensor = Tensor[Float](batchsize, 3, 224, 224)
//    image.collect().grouped(batchsize).foreach{batch =>
//      val size = batch.size
//      (0 until size).foreach{i =>
//        inputTensor.select(1, i + 1).copy(batch(i)._1)
//      }
//      val output = model.forward(inputTensor).toTensor[Float]
//      (0 until size).map{i =>
//        (batch(i)._2, output.valueAt(i + 1, 1),
//          output.valueAt(i + 1, 2))
//      }
//
//    }

    val e = (System.nanoTime() - s) / 1e9
    val thp = image.count() / e
    println ("Time is ", e)
    println ("Throuphput is ", thp)
//    println(res.mkString("\n"))
//    val writer = new PrintWriter(new File(params.outputPath))
//    res.foreach(x =>
//      {
////        println(x.toString)
//        val str = x._1.toString.split("\n")
//        writer.write(x._1 + " " + x._2 + " " + x._3 + "\n")
//      })
//    writer.close()
    //    model = new Resnet50InferenceModel()
//    model.doLoadCaffe(defPath, modelPath)
//
//
//    val imgPath = sc.textFile(params.inputPath)
//
//    val tensor = model.preprocess(imgPath).resize()
//
//    model.doPredict(tensor)

  }

}
