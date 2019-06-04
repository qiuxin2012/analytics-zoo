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

import java.io.{File, PrintWriter}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, MklDnn}
import com.intel.analytics.zoo.app.ImageProcessing
import com.intel.analytics.zoo.models.image.imageclassification.ImageClassifier
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

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

  def read(sc: SparkContext,
           inputPath: String,
           numP: Int = 1): RDD[(Tensor[Float], String)] = {
    val coreNumber = EngineRef.getCoreNumber()
    val imgPath123 = sc.textFile(inputPath,
      numP).map(_.trim())
      .filter(_.size > 0).cache()
    imgPath123.count()
    val imgPath = imgPath123.coalesce(numP).cache()

    imgPath.count()
    val cachePathTime = System.nanoTime()
    val image = imgPath.mapPartitions{ path =>
      val preProcessing = Array.tabulate(coreNumber)(_ =>
        new ImageProcessing()
      )
      path.grouped(coreNumber).flatMap{batchPath =>
        batchPath.indices.toParArray.map{i =>
          (preProcessing(i).preprocess(batchPath(i)), batchPath(i))
        }
      }
    }.cache()
    image.count()
    println(s"cache data cost ${(System.nanoTime() - cachePathTime) / 1e9}")
    image
  }

  def convert[T: ClassTag](args: Object*)(
    implicit ev: TensorNumeric[T]): Module[T] = {
    val obj = "com.intel.analytics.bigdl.utils.intermediate.ConversionUtils"
    val methodName = "convert"
    val clazz = Class.forName(obj)
    val argsWithTag = args ++ Seq(implicitly[reflect.ClassTag[T]], ev)
    val method =
      try {
        clazz.getMethod(methodName, argsWithTag.map(_.getClass): _*)
      } catch {
        case t: Throwable =>
          val methods = clazz.getMethods().filter(_.getName() == methodName)
              .filter(_.getParameterCount == argsWithTag.size)
          require(methods.length == 1,
            s"We should only found one result, but got ${methodName}: ${methods.length}")
          methods(0)
      }
    method.invoke(obj, argsWithTag: _*).asInstanceOf[Module[T]]
  }


  def main(args: Array[String]): Unit = {
    val params = parser.parse(args, Param()).get
//    val sc = NNContext.initNNContext("Caffe Test")
    val conf = Engine.createSparkConf()
      .setAppName("test on caffe model")
    val sc = new SparkContext(conf)
    Engine.init
    val batchsize = params.batchSize
    val numberOfPartiton = EngineRef.getEngineType() match {
      case MklDnn => EngineRef.getNodeNumber()
      case MklBlas => EngineRef.getCoreNumber() * EngineRef.getNodeNumber()
    }

    val image = read(sc, params.inputPath, numberOfPartiton)

    val model = if (params.defPath != "") {
      val loadedModel = Module.loadCaffeModel[Float](params.defPath, params.modelPath).toGraph()
      convert[Float](loadedModel, Boolean.box(false))
    } else {
      val loadedModel = Module.loadModule[Float](params.modelPath).toGraph()
      convert[Float](loadedModel, Boolean.box(false))
    }

    val s = System.nanoTime()
    val bcModel = ModelBroadcast[Float]().broadcast(sc, model)
    val res = image.mapPartitions{imageTensor =>
      val localModel = bcModel.value()
      val inputTensor = Tensor[Float](batchsize, 3, 224, 224)
      imageTensor.grouped(batchsize).flatMap{batch =>
        val size = batch.size
        val startCopy = System.nanoTime()
        (0 until size).toParArray.foreach { i =>
          inputTensor.select(1, i + 1).copy(batch(i)._1)
        }
        logger.info(s"Copy elapsed ${(System.nanoTime() - startCopy) / 1e9} s")
        val start = System.nanoTime()
        val output = localModel.forward(inputTensor).toTensor[Float]
        val end = System.nanoTime()
        logger.info(s"elapsed ${(end - start) / 1e9} s")
        (0 until size).map{i =>
          (batch(i)._2, output.valueAt(i + 1, 1),
            output.valueAt(i + 1, 2))
        }
      }
    }.collect()

    val e = (System.nanoTime() - s) / 1e9
    val thp = image.count() / e
    println ("Time is ", e)
    println ("Throuphput is ", thp)
    val writer = new PrintWriter(new File(params.outputPath))
    res.foreach(x =>
      {
        writer.write(s"${x._1} ${x._2}  ${x._3}\n")
      })
    writer.close()
  }

}
