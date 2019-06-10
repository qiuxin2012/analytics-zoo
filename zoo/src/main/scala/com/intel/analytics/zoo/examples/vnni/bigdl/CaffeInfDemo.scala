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
import com.intel.analytics.bigdl.models.resnet.{Convolution, ResNet}
import com.intel.analytics.bigdl.models.resnet.ResNet.DatasetType.ImageNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.zoo.app.ImageProcessing
import com.intel.analytics.zoo.models.image.imageclassification.ImageClassifier
import com.intel.analytics.bigdl.numeric.NumericFloat
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

  def caffe2zoo(model: Module[Float]): Module[Float] = {
    val newModel =
      graph(2, T("depth" -> 50, "shortcutType" -> ShortcutType.B, "dataSet" -> ImageNet,
        "optnet" -> false))

    val pt = model.getParametersTable()
    val newPt = newModel.getParametersTable()

    // copy parameter without scale
    newPt.keySet.map(_.toString).foreach{ key =>
      pt[Table](key).keySet.foreach{key2 =>
        newPt[Table](key).apply[Tensor[Float]](key2).copy(
          pt[Table](key).apply[Tensor[Float]](key2)
        )
      }
    }

    // copy parameter from scale to bn
    pt.keySet.map(_.toString).filter(_.contains("scale")).foreach{key =>
      val bnkey = key.replace("scale", "bn")
      pt[Table](key).keySet.foreach{k =>
        newPt[Table](bnkey)[Tensor[Float]](k).copy(
          pt[Table](key)[Tensor[Float]](k)
        )
      }
    }
    newModel
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
      val loadedModel = Module.loadCaffeModel[Float](params.defPath, params.modelPath)
      convert[Float](caffe2zoo(loadedModel), Boolean.box(false)).evaluate()
    } else {
      val loadedModel = Module.loadModule[Float](params.modelPath).quantize()
      loadedModel.evaluate()
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

  var iChannels = 0
  def graph(classNum: Int, opt: Table): Module[Float] = {
    val depth = opt.get("depth").getOrElse(18)
    val shortCutType = opt.get("shortcutType")
    val shortcutType = shortCutType.getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
    val dataset = opt.getOrElse("dataSet", DatasetType.CIFAR10).asInstanceOf[DatasetType]
    val optnet = opt.get("optnet").getOrElse(true)

    def shortcutFunc(nInputPlane: Int, nOutputPlane: Int, stride: Int, input: ModuleNode[Float],
                     name: String)
    : ModuleNode[Float] = {
      val useConv = shortcutType == ShortcutType.C ||
        (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

      if (useConv) {
        val conv1 = Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride,
          optnet = optnet).setName(s"res${name}_branch1").inputs(input)
        val bn1 = Sbn(nOutputPlane).setName(s"bn${name}_branch1").inputs(conv1)
        bn1
      } else if (nInputPlane != nOutputPlane) {
        val pool1 = SpatialAveragePooling(1, 1, stride, stride).inputs(input)
        val mul1 = MulConstant(0f).inputs(pool1)
        val concat = JoinTable(2, 0).inputs(pool1, mul1)
        concat
      } else {
        input
      }
    }

    def bottleneckFunc(n: Int, stride: Int, input: ModuleNode[Float], name: String): ModuleNode[Float] = {
      val nInputPlane = iChannels
      iChannels = n * 4

      val conv1 = Convolution(nInputPlane, n, 1, 1, 1, 1, 0, 0, optnet = optnet)
        .setName(s"res${name}_branch2a").inputs(input)
      val bn1 = Sbn(n).setName(s"bn${name}_branch2a").inputs(conv1)
      val relu = ReLU(true).setName(s"res${name}_branch2a_relu").inputs(bn1)
      val conv2 = Convolution(n, n, 3, 3, stride, stride, 1, 1, optnet = optnet)
        .setName(s"res${name}_branch2b").inputs(relu)
      val bn2 = Sbn(n).setName(s"bn${name}_branch2b").inputs(conv2)
      val relu2 = ReLU(true).setName(s"res${name}_branch2b_relu").inputs(bn2)
      val conv3 = Convolution(n, n*4, 1, 1, 1, 1, 0, 0, optnet = optnet)
        .setName(s"res${name}_branch2c").inputs(relu2)
      val sbn = Sbn(n * 4).setInitMethod(Zeros, Zeros).setName(s"bn${name}_branch2c").inputs(conv3)

      val shortcut = shortcutFunc(nInputPlane, n * 4, stride, input, name)
      val add = CAddTable(true).setName(s"res${name}").inputs(shortcut, sbn)
      val output = ReLU(true).setName(s"res${name}_relu").inputs(add)
      output
    }

    def layer(block: (Int, Int, ModuleNode[Float], String) => ModuleNode[Float], features: Int,
              count: Int, id: Int, stride: Int = 1)(input: ModuleNode[Float]): ModuleNode[Float] = {
      var output = block(features, stride, input, s"${id}a")
      (1 until count).foreach {i =>
        output = block(features, 1, output, s"${id}${('a' + i).toChar}")
      }
      output
    }

    val model = if (dataset == DatasetType.ImageNet) {
      val cfg = Map(
        50 -> ((3, 4, 6, 3), 2048,
          bottleneckFunc: (Int, Int, ModuleNode[Float], String) => ModuleNode[Float]) //,
//        101 -> ((3, 4, 23, 3), 2048,
//          bottleneckFunc: (Int, Int, ModuleNode[Float]) => ModuleNode[Float]),
//        152 -> ((3, 8, 36, 3), 2048,
//          bottleneckFunc: (Int, Int, ModuleNode[Float]) => ModuleNode[Float]),
//        200 -> ((3, 24, 36, 3), 2048,
//          bottleneckFunc: (Int, Int, ModuleNode[Float]) => ModuleNode[Float])
      )

      require(cfg.keySet.contains(depth), s"Invalid depth ${depth}")

      val (loopConfig, nFeatures, block) = cfg.get(depth).get
      iChannels = 64
      logger.info(" | ResNet-" + depth + " ImageNet")

      val input = Input()
      val conv1 = Convolution(3, 64, 7, 7, 2, 2, 3, 3,
        optnet = optnet, propagateBack = false).setName("conv1").inputs(input)
      val bn = Sbn(64).setName("bn_conv1").inputs(conv1)
      val relu = ReLU(true).setName("conv1_relu").inputs(bn)
      val pool = SpatialMaxPooling(3, 3, 2, 2, 0, 0).ceil().setName("pool1").inputs(relu)
      val layer1 = layer(block, 64, loopConfig._1, 2)(pool)
      val layer2 = layer(block, 128, loopConfig._2, 3, 2)(layer1)
      val layer3 = layer(block, 256, loopConfig._3, 4, 2)(layer2)
      val layer4 = layer(block, 512, loopConfig._4, 5, 2)(layer3)
      val pool2 = SpatialAveragePooling(7, 7, 1, 1).setName("pool5").inputs(layer4)
      val view = View(nFeatures).setNumInputDims(3).inputs(pool2)
      val fc = Linear(nFeatures, classNum, true, L2Regularizer(1e-4), L2Regularizer(1e-4))
        .setName(s"fc1000").setInitMethod(RandomNormal(0.0, 0.01), Zeros).inputs(view)
      val output = SoftMax().setName("probt").inputs(fc)
      Graph(input, output)
    } else {
      throw new IllegalArgumentException(s"Invalid dataset ${dataset}")
    }
    model
  }

}

object Sbn {
  def apply[@specialized(Float, Double) T: ClassTag](
      nOutput: Int,
      eps: Double = 1e-5, // 1e-5 in caffe
      momentum: Double = 0.1,
      affine: Boolean = true)(implicit ev: TensorNumeric[T]): SpatialBatchNormalization[T] = {
    SpatialBatchNormalization[T](nOutput, eps, momentum, affine).setInitMethod(Ones, Zeros)
  }
}
