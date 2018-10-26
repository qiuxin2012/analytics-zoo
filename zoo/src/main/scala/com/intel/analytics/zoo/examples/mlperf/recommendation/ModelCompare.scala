package com.intel.analytics.zoo.examples.mlperf.recommendation

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.{AbstractDataSet, LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.nn.BCECriterion
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.models.image.imageclassification.Dataset
import com.intel.analytics.zoo.models.recommendation.NeuralCFV2

object ModelCompare {
  def main(args: Array[String]): Unit = {
    Engine.init
    import scala.collection.JavaConverters._

    val userCount = 128
    val itemCount = 100
    val hiddenLayers = Array(256, 256, 128, 64)
    val numFactors = 64
    val beta1 = 0.8
    val beta2 = 0.99
    val learningRate = 0.01
    val epoch = 1
    val iteration = 2
    val ei = 10
    val endTrigger = Trigger.maxIteration(ei * (epoch - 1) + iteration)

    val ncf = NeuralCFV2[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 1,
      userEmbed = hiddenLayers(0) / 2,
      itemEmbed = hiddenLayers(0) / 2,
      hiddenLayers = hiddenLayers.slice(1, hiddenLayers.length),
      mfEmbed = numFactors)

    val pyBigDL = new PythonBigDL[Float]()
    val pytorchW = com.intel.analytics.bigdl.utils.File
      .load[java.util.HashMap[String, JTensor]]("init_model.obj")
      .asScala.map(v => (v._1, pyBigDL.toTensor(v._2)))
    val embeddingNames = Array("mfUserEmbedding", "mfItemEmbedding",
      "mlpUserEmbedding", "mlpItemEmbedding")
    val fcNames = Array("fc256->256", "fc256->128",
      "fc128->64", "fc128->1")
    embeddingNames.foreach{name =>
      ncf.ncfModel(name).get.setWeightsBias(Array(pytorchW(s"${name}_weight")))
    }
    fcNames.foreach{name =>
      ncf.ncfModel(name).get.setWeightsBias(Array(
        pytorchW(s"${name}_weight"), pytorchW(s"${name}_bias")))
    }

    val ncfLocal = ncf.cloneModule()

    val data = com.intel.analytics.bigdl.utils.File
      .load[java.util.HashMap[String, JTensor]]("data.obj")
      .asScala.map(v => (v._1, pyBigDL.toTensor(v._2)))
    val dataArray = (0 until data.size / 3).map{ite =>
      val userId = data(s"user$ite")
      val itemId = data(s"item$ite")
      val label = data(s"label$ite")

      val input = Tensor[Float](userId.size(1), 2)
      input.select(2, 1).copy(userId)
      input.select(2, 2).copy(itemId)
      input.add(1)
      (input, label)
    }.toArray
    val miniBatchArray = dataArray.map(v => MiniBatch(v._1, v._2))
    val trainDataset = new SequenceLocalArrayDataSet[MiniBatch[Float]](miniBatchArray)

    val optimMethods = Map(
      "embeddings" -> new EmbeddingAdam2[Float](
        learningRate = learningRate,
        beta1 = beta1,
        beta2 = beta2,
        userCount = userCount,
        itemCount = itemCount),
      "linears" -> new ParallelAdam[Float](
        learningRate = learningRate,
        beta1 = beta1,
        beta2 = beta2))
    val optimMethod = new Adam[Float](
      learningRate = learningRate,
      beta1 = beta1,
      beta2 = beta2)

    val criterion = BCECriterion[Float]()
    val optimizer = new NCFOptimizer2[Float](ncf, trainDataset, criterion)

//    val endTrigger = Trigger.maxEpoch(10)
    optimizer
//      .setOptimMethod(optimMethod)
      .setOptimMethods(optimMethods)
      .setEndWhen(endTrigger)
      .optimize()

    new LocalOptimizer[Float](ncfLocal,
       trainDataset, criterion)
      .setOptimMethod(optimMethod)
      .setEndWhen(endTrigger)
      .optimize()


    val finalWeight = com.intel.analytics.bigdl.utils.File
      .load[java.util.HashMap[String, JTensor]](s"e${epoch-1}i${iteration-1}.obj")
      .asScala.map(v => (v._1, pyBigDL.toTensor(v._2)))

    embeddingNames.foreach{name =>
      val param = ncf.ncfModel(name).get.getParametersTable()
      val pytorchParam = finalWeight(s"${name}_weight")
      val pytorchGrad = finalWeight(s"${name}_gradWeight")
      val localParam = ncfLocal.ncfModel(name).get.getParametersTable()
      println()

    }
    fcNames.foreach{name =>
      val param = ncf.ncfModel(name).get.parameters()
      val linearWeight = finalWeight(s"${name}_weight")
      val linearBias = finalWeight(s"${name}_bias")
      val initWeight = pytorchW(s"${name}_weight")
      val initBias = pytorchW(s"${name}_bias")
      println
    }

  }


  class SequenceLocalArrayDataSet[T] (buffer: Array[T]) extends LocalDataSet[T] {
    override def shuffle(): Unit = {
    }

    override def data(train: Boolean): Iterator[T] = {
      new Iterator[T] {
        private val index = new AtomicInteger()

        override def hasNext: Boolean = {
          if (train) {
            true
          } else {
            index.get() < buffer.length
          }
        }

        override def next(): T = {
          val curIndex = index.getAndIncrement()
          if (train || curIndex < buffer.length) {
            buffer(if (train) (curIndex % buffer.length) else curIndex)
          } else {
            null.asInstanceOf[T]
          }
        }
      }
    }

    override def size(): Long = buffer.length
  }
}
