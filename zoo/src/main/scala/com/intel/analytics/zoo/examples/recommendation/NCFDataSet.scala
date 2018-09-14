package com.intel.analytics.zoo.examples.recommendation

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator}

import scala.util.Random

class NCFDataSet[MiniBatch[Float]] private[dataset](
    trainSet: Seq[(Int, Set[Int])],
    valPos: Map[Int, Int],
    train: Seq[(Int, Set[Int])],
    trainNegatives: Int, batchSize: Int) extends LocalDataSet[MiniBatch[Float]] {
  val input = Tensor[Float](batchSize, 2)
  val label = Tensor[Float](batchSize, 1)
  val miniBatch = MiniBatch(Array(input), Array(label))

  val trainSize = trainSet.map(_._2.size).sum
  // origin set use to random train negatives
  val originSet = trainSet.map(v => (v._1, v._2 + valPos(v._1)))

  val trainBuffer = new Array[Float](trainSize * 2)
  NCFDataSet.copy(trainSet, trainBuffer)

  val buffer = new Array[Float](trainSize * (1 + trainNegatives) * 2 )

  override def shuffle(): Unit = {
    RandomGenerator.shuffle(buffer)
  }

  override def data(train: Boolean): Iterator[MiniBatch[Float]] = {
    new Iterator[MiniBatch[Float]] {
      private val index = new AtomicInteger()

      override def hasNext: Boolean = {
        if (train) {
          true
        } else {
          index.get() < buffer.length
        }
      }

      override def next(): MiniBatch[Float] = {
        val curIndex = index.getAndIncrement()
        if (train || curIndex < buffer.length) {
          miniBatch
        } else {
          null.asInstanceOf[MiniBatch[Float]]
        }
      }
    }
  }

  override def size(): Long = buffer.length
}

object NCFDataSet {
  def copy(trainSet: Seq[(Int, Set[Int])], trainBuffer: Array[Float]): Unit = {
    var i = 0
    var offset = 0
    while(i < trainSet.size) {
      val userId = trainSet(i)._1
      val itemIds = trainSet(i)._2.toIterator

      while(itemIds.hasNext) {
        val itemId = itemIds.next()
        trainBuffer(offset) = userId
        trainBuffer(offset + 1) = itemId

        offset += 2
      }

      i += 1
    }

  }

  def shuffle(buffer: Array[Float]): Unit = {
    var i = 0
    val length = buffer.length / 2
    while (i < length) {
      val exchange = RandomGenerator.RNG.uniform(0, length - i).toInt + i
      val tmp1 = buffer(exchange * 2)
      val tmp2 = buffer(exchange * 2 + 1)

      buffer(exchange * 2) = buffer(i * 2)
      buffer(exchange * 2 + 1) = buffer(i * 2 + 1)
      buffer(2 * i) = tmp1
      buffer(2 * i + 1) = exchange + 1
      i += 1
    }
  }

  def generateNegatives(originSet: Seq[(Int, Set[Int])],
                        buffer: Array[Float],
                        trainNeg: Int,
                        processes: Int): Unit = {
    val size = Math.ceil(originSet.size / processes).toInt
    val lastOffset = originSet.size - size * (processes - 1)
    val processesOffset = Array.tabulate[Int](processes)(_ * size)

    val numItems = processesOffset.map{offset =>
      val length = if(offset == lastOffset) {
        originSet.length - offset
      } else {
        size
      }
      var numItem = 0
      var i = 0
      while (i < length) {
        numItem += originSet(i + offset)._2.size
        i += 1
      }
      (length, offset, numItem)
    }

    val numItemAndOffset = (0 to processes).map{p =>
      (numItems(p)._1, numItems(p)._2,
        numItems(p)._3, numItems.slice(0, p + 1).map(_._3).sum)
    }

    numItemAndOffset.foreach{v =>
      val rand = new Random(System.currentTimeMillis())
      val length = v._1
      var offset = v._2
      val numItem = v._3
      var itemOffset = v._4

      while(offset < v._2 + length) {
        val userId = originSet(offset)._1
        val items = originSet(offset)._2



        offset += 1
      }

    }




  }
}
