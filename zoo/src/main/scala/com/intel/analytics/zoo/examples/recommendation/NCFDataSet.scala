package com.intel.analytics.zoo.examples.recommendation

import java.util
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator}

import scala.util.Random

class NCFDataSet (
    trainSet: Seq[(Int, Set[Int])],
    valPos: Map[Int, Int],
    trainNegatives: Int,
    batchSize: Int,
    userCount: Int,
    itemCount: Int) extends LocalDataSet[MiniBatch[Float]] {

  val trainSize = trainSet.map(_._2.size).sum
  // origin set use to random train negatives
  val originSet = trainSet.map(v => (v._1, v._2 + valPos(v._1)))

  val trainPositiveBuffer = new Array[Float](trainSize * 2)
  NCFDataSet.copy(trainSet, trainPositiveBuffer)

  val inputBuffer = new Array[Float](trainSize * (1 + trainNegatives) * 2 )
  val labelBuffer = new Array[Float](trainSize * (1 + trainNegatives))

  override def shuffle(): Unit = {
    NCFDataSet.generateNegativeItems(originSet,
                              inputBuffer,
                              trainNegatives,
      4, // TODO
                              itemCount)
    System.arraycopy(trainPositiveBuffer, 0, inputBuffer,
      trainSize * trainNegatives * 2, trainSize * 2)
    util.Arrays.fill(labelBuffer, 0, trainSize * trainNegatives, 0)
    util.Arrays.fill(labelBuffer, trainSize * trainNegatives, trainSize * (1 + trainNegatives), 1)
    NCFDataSet.shuffle(inputBuffer, labelBuffer)
  }

  override def data(train: Boolean): Iterator[MiniBatch[Float]] = {
    new Iterator[MiniBatch[Float]] {
      val input = Tensor[Float](batchSize, 2)
      val label = Tensor[Float](batchSize, 1)
      val miniBatch = MiniBatch(Array(input), Array(label))

      private val index = new AtomicInteger()
      private val numOfSample = inputBuffer.length / 2
      private val numMiniBatch = math.ceil(numOfSample.toFloat / batchSize).toInt

      override def hasNext: Boolean = {
        index.get() < inputBuffer.length
      }

      override def next(): MiniBatch[Float] = {
        val curIndex = index.getAndIncrement()  % numMiniBatch
        if (curIndex < numMiniBatch - 1) {
          System.arraycopy(inputBuffer, curIndex * 2 * batchSize,
            input.storage().array(), 0, batchSize * 2)
          System.arraycopy(labelBuffer, curIndex * batchSize,
            label.storage().array(), 0, batchSize)
          miniBatch
        } else if (curIndex == numMiniBatch - 1) {
          // TODO
          val restItem = numOfSample - curIndex * batchSize
          System.arraycopy(inputBuffer, curIndex * 2 * batchSize,
            input.storage().array(), 0, restItem * 2)
          System.arraycopy(labelBuffer, curIndex * batchSize,
            label.storage().array(), 0, restItem)
          input.resize(restItem, 2)
          label.resize(restItem, 1)
          miniBatch
        } else {
          null
        }
      }
    }
  }

  override def size(): Long = inputBuffer.length / 2
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

  def shuffle(inputBuffer: Array[Float], labelBuffer: Array[Float]): Unit = {
    var i = 0
    val length = inputBuffer.length / 2
    while (i < length) {
      val exchange = RandomGenerator.RNG.uniform(0, length - i).toInt + i
      val tmp1 = inputBuffer(exchange * 2)
      val tmp2 = inputBuffer(exchange * 2 + 1)
      inputBuffer(exchange * 2) = inputBuffer(i * 2)
      inputBuffer(exchange * 2 + 1) = inputBuffer(i * 2 + 1)
      inputBuffer(2 * i) = tmp1
      inputBuffer(2 * i + 1) = tmp2

      val labelTmp = labelBuffer(exchange)
      labelBuffer(exchange) = labelBuffer(i)
      labelBuffer(i) = labelTmp
      i += 1
    }
  }

  def generateNegativeItems(originSet: Seq[(Int, Set[Int])],
                        buffer: Array[Float],
                        trainNeg: Int,
                        processes: Int,
                        itemCount: Int): Unit = {
    val size = Math.ceil(originSet.size / processes).toInt
    val lastOffset = size * (processes - 1)
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
        numItem += originSet(i + offset)._2.size - 1 // discard validation positive
        i += 1
      }
      (length, offset, numItem)
    }

    val numItemAndOffset = (0 until processes).map{p =>
      (numItems(p)._1, numItems(p)._2,
        numItems(p)._3, numItems.slice(0, p).map(_._3).sum)
    }

    numItemAndOffset.foreach{v =>
      val rand = new Random(System.nanoTime())
      val length = v._1
      var offset = v._2
      val numItem = v._3
      var itemOffset = v._4

      while(offset < v._2 + length) {
        val userId = originSet(offset)._1
        val items = originSet(offset)._2

        while(itemOffset < v._4 + numItem) {
          var i = 0
          while (i < trainNeg) {
            var negItem = rand.nextInt(itemCount) + 1
            while (items.contains(negItem)) {
              negItem = rand.nextInt(itemCount) + 1
            }
            val negItemOffset = itemOffset * 2 * trainNeg + i * 2
            buffer(negItemOffset) = userId
            buffer(negItemOffset + 1) = negItem

            i += 1
          }

          itemOffset += 1
        }

        offset += 1
      }
    }

  }
}
