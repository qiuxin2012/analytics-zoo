package com.intel.analytics.zoo.examples.mlperf.recommendation

import java.util
import java.util.concurrent.{Executors, ThreadFactory}
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.util.Random

class NCFDataSet (
    trainSet: Seq[(Int, Set[Int])],
    valPos: Map[Int, Int],
    trainNegatives: Int,
    batchSize: Int,
    userCount: Int,
    itemCount: Int,
    var seed: Int = 1,
    val processes: Int = 10) extends LocalDataSet[MiniBatch[Float]] {
  println(s"creating ncfDataset with ${processes} thread")

  val trainSize = trainSet.map(_._2.size).sum
  // origin set use to random train negatives
  val originSet = trainSet.map(v => (v._1, v._2 + valPos(v._1)))

  val trainPositiveBuffer = new Array[Float](trainSize * 2)
  NCFDataSet.copy(trainSet, trainPositiveBuffer)

  val inputBuffer = new Array[Float](trainSize * (1 + trainNegatives) * 2 )
  val labelBuffer = new Array[Float](trainSize * (1 + trainNegatives))

  override def shuffle(): Unit = {
    val start = System.currentTimeMillis()
    NCFDataSet.generateNegativeItems(originSet,
                              inputBuffer,
                              trainNegatives,
      processes, // TODO
                              itemCount,
      seed)
    println(s"gen neg time ${System.currentTimeMillis() - start} ms")
    System.arraycopy(trainPositiveBuffer, 0, inputBuffer,
      trainSize * trainNegatives * 2, trainSize * 2)
    util.Arrays.fill(labelBuffer, 0, trainSize * trainNegatives, 0)
    util.Arrays.fill(labelBuffer, trainSize * trainNegatives, trainSize * (1 + trainNegatives), 1)
    println(s"fill time cost ${System.currentTimeMillis() - start} ms")
    NCFDataSet.shuffle(inputBuffer, labelBuffer, seed, processes)
    println(s"shuffle time cost ${System.currentTimeMillis() - start} ms")
    seed += itemCount
    println(s"ncf dataset change seed to ${seed}")
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
        index.get() < numMiniBatch
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

  def shuffle(inputBuffer: Array[Float],
              labelBuffer: Array[Float],
              seed: Int,
              parallelism: Int): Unit = {
    val length = inputBuffer.length / 2
    val extraSize = length % parallelism
    val taskSize = math.floor(length / parallelism).toInt

    val seeds = Array.tabulate(parallelism)(i =>{
      val rand = new Random(seed + i)
      val length = if (i < extraSize) taskSize + 1 else taskSize
      (i, length, rand)
    }).par
    seeds.foreach{v =>
      val offset = v._1
      val length = v._2
      val rand = v._3
      var i = 0
      while(i < length) {
        val ex = rand.nextInt(length) * parallelism + offset
        val current = i * parallelism + offset
        if (ex != current) {
          exchange(inputBuffer, labelBuffer,
            current, ex)

        }
        i += 1
      }
    }
  }

  private def exchange(inputBuffer: Array[Float],
                       labelBuffer: Array[Float],
                       current: Int, exchange: Int): Unit = {
    val tmp1 = inputBuffer(exchange * 2)
    val tmp2 = inputBuffer(exchange * 2 + 1)
    inputBuffer(exchange * 2) = inputBuffer(current * 2)
    inputBuffer(exchange * 2 + 1) = inputBuffer(current * 2 + 1)
    inputBuffer(2 * current) = tmp1
    inputBuffer(2 * current + 1) = tmp2

    val labelTmp = labelBuffer(exchange)
    labelBuffer(exchange) = labelBuffer(current)
    labelBuffer(current) = labelTmp
  }

  def generateNegativeItems(originSet: Seq[(Int, Set[Int])],
                        buffer: Array[Float],
                        trainNeg: Int,
                        processes: Int,
                        itemCount: Int,
                        seed: Int): Unit = {
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
        numItems.slice(0, p).map(_._3).sum * trainNeg)
    }.par

    numItemAndOffset.foreach{ v =>
      val length = v._1
      var offset = v._2
      var itemOffset = v._3
      val rand = new Random(offset + seed)

      while (offset < v._2 + length) {
        val userId = originSet(offset)._1
        val items = originSet(offset)._2
        var i = 0
        while (i < (items.size - 1) * trainNeg) {
          var negItem = rand.nextInt(itemCount) + 1
          while (items.contains(negItem)) {
            negItem = rand.nextInt(itemCount) + 1
          }
          val negItemOffset = itemOffset * 2
          buffer(negItemOffset) = userId
          buffer(negItemOffset + 1) = negItem

          i += 1
          itemOffset += 1
        }
        offset += 1
      }
    }
  }
}
