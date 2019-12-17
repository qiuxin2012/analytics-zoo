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

package com.intel.analytics.zoo.feature

import java.nio.file.Paths
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.{AbstractDataSet, DistributedDataSet, MiniBatch, Transformer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.zoo.feature.common.{ArrayLike, ArrayLikeWrapper}
import com.intel.analytics.zoo.feature.pmem._
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.api.net.{PytorchModel, PytorchModelWrapper}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.slf4j.{Logger, LoggerFactory}
import jep._

import scala.reflect.ClassTag

/**
 * A set of data which is used in the model optimization process. The FeatureSet can be access in
 * a random data sample sequence. In the training process, the data sequence is a looped endless
 * sequence. While in the validation process, the data sequence is a limited length sequence.
 * User can use the data() method to get the data sequence.
 *
 * The sequence of the data is not fixed. It can be changed by the shuffle() method.
 *
 * @tparam D Data type
 * @tparam DataSequence Represent a sequence of data
 */
trait AbstractFeatureSet[D, DataSequence] extends AbstractDataSet[D, DataSequence]{
  /**
   * Get a sequence of data
   *
   * @param train if the data is used in train. If yes, the data sequence is a looped endless
   *              sequence, or it has a limited length.
   * @return data sequence
   */
  def data(train: Boolean): DataSequence

  /**
   * Change the order of the data sequence from the data set
   */
  def shuffle(): Unit

  /**
   * Total size of the data set
   * @return
   */
  def size(): Long

  /**
   * Helper function to transform the data type in the data set.
   * @param transformer
   * @tparam C
   * @return
   */
  override def transform[C: ClassTag](transformer: Transformer[D, C]): FeatureSet[C]

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  /**
   * Helper function to transform the data type in the data set.
   *
   * @param transformer
   * @tparam C
   * @return
   */
  override def -> [C: ClassTag](transformer: Transformer[D, C]): FeatureSet[C] = {
    this.transform(transformer)
  }

  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName

  /**
   * Convert FeatureSet to com.intel.analytics.bigdl.DataSet
   * @return DataSet[D]
   */
  def toDataSet(): DataSet[D]
}

/**
 * Represent a distributed data. Use RDD to go through all data.
 *
 * @tparam T
 */
trait DistributedFeatureSet[T] extends AbstractFeatureSet[T, RDD[T]] {
  def numOfSlice: Int = 1

  override def transform[C: ClassTag](transformer: Transformer[T, C]): DistributedFeatureSet[C] = {
    val preFeatureSet = this

    val broadcast = this.originRDD().sparkContext.broadcast(transformer)

    val cachedTransformer =
      preFeatureSet.originRDD().mapPartitions(_ => Iterator
        .single(broadcast.value.cloneTransformer())
      ).setName(s"Cached Transformer of ${preFeatureSet.originRDD().name}").persist()

    new DistributedFeatureSet[C] {
      originFeatureSet = preFeatureSet.originFeatureSet

      override def numOfSlice: Int = originFeatureSet.numOfSlice

      override def size(): Long = preFeatureSet.size()

      override def shuffle(): Unit = preFeatureSet.shuffle()

      override def data(train: Boolean): RDD[C] =
        preFeatureSet.data(train).zipPartitions(cachedTransformer)(
          (data, tran) => tran.next()(data))

      override def originRDD(): RDD[_] = preFeatureSet.originRDD()

      override def cache(): Unit = {
        cachedTransformer.count()
        isCached = true
      }

      override def unpersist(): Unit = {
        preFeatureSet.unpersist()
        cachedTransformer.unpersist()
        isCached = false
      }

      override def toDistributed(): DistributedDataSet[C] = {
        new DistributedDataSetWrapper[C](this)
      }
    }
  }

  /**
   * Get the 'origin' RDD of the dataset.
   *
   * @return
   */
  def originRDD(): RDD[_]

  /**
   * Trigger the computation of this dataset and cache it in memory.
   */
  def cache(): Unit = {
    if (originRDD() != null) {
      originRDD().count()
    }
    isCached = true
  }

  /**
   * Unpersist rdd.
   */
  def unpersist(): Unit = {
    if (originRDD() != null) {
      originRDD().unpersist()
      isCached = false
    }
  }

  protected var originFeatureSet: DistributedFeatureSet[Any] =
    this.asInstanceOf[DistributedFeatureSet[Any]]

  /**
   * Check if rdd is cached.
   */
  var isCached = false

  override def toDataSet(): DataSet[T] = {
    toDistributed()
  }
}

/**
 * Wrap a featureSet to DistributedDataSet.
 * @param featureSet
 * @tparam T
 */
private[zoo] class DistributedDataSetWrapper[T: ClassTag](featureSet: DistributedFeatureSet[T])
  extends DistributedDataSet[T]{

  override def data(train: Boolean): RDD[T] = {
    featureSet.data(train)
  }

  override def size(): Long = featureSet.size()

  override def shuffle(): Unit = {
    featureSet.shuffle()
  }

  override def originRDD(): RDD[_] = featureSet.originRDD()

  override def cache(): Unit = {
    featureSet.cache()
  }

  override def unpersist(): Unit = {
    featureSet.unpersist()
  }

}

/**
 * Wrap a RDD as a FeatureSet.
 * @param buffer
 */
// T is the returning value type. like ByteRecord
class CachedDistributedFeatureSet[T: ClassTag]
(buffer: RDD[ArrayLike[T]],
 sequentialOrder: Boolean = false,
 shouldShuffle: Boolean = true)
  extends DistributedFeatureSet[T]{

  protected lazy val count: Long = buffer.mapPartitions(iter => {
    require(iter.hasNext)
    val array = iter.next()
    require(!iter.hasNext)
    Iterator.single(array.length)
  }).reduce(_ + _)

  protected var indexes: RDD[(Array[Int], AtomicInteger)] = buffer.mapPartitions(iter => {
    Iterator.single[(Array[Int], AtomicInteger)](((0 until iter.next().length).toArray[Int],
      new AtomicInteger(0)))
  }).setName(s"origin index of ${buffer.name}").cache()

  override def data(train: Boolean): RDD[T] = {
    val _train = train
    val _seq = sequentialOrder
    buffer.zipPartitions(indexes)((dataIter, indexIter) => {
      val (indexes, seqOffset) = indexIter.next()


      val maxOffset = math.max(1, indexes.length)
      val localData = dataIter.next()
      val offset = if (_train && !_seq) {
        RandomGenerator.RNG.uniform(0, maxOffset).toInt
      } else if (_train && _seq) {
        seqOffset.get()
      } else {
        0
      }
      seqOffset.set(offset)

      new Iterator[T] {
        private val _offset = seqOffset

        override def hasNext: Boolean = {
          if (_train) true else _offset.get() < localData.length
        }

        override def next(): T = {
          val i = _offset.getAndIncrement()
          if (_train && i >= localData.length) {
            this.synchronized {
              val value = _offset.get()
              if (value >= localData.length) {
                _offset.set(value % localData.length)
              }
            }
          }
          if (_train) {
            // indexes is an Array, we should improve this
            // as the maximum length is limited by Int.max
            localData(indexes(i % localData.length))
          } else {
            if (i < localData.length) {
              localData(indexes(i))
            } else {
              null.asInstanceOf[T]
            }
          }
        }
      }
    })
  }

  override def size(): Long = count

  override def shuffle(): Unit = {
    if (shouldShuffle) {
      indexes.unpersist()
      indexes = buffer.mapPartitions(iter => {
        Iterator.single((RandomGenerator.shuffle((0 until iter.next().length).toArray),
          new AtomicInteger(0)))
      }).setName(s"shuffled index of ${buffer.name}").cache()
    }
  }

  override def originRDD(): RDD[_] = buffer

  override def cache(): Unit = {
    buffer.cache().count()
    indexes.cache().count()
    isCached = true
  }

  override def unpersist(): Unit = {
    FeatureSet.logger.info(s"Unpersisting ${buffer.name}.")
    buffer.map(_.free()).count()
    buffer.unpersist()
    indexes.unpersist()
    isCached = false
  }

  override def toDistributed(): DistributedDataSet[T] = {
    new DistributedDataSetWrapper[T](this)
  }
}

object PythonLoaderFeatureSet{
  def loadPytorchLoader(
      loaderName: String,
      dataset: Array[Byte],
      interpRdd: RDD[SharedInterpreter]): Unit = {
    val bcDataSet = interpRdd.sparkContext.broadcast(dataset)
    val imports = s"""
      |import numpy as np
      |from torchvision import datasets, transforms
      |import torch
      |import pickle
      |""".stripMargin
    val load = s"""
      |by = bytes(b % 256 for b in pyjarray)
      |${loaderName} = pickle.loads(by)
      |""".stripMargin
    interpRdd.mapPartitions{iter =>
      val interp = iter.next()
      interp.exec(imports)
      interp.set("pyjarray", bcDataSet.value)
      interp.exec(load)
      interp.exec("by = bytes(b % 256 for b in pyjarray)")
      interp.exec("loader = pickle.loads(by)")
      Iterator.single(interp)
    }.count()

  }

  private var sharedInterpRDD: RDD[SharedInterpreter] = null
  def getOrCreateInterpRdd(): RDD[SharedInterpreter] = {
    if (sharedInterpRDD == null) {
      this.synchronized {
        if (sharedInterpRDD == null) {
          println("creating interp RDD")
          val sc = SparkContext.getOrCreate()
          val nodeNumber = EngineRef.getNodeNumber()
          // TODO: make sure 1 executor 1 partition
          val originRdd = sc.parallelize(
            Array.tabulate(nodeNumber)(_ => "123123"), nodeNumber * 10)
            .mapPartitions(_ => (0 until 100000).toIterator)
            .coalesce(nodeNumber)
            .setName("PartitionRDD")
            .cache()
          originRdd.count()
          originRdd.count()
          // load pytorch library before jep, or libCaffe2.so will conflict.
          originRdd.mapPartitions{
            _ => PytorchModelWrapper.load()
            Iterator.single(1)
          }.count()
          sharedInterpRDD = originRdd.mapPartitions { iter =>
            val interp = getOrCreateInterpreter()
            Iterator.single(interp)
          }.setName("SharedInterpRDD").cache()
          sharedInterpRDD.count()
        }
      }
    }
    sharedInterpRDD
  }

  private var sharedInterpreter: SharedInterpreter = null
  def getOrCreateInterpreter(): SharedInterpreter = {
    if (sharedInterpreter == null) {
      this.synchronized {
        if (sharedInterpreter == null) {
          sharedInterpreter = new SharedInterpreter()
        }
      }
    }
    sharedInterpreter
  }
}

class PythonLoaderFeatureSet[T: ClassTag](
    dataset: Array[Byte],
    inputs: Array[String],
    outputs: Array[String]) extends DistributedFeatureSet[T] {
  import PythonLoaderFeatureSet._
  protected val namePostfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode())
  protected val loaderName = s"loader${namePostfix}"
  println(dataset.length)

  val sharedInterp = getOrCreateInterpRdd()
  loadPytorchLoader(loaderName, dataset, sharedInterp)
  override def originRDD(): RDD[_] = {
    sharedInterp
  }

  override def data(train: Boolean): RDD[T] = {
    val loaderName = this.loaderName
    val iterName = s"${loaderName}_iter"
    if (train) {
      sharedInterp.mapPartitions{dataIter =>
        val interp = dataIter.next()
        new Iterator[T] {
          val nextCode =
            s"""
               |batch_idx, (data, target) = next($iterName)
               |""".stripMargin

          override def hasNext: Boolean = {
            true
          }

          override def next(): T = {
            val stat = System.nanoTime()
            try {
              interp.exec(nextCode)
            } catch {
              case e: Exception =>
                if(e.getMessage().contains("StopIteration")) {
                  interp.exec(s"${iterName} = enumerate($loaderName)")
                  interp.exec(nextCode)
                }
            }
            val input = interp.getValue("data.numpy()").asInstanceOf[NDArray[Array[Float]]]
            val target = interp.getValue("target.numpy()").asInstanceOf[NDArray[Array[Long]]]
            val r = MiniBatch[Float](Tensor[Float](input.getData, input.getDimensions),
              Tensor[Float](target.getData().map(_.toFloat), target.getDimensions)
            ).asInstanceOf[T]
            println(s"${loaderName} next cost ${(System.nanoTime() - stat) / 1e9} s")
            r
          }
        }
      }
    } else {
      sharedInterp.mapPartitions{ dataIter =>
        val interp = dataIter.next()
        val len = interp.getValue(s"len($loaderName)").asInstanceOf[Long]
        interp.exec(s"${iterName} = enumerate($loaderName)")
        new Iterator[T] {
          var i = 0
          val nextCode =
            s"""
               |batch_idx, (data, target) = next($iterName)
               |""".stripMargin

          override def hasNext: Boolean = {
            i < len
          }

          override def next(): T = {
            val stat = System.nanoTime()
            i += 1
            interp.exec(nextCode)
            val input = interp.getValue("data.numpy()").asInstanceOf[NDArray[Array[Float]]]
            val target = interp.getValue("target.numpy()").asInstanceOf[NDArray[Array[Long]]]
            val r = MiniBatch[Float](Tensor[Float](input.getData, input.getDimensions),
              Tensor[Float](target.getData().map(_.toFloat), target.getDimensions)
            ).asInstanceOf[T]
            println(s"${loaderName} next cost ${(System.nanoTime() - stat) / 1e9} s")
            r
          }
        }

      }
    }

  }

  override def shuffle(): Unit = {

  }

  override def size(): Long = {
    data(false).count()
  }

  override def toDistributed(): DistributedDataSet[T] = {
    new DistributedDataSetWrapper[T](this)
  }
}

/**
 * Wrap a RDD as a FeatureSet. RDD will be persist on local disk, and will load
 * one slice of the data to memory for the training.
 * @param origin cached rdd
 * @param numSlice number of RDD slice. During the training, only 1/numSlice of
 *                 originRDD is loaded into memory.
 */
// T is the returning value type. like ByteRecord
class DiskFeatureSet[T: ClassTag]
(origin: RDD[T], val numSlice: Int)
  extends DistributedFeatureSet[T]{
  require(numSlice != 1,
    s"Detected numSlice = 1, Please use MemoryType DRAM to " +
      s"cache all data into memory.")

  require(numSlice == 0 || numSlice >= 2,
    s"excepted numSlice == 0 or >= 2, but got $numSlice")

  override def numOfSlice: Int = numSlice

  protected val buffer = origin.coalesce(EngineRef.getNodeNumber(), true)
    .persist(StorageLevel.DISK_ONLY)
    .setName("Origin Data Cached on Disk")
  protected lazy val count: Long = buffer.count()

  protected var currentSlice: RDD[T] = null
  protected var currentFeatureSet: DistributedFeatureSet[T] = null
  protected var trained: Boolean = false
  if (numSlice != 0) {
    newSample()
  }

  private def newSample() = {
    currentSlice = buffer.sample(false, 1.0 / numSlice)
      .setName(s"1/${numSlice} of ${origin.name}")
    currentFeatureSet = DRAMFeatureSet.rdd(currentSlice)
    trained = false
  }

  override def data(train: Boolean): RDD[T] = {
    if (numSlice == 0) {
      if (train) {
        throw new IllegalArgumentException("No training data in memory," +
          "because numSlice is zero. numSlice should >= 2 " +
          "in a training FeatureSet.")
      } else {
        buffer
      }
    } else {
      if (train) {
        if (trained) {
          if (currentFeatureSet != null) {
            currentFeatureSet.unpersist()
          }
          newSample()
        }
        currentFeatureSet.shuffle()
        trained = true
        currentFeatureSet.data(train)
      } else {
        trained = false
        currentFeatureSet.data(train)
      }
    }
  }

  override def size(): Long = count

  override def shuffle(): Unit = {
  }

  override def originRDD(): RDD[_] = buffer

  override def cache(): Unit = {
    buffer.persist(StorageLevel.DISK_ONLY)
    buffer.count()
  }

  override def unpersist(): Unit = {
    buffer.unpersist()
  }

  override def toDistributed(): DistributedDataSet[T] = {
    new DistributedDataSetWrapper[T](this)
  }
}

object DRAMFeatureSet {
  def rdd[T: ClassTag](data: RDD[T],
                       sequentialOrder: Boolean = false,
                       shuffle: Boolean = true): DistributedFeatureSet[T] = {
    val arrayLikeRDD = data.mapPartitions(iter => {
      Iterator.single(new ArrayLikeWrapper(iter.toArray))
    }).setName(s"cached feature set: ${data.name} in DRAM" )
      .cache().asInstanceOf[RDD[ArrayLike[T]]]
    new CachedDistributedFeatureSet[T](arrayLikeRDD, sequentialOrder, shuffle)
  }
}

object FeatureSet {
  val logger: Logger = LoggerFactory.getLogger(this.getClass)
  private[zoo] def python[T: ClassTag](
      dataset: Array[Byte],
      inputs: Array[String],
      targets: Array[String]): PythonLoaderFeatureSet[T] = {
    new PythonLoaderFeatureSet[T](dataset, inputs, targets)
  }

  def rdd[T: ClassTag](
       data: RDD[T],
       memoryType: MemoryType = DRAM,
       dataStrategy: DataStrategy = PARTITIONED,
       sequentialOrder: Boolean = false,
       shuffle: Boolean = true): DistributedFeatureSet[T] = {
    dataStrategy match {
      case PARTITIONED =>
        val nodeNumber = EngineRef.getNodeNumber()
        val repartitionedData = data.coalesce(nodeNumber, true).setName(data.name)
        memoryType match {
          case DRAM =>
            DRAMFeatureSet.rdd(repartitionedData, sequentialOrder, shuffle)
          case PMEM =>
            logger.info("~~~~~~~ Caching with AEP ~~~~~~~")
            PmemFeatureSet.rdd(repartitionedData, PMEM, sequentialOrder, shuffle)
          case DIRECT =>
            logger.info("~~~~~~~ Caching with DIRECT ~~~~~~~")
            PmemFeatureSet.rdd[T](repartitionedData, DIRECT, sequentialOrder, shuffle)
          case diskM: DISK_AND_DRAM =>
            logger.info(s"~~~~~~~ Caching with DISK_AND_DRAM(${diskM.numSlice}) ~~~~~~~")
            if (sequentialOrder) {
              throw new IllegalArgumentException("DiskFeatureSet does not support" +
                " sequentialOrder.")
            }

            if (!shuffle) {
              throw new IllegalArgumentException("DiskFeatureSet must use shuffle.")
            }
            new DiskFeatureSet[T](data, diskM.numSlice)
          case _ =>
            throw new IllegalArgumentException(
              s"MemoryType: ${memoryType} is not supported at the moment")
        }

      case _ =>
        throw new IllegalArgumentException(
          s"DataStrategy ${dataStrategy} is not supported at the moment")

    }
  }
}
