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

package com.intel.analytics.zoo.examples.inception

import java.net.URI
import java.nio.ByteBuffer

import com.intel.analytics.bigdl.dataset.DataSet.SeqFileFolder
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{BGRImgCropper, BGRImgNormalizer, BytesToBGRImg, CropCenter, MTLabeledBGRImgToBatch, HFlip => DatasetHFlip}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, T}
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.feature.{DistributedFeatureSet, FeatureSet}
import com.intel.analytics.zoo.feature.pmem._
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileStatus, FileSystem, Path}
import org.apache.hadoop.io.SequenceFile.Reader
import org.apache.hadoop.io.{SequenceFile, Text}
import org.apache.hadoop.mapred.{FileSplit, SequenceFileInputFormat, SequenceFileRecordReader}
import org.apache.log4j.Logger
import org.apache.spark.SparkContext

object ImageNet2012 {
  val logger = Logger.getLogger(this.getClass)

  /**
   * Extract hadoop sequence files from an HDFS path
   *
   * @param url
   * @param sc
   * @param classNum
   * @return
   */
  private[inception] def readFromSeqFiles(
        url: String, sc: SparkContext, classNum: Int) = {
    val nodeNumber = EngineRef.getNodeNumber()
    val coreNumber = EngineRef.getCoreNumber()
    val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text],
      nodeNumber * coreNumber).map(image => {
      ByteRecord(image._2.copyBytes(), readLabel(image._1).toFloat)
    }).filter(_.label <= classNum)
    rawData
  }

  private[inception] def readFromSeqFilesReplicated(
        url: String, sc: SparkContext, classNum: Int) = {
    val nodeNumber = EngineRef.getNodeNumber()
    val coreNumber = EngineRef.getCoreNumber()

    val path = new Path(url)
    val fs = FileSystem.get(path.toUri, new Configuration())
    val fileNames = fs.listStatus(path).map(v => (v.getPath.toUri, v.getBlockSize, v.getLen))
    val bcFileNames = sc.broadcast(fileNames)
    val uris = sc.range(0, nodeNumber, 1, nodeNumber)
      .coalesce(nodeNumber, true)
      .mapPartitions{_ =>
      Iterator.single(bcFileNames.value.clone())
    }.flatMap(v => RandomGenerator.shuffle(v)).
      flatMap{v =>
//      map{v =>
        val uri = v._1
        val blockSize = v._2
        val fileLength = v._3
        val fileNum = Math.floor(fileLength.toDouble / blockSize).toLong
//      val conf = new Configuration()

//      val inputFormat = new SequenceFileInputFormat[Text, Text]()
//      inputFormat.getSplits(conf, fileNum.toInt)
        val splits = Array.tabulate(fileNum.toInt)(i =>
//          if(i != fileNum - 1) {
            new FileSplit(new Path(uri), i * blockSize, blockSize, Array[String]())
//          } else {
//            new FileSplit(file.getPath, i * file.getBlockSize, file.getBlockSize)
//          }
        )
        if(fileLength % blockSize == 0) {
          splits
        } else {
          splits ++
            Array(new FileSplit(new Path(uri), fileNum * blockSize,
             fileLength - fileNum * blockSize, Array[String]()))
        }
//        new FileSplit(new Path(uri), 0, fileLength, Array[String]())
      }
      .cache().setName("cached uris")
    uris.count()
//    uris.map{uri =>
//      val toByteRecord = SeqFileToBytes()
//      toByteRecord(Iterator.single(uri))
//    }
    val toByteRecord = SeqFileToBytes()
    val result = toByteRecord.apply(uris)
    logger.info(uris.map{uri =>
      val toByteRecord = SeqFileToBytes()
      toByteRecord.apply(Iterator.single(uri))
    }.mapPartitions{v =>
      val iteratorArray = v.toArray
      val nativeArrayConverter = iteratorArray.map(iter =>
        (iter, new ByteRecordConverter())).par
      val record = nativeArrayConverter.map{v =>
        var totalBytes: Long = 0L
        var totalRecordNum = 0L
        v._1.foreach{ record =>
          totalRecordNum += 1
          totalBytes += v._2.getBytesPerRecord(record)
        }
        (totalRecordNum, totalBytes)
      }
      record.toIterator
    }.reduce((a, b) => (a._1 + b._1, a._2 + b._2)))
    result.count()
    result
  }

  /**
   * get label from text of sequence file,
   *
   * @param data text of sequence file, this text can split into parts by "\n"
   * @return
   */
  private def readLabel(data: Text): String = {
    val dataArr = data.toString.split("\n")
    if (dataArr.length == 1) {
      dataArr(0)
    } else {
      dataArr(1)
    }
  }

  def apply(
    path : String,
    sc: SparkContext,
    imageSize : Int,
    batchSize : Int,
    nodeNumber: Int,
    coresPerNode: Int,
    classNumber: Int,
    memoryType: MemoryType = DRAM,
    opencvPreprocessing: Boolean = false,
    replicated: Boolean = false
  )
  : FeatureSet[MiniBatch[Float]] = {
    if (opencvPreprocessing) {
      logger.info("Using opencv preprocessing for training set")
      opencv(path, sc, imageSize, batchSize,
        nodeNumber, coresPerNode, classNumber, memoryType, replicated)
    } else {
      val featureSet = if (replicated) {
        logger.info("Replicated dataset  aaa")
        println(readFromSeqFiles(path, sc, classNumber).count())
        val rawData = readFromSeqFilesReplicated(path, sc, classNumber)
          .setName("Replicated ImageNet2012 Training Set")
        FeatureSet.rdd(rawData, memoryType = memoryType, REPLICATED)
      } else {
        val rawData = readFromSeqFiles(path, sc, classNumber)
          .setName("ImageNet2012 Training Set")
        FeatureSet.rdd(rawData, memoryType = memoryType, PARTITIONED)
      }
      featureSet.cache()
      featureSet.transform(
        MTLabeledBGRImgToBatch[ByteRecord](
          width = imageSize,
          height = imageSize,
          batchSize = batchSize,
          transformer = (BytesToBGRImg()
            -> BGRImgCropper(imageSize, imageSize)
            -> DatasetHFlip(0.5)
            -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225))
        ))
    }
  }

  private[inception] def byteRecordToImageFeature(record: ByteRecord): ImageFeature = {
    val rawBytes = record.data
    val label = Tensor[Float](T(record.label))
    val imgBuffer = ByteBuffer.wrap(rawBytes)
    val width = imgBuffer.getInt
    val height = imgBuffer.getInt
    val bytes = new Array[Byte](3 * width * height)
    System.arraycopy(imgBuffer.array(), 8, bytes, 0, bytes.length)
    val imf = ImageFeature(bytes, label)
    imf(ImageFeature.originalSize) = (height, width, 3)
    imf
  }

  def opencv(
        path : String,
        sc: SparkContext,
        imageSize : Int,
        batchSize : Int,
        nodeNumber: Int,
        coresPerNode: Int,
        classNumber: Int,
        memoryType: MemoryType = DRAM,
        replicated: Boolean = false): FeatureSet[MiniBatch[Float]] = {
    val featureSet = if (replicated) {
      logger.info("Replicated dataset")
      val rawData = readFromSeqFilesReplicated(path, sc, classNumber)
        .map(byteRecordToImageFeature(_))
        .setName("Replicated ImageNet2012 Training Set")
      FeatureSet.rdd(rawData, memoryType = memoryType, REPLICATED)
    } else {
      val rawData = readFromSeqFiles(path, sc, classNumber)
        .map(byteRecordToImageFeature(_))
        .setName("ImageNet2012 Training Set")
      FeatureSet.rdd(rawData, memoryType = memoryType, PARTITIONED)
    }
    val transformer = ImagePixelBytesToMat() ->
      ImageRandomCrop(imageSize, imageSize) ->
      ImageChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      ImageMatToTensor[Float](true) ->
      ImageSetToSample[Float](inputKeys = Array(ImageFeature.imageTensor),
        targetKeys = Array(ImageFeature.label)) ->
      ImageFeatureToSample[Float]() ->
      SampleToMiniBatch[Float](batchSize)
    featureSet.transform(transformer)
  }
}


object ImageNet2012Val {
  val logger = Logger.getLogger(this.getClass)

  def apply(
    path : String,
    sc: SparkContext,
    imageSize : Int,
    batchSize : Int,
    nodeNumber: Int,
    coresPerNode: Int,
    classNumber: Int,
    memoryType: MemoryType = DRAM,
    opencvPreprocessing: Boolean = false
  ): FeatureSet[MiniBatch[Float]] = {
    if (opencvPreprocessing) {
      logger.info("Using opencv preprocessing for validation set")
      opencv(path, sc, imageSize, batchSize,
        nodeNumber, coresPerNode, classNumber, memoryType)
    } else {
      val rawData = ImageNet2012.readFromSeqFiles(path, sc, classNumber)
        .setName("ImageNet2012 Validation Set")
      val featureSet = FeatureSet.rdd(rawData, memoryType = memoryType)
      featureSet.transform(
        MTLabeledBGRImgToBatch[ByteRecord](
          width = imageSize,
          height = imageSize,
          batchSize = batchSize,
          transformer = (BytesToBGRImg()
            -> BGRImgCropper(imageSize, imageSize, CropCenter)
            -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225))
        ))
    }
  }

  def opencv(
        path : String,
        sc: SparkContext,
        imageSize : Int,
        batchSize : Int,
        nodeNumber: Int,
        coresPerNode: Int,
        classNumber: Int,
        memoryType: MemoryType = DRAM): FeatureSet[MiniBatch[Float]] = {
    val rawData = ImageNet2012.readFromSeqFiles(path, sc, classNumber)
      .map(ImageNet2012.byteRecordToImageFeature(_))
      .setName("ImageNet2012 Validation Set")
    val featureSet = FeatureSet.rdd(rawData, memoryType = memoryType)
    val transformer = ImagePixelBytesToMat() ->
      ImageCenterCrop(imageSize, imageSize) ->
      ImageChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      ImageMatToTensor[Float](true) ->
      ImageSetToSample[Float](inputKeys = Array(ImageFeature.imageTensor),
        targetKeys = Array(ImageFeature.label)) ->
      ImageFeatureToSample[Float]() -> SampleToMiniBatch[Float](batchSize)
    featureSet.transform(transformer)
  }

}

object SeqFileToBytes {
  def apply(): SeqFileToBytes = new SeqFileToBytes()
  val logger = Logger.getLogger(this.getClass)
}

/**
 * Read byte records from local hadoop sequence files.
 */
class SeqFileToBytes extends Transformer[FileSplit, ByteRecord] {

  @transient
  private var key: Text = null

  @transient
  private var value: Text = null

  @transient
  private var reader: SequenceFileRecordReader[Text, Text] = null

  @transient
  private var oneRecordBuffer: ByteRecord = null

  override def apply(prev: Iterator[FileSplit]): Iterator[ByteRecord] = {
    new Iterator[ByteRecord] {
      override def next(): ByteRecord = {
        if (oneRecordBuffer != null) {
          val res = oneRecordBuffer
          oneRecordBuffer = null
          return res
        }

        if (key == null) {
          key = new Text()
        }
        if (value == null) {
          value = new Text
        }
        if (reader == null || !reader.next(key, value)) {
          if (reader != null) {
            reader.close()
          }

          val path = prev.next()
          SeqFileToBytes.logger.info(s"Loading ${path.getPath}:${path.getStart}+${path.getLength}")
          reader = new SequenceFileRecordReader[Text, Text](new Configuration(), path)
//            new SequenceFile.Reader(new Configuration,
//            Reader.file(path.getPath),
//            Reader.bufferSize(10485760))

          reader.next(key, value)
        }

        ByteRecord(value.copyBytes(), SeqFileFolder.readLabel(key).toFloat)
      }

      override def hasNext: Boolean = {
        if (oneRecordBuffer != null) {
          true
        } else if (reader == null) {
          prev.hasNext
        } else {
          if (reader.next(key, value)) {
            oneRecordBuffer = ByteRecord(value.copyBytes(),
              SeqFileFolder.readLabel(key).toFloat)
            return true
          } else {
            prev.hasNext
          }
        }
      }
    }
  }
}
