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

package com.intel.analytics.zoo.examples.mlperf.recommendation

import java.io.{DataOutputStream, FileOutputStream}
import java.nio.ByteBuffer
import java.util

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.common.NNContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SQLContext}
import scopt.OptionParser

import scala.io.Source
import scala.util.{Random, Sorting}

object GenerateData {
  import NeuralCFexample._

  def main(args: Array[String]): Unit = {

    val defaultParams = NeuralCFParams()

    // run with ml-20m, please use
    val parser = new OptionParser[NeuralCFParams]("NCF Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[String]("dataset")
        .text(s"dataset, ml-20m or ml-1m, default is ml-1m")
        .action((x, c) => c.copy(dataset = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
      opt[Int]("trainNeg")
        .text("The Number of negative instances to pair with a positive train instance.")
        .action((x, c) => c.copy(trainNegtiveNum = x))
      opt[Int]("valNeg")
        .text("The Number of negative instances to pair with a positive validation instance.")
        .action((x, c) => c.copy(valNegtiveNum = x))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: NeuralCFParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("NCFExample").set("spark.sql.crossJoin.enabled", "true")
      .set("spark.driver.maxResultSize", "2048")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val inputDir = param.inputDir

    val (ratings, userCount, itemCount, itemMapping) =
      loadPublicData(sqlContext, param.inputDir, param.dataset)
    ratings.cache()
    println(s"${userCount} ${itemCount}")

    val (evalPos, trainSet, valSamples) = generateTrainValSet(ratings, userCount, itemCount,
      trainNegNum = param.trainNegtiveNum, valNegNum = param.valNegtiveNum)
    val ncfDataSet = new NCFDataSet(trainSet, evalPos,
      param.trainNegtiveNum, param.batchSize, userCount, itemCount)
    ncfDataSet.shuffle()
    val trainIterator = ncfDataSet.data(true)
    saveTrainToBinary(trainIterator, "/tmp/1234/")



    println()

  }

  def generateTrainValSet(
        rating: DataFrame,
        userCount: Int,
        itemCount: Int,
        trainNegNum: Int = 4,
        valNegNum: Int = 100): (Map[Int, Int], Array[(Int, Set[Int])],
    Array[Sample[Float]]) = {
//    val maxTimeStep = rating.groupBy("userId").max("timestamp").collect().map(r => (r.getInt(0), r.getInt(1))).toMap
//    val bcT = rating.sparkSession.sparkContext.broadcast(maxTimeStep)
    case class UserItem(userId: Int, ItemId: Int, timestamp: Int)
    trait UserItemOrdering extends Ordering[UserItem] {
      def compare(x: UserItem, y: UserItem): Int = x.timestamp - y.timestamp
    }
    implicit object UserItem extends UserItemOrdering

    val evalPos = rating.rdd.groupBy(_.getInt(0))
      .map(v => (v._1, v._2.toArray.map(row => UserItem(row.getInt(0), row.getInt(1), row.getInt(3)))))
      .map(v => {
        Sorting.quickSort(v._2)
        (v._1, v._2.last.ItemId)
      }).collect().toMap
//    // TODO: testing
//    val evalPos2 = rating.filter(r => bcT.value.apply(r.getInt(0)) == r.getInt(3)).rdd.groupBy(_.getInt(0))
//      .collect().sortBy(_._1)
//    val evalPos3 = rating.filter(r => bcT.value.apply(r.getInt(0)) == r.getInt(3)).rdd.groupBy(_.getInt(0))
//      .map(pos => (pos._1, pos._2.last.getInt(1)))
//      .collect().sortBy(_._1)

//    val positives = Source.fromFile("test-ratings.csv").getLines()
//    val evalPos = positives.toArray.map(_.split("\t")).map(v => (v(0).toInt + 1, v(1).toInt + 1)).toMap

    val groupedRdd = rating.rdd.groupBy(_.getAs[Int]("userId")).cache()
    val negRdd = groupedRdd.map{v =>
      val userId = v._1
      val items = scala.collection.mutable.Set(v._2.map(_.getAs[Int]("itemId")).toArray: _*)
      val gen = new Random(userId)
      var i = 0

      val negs = new Array[Int](valNegNum)
      // gen negative sample to validation
      while(i < valNegNum) {
        val negItem = gen.nextInt(itemCount) + 1
        if (!items.contains(negItem)) {
          negs(i) = negItem
          i += 1
        }
      }

      (userId, negs)
    }

    val trainSet = groupedRdd.map(v => (v._1, v._2.toArray.map(_.getInt(1))
      .filter(_ != evalPos(v._1)).toSet)).collect()

    val valSamples = negRdd.map(record => {
      val userId = record._1
      val negs = record._2
      val posItem = evalPos(userId)
      val distinctNegs = negs.distinct
      val testFeature = Tensor[Float](1 + negs.size, 2)
      testFeature.select(2, 1).fill(userId)
      val testLabel = Tensor[Float](1 + negs.size).fill(0)
      var i = 1
      while (i <= distinctNegs.size) {
        testFeature.setValue(i, 2, distinctNegs(i - 1))
        i += 1
      }
      testFeature.setValue(i, 2, posItem)
      testLabel.setValue(i, 1)
      testFeature.narrow(1, i + 1, negs.size - distinctNegs.size).fill(1)
      testLabel.narrow(1, i + 1, negs.size - distinctNegs.size).fill(-1)

      Sample(testFeature, testLabel)
    }).collect()

    (evalPos, trainSet, valSamples)
  }

  def saveTrainToBinary(data: Iterator[MiniBatch[Float]], des: String, batchSize: Int = 2048): Unit = {
    val startTime = System.currentTimeMillis()
    new java.io.File(des + s"/label").mkdirs()
    new java.io.File(des + s"/input").mkdirs()

    val labelBuffer = ByteBuffer.allocate(batchSize)
    val inputBuffer = ByteBuffer.allocate(batchSize * 4 * 2)

    var count = 0

    while(data.hasNext) {

      val labelFile = new DataOutputStream(new FileOutputStream(des + s"/label/${count}.obj"))
      val inputFile = new DataOutputStream(new FileOutputStream(des + s"/input/${count}.obj"))
      val batch = data.next()

      val input = batch.getInput().toTensor[Float]
      val target = batch.getTarget().toTensor[Float]
      val nElement = target.nElement()
      inputBuffer.asFloatBuffer().put(input.storage().array())
      labelBuffer.clear()
      labelBuffer.put(target.storage().array().map(_.toByte))
//      if (nElement == batchSize) {
//        inputFile.write(inputBuffer.array())
//        labelFile.write(labelBuffer.array())
//        inputFile.close()
//        labelFile.close()
//      } else{
//        inputFile.write(inputBuffer.array().slice(0, 2 * nElement * 4))
//        labelFile.write(labelBuffer.array().slice(0, nElement))
//        inputFile.close()
//        labelFile.close()
//      }

      count += target.nElement()
    }
    println(s"save $count record to ${des}: ${System.currentTimeMillis() - startTime}ms")
  }

//  def saveTestTobinary(evalSample: String,
//                                 des: String, numNegs: Int = 999): Unit = {
//    val startTime = System.currentTimeMillis()
//    val positives = Source.fromFile(posFile).getLines()
//    val negatives = Source.fromFile(negFile).getLines()
//    val testByteBuffer = ByteBuffer.allocate((numNegs + 2) * 4)
//    val testBuffer = testByteBuffer.asFloatBuffer()
//    (1 to numNegs + 1).foreach{i =>
//      testBuffer.put(i, 1)
//    }
//
//    new java.io.File(des).mkdirs()
//    while(positives.hasNext && negatives.hasNext) {
//      val pos = positives.next().split("\t")
//      val userId = pos(0).toFloat
//      val testFile = new DataOutputStream(new FileOutputStream(des + s"/${userId.toInt + 1}.obj"))
//      val posItem = pos(1).toFloat
//      val neg = negatives.next().split("\t").map(_.toFloat)
//      val distinctNegs = neg.distinct
//      testBuffer.put(0, distinctNegs.length + 1)
//
//      var i = 1
//      while (i <= distinctNegs.length) {
//        testBuffer.put(i, distinctNegs(i - 1) + 1)
//        i += 1
//      }
//      testBuffer.put(i, posItem + 1)
//
//      testFile.write(testByteBuffer.array())
//      testFile.close()
//
//    }
//    println(s"convert test path: ${System.currentTimeMillis() - startTime}ms")
//  }

}
