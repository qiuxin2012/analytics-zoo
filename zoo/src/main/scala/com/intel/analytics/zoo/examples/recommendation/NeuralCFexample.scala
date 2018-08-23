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

package com.intel.analytics.zoo.examples.recommendation

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.{BCECriterion, ClassNLLCriterion}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.recommendation.{NeuralCF, UserItemFeature, Utils}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions._
import scopt.OptionParser

import scala.reflect.ClassTag
import scala.util.Random

case class NeuralCFParams(val inputDir: String = "./data/ml-1m",
                          val batchSize: Int = 256,
                          val nEpochs: Int = 10,
                          val learningRate: Double = 1e-3,
                          val learningRateDecay: Double = 1e-6,
                          val iteration: Int = 0,
                          val tfVersion: Boolean = true
                    )

case class Rating(userId: Int, itemId: Int, label: Int, timeStep: Long, train: Boolean)

object NeuralCFexample {

  def main(args: Array[String]): Unit = {

    val defaultParams = NeuralCFParams()

    val parser = new OptionParser[NeuralCFParams]("NCF Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
      opt[Int]('i', "iter")
        .text("iteration numbers")
        .action((x, c) => c.copy(iteration = x))
      opt[Double]('l', "lRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x.toDouble))
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
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (ratings, userCount, itemCount, itemMapping) = loadPublicData(sqlContext, param.inputDir)
    val r = ratings.select("label").distinct().collect()

    val isImplicit = false
    val ncf = NeuralCF[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 1,
      userEmbed = 32,
      itemEmbed = 32,
      hiddenLayers = Array(32, 16, 8),
      mfEmbed = 8)


    val (trainDataFrame, valDataFrame) = generateTrainValData(ratings, userCount, itemCount)

    trainDataFrame.count()
    val r1 = trainDataFrame.select("label").distinct().collect()
    valDataFrame.count()
    val r2 = valDataFrame.select("label").distinct().collect()

    val trainpairFeatureRdds =
      assemblyFeature(isImplicit, trainDataFrame, userCount, itemCount)
    println("train count: " + trainpairFeatureRdds.count())
    val validationpairFeatureRdds =
      assemblyValFeature(isImplicit, valDataFrame, userCount, itemCount)
    println("val count: " + validationpairFeatureRdds.count())

    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)
    val a = validationRdds.take(5)
    val valDataset = DataSet.array(validationRdds.collect()) -> SampleToMiniBatch[Float](4)

    val sampleToMiniBatch = SampleToMiniBatch[Float](param.batchSize)

    val optimizer = new LocalOptimizer[Float](ncf,
      (DataSet.array[Sample[Float]](trainRdds.collect()) -> sampleToMiniBatch).toLocal(),
      BCECriterion[Float]())

//    val optimizer = Optimizer(
//      model = ncf,
//      sampleRDD = trainRdds,
//      criterion = BCECriterion[Float](),
//      batchSize = param.batchSize)

    val optimMethod = new SGD[Float](
      learningRate = param.learningRate)
//    val optimMethod = new ParallelAdam[Float](
//      learningRate = param.learningRate,
//      learningRateDecay = param.learningRateDecay)

    val endTrigger = if(param.iteration != 0 ) {
      Trigger.maxIteration(param.iteration)
    } else {
      Trigger.maxEpoch(param.nEpochs)
    }

    val appname = System.nanoTime()
    val trainSummary = new TrainSummary("/tmp/ncf-bigdl", s"${appname}")
    trainSummary.setSummaryTrigger("Parameters", Trigger.everyEpoch)
    val valSummary = new ValidationSummary("/tmp/ncf-bigdl", s"${appname}")

    optimizer
      .setOptimMethod(optimMethod)
        .setValidation(Trigger.everyEpoch, valDataset, Array(new Top1Accuracy()))
      .setTrainSummary(trainSummary)
      .setValidationSummary(valSummary)
      .setEndWhen(endTrigger)
      .optimize()

    val results = ncf.predict(validationRdds)
    results.take(5).foreach(println)
    val resultsClass = ncf.predictClass(validationRdds)
    resultsClass.take(5).foreach(println)

    val userItemPairPrediction = ncf.predictUserItemPair(validationpairFeatureRdds)

    userItemPairPrediction.take(5).foreach(println)

    val userRecs = ncf.recommendForUser(validationpairFeatureRdds, 3)
    val itemRecs = ncf.recommendForItem(validationpairFeatureRdds, 3)

    userRecs.take(10).foreach(println)
    itemRecs.take(10).foreach(println)
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String): (DataFrame, Int, Int, Map[Int, Int]) = {
    import sqlContext.implicits._
    val ratings = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::")
        Rating(line(0).toInt, line(1).toInt, 1, line(3).toLong, true)
      }).toDF()

    val minMaxRow = ratings.agg(max("userId")).collect()(0)
    val userCount = minMaxRow.getInt(0)

    val uniqueMovie = ratings.rdd.map(_.getAs[Int]("itemId")).distinct().collect().sortWith(_ < _)
    val mapping = uniqueMovie.zip(1 to uniqueMovie.length).toMap

    val bcMovieMapping = sqlContext.sparkContext.broadcast(mapping)

    val mappingUdf = udf((itemId: Int) => {
     bcMovieMapping.value(itemId)
    })
    val mappedItemID = mappingUdf.apply(col("itemId"))
    val mappedRating = ratings//.drop(col("itemId"))
      .withColumn("itemId", mappedItemID)
    mappedRating.show()


    (mappedRating, userCount, uniqueMovie.length, mapping)
  }

  def generateTrainValData(rating: DataFrame, userCount: Int, itemCount: Int,
                           trainNegNum: Int = 4, valNegNum: Int = 999): (DataFrame, DataFrame) = {
    val maxTimeStep = rating.groupBy("userId").max("timeStep").collect().map(r => (r.getInt(0), r.getLong(1))).toMap
    val bcT = rating.sparkSession.sparkContext.broadcast(maxTimeStep)
    val evalPos = rating.filter(r => bcT.value.apply(r.getInt(0)) == r.getLong(3)).dropDuplicates("userId")
      .collect().toSet
    val bcEval = rating.sparkSession.sparkContext.broadcast(evalPos)

    val negDataFrame = rating.sqlContext.createDataFrame(
      rating.rdd.groupBy(_.getAs[Int]("userId")).flatMap{v =>
        val userId = v._1
        val items = scala.collection.mutable.Set(v._2.map(_.getAs[Int]("itemId")).toArray: _*)
        val itemNumOfUser = items.size
        val gen = new Random()
        gen.setSeed(userId)
        var i = 0
        val totalNegNum = if (trainNegNum * itemNumOfUser > itemCount - itemNumOfUser) {
          itemCount - itemNumOfUser + valNegNum
        } else {
          trainNegNum * itemNumOfUser + valNegNum
        }
        val negs = new Array[Rating](totalNegNum)
        // gen negative sample to validation
        while(i < valNegNum) {
          val negItem = Random.nextInt(itemCount) + 1
          if (!items.contains(negItem)) {
            negs(i) = Rating(userId, negItem, 0, 0L, false)
            i += 1
          }
        }

        // gen negative sample for train
        while(i < totalNegNum) {
          val negItem = gen.nextInt(itemCount) + 1
          if (!items.contains(negItem)) {
            negs(i) = Rating(userId, negItem, 0, 0L, true)
            items.add(negItem)
            i += 1
          }
        }
        negs.toIterator
    })
    println("neg train" + negDataFrame.filter(_.getAs[Boolean]("train")).count())
    println("neg eval" + negDataFrame.filter(!_.getAs[Boolean]("train")).count())

    (negDataFrame.filter(_.getAs[Boolean]("train"))
      .union(rating.filter(r => !bcEval.value.contains(r))),
      negDataFrame.filter(!_.getAs[Boolean]("train"))
        .union(rating.filter(r => bcEval.value.contains(r))))

  }

  def assemblyFeature(isImplicit: Boolean = false,
                      indexed: DataFrame,
                      userCount: Int,
                      itemCount: Int): RDD[UserItemFeature[Float]] = {

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(indexed)
      negativeDF.unionAll(indexed.withColumn("label", lit(2)))
    }
    else indexed

    val rddOfSample: RDD[UserItemFeature[Float]] = unioned
      .select("userId", "itemId", "label")
      .rdd.map(row => {
      val uid = row.getAs[Int](0)
      val iid = row.getAs[Int](1)

      val label = row.getAs[Int](2)
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
    })
    rddOfSample
  }

  def assemblyValFeature(isImplicit: Boolean = false,
                      indexed: DataFrame,
                      userCount: Int,
                      itemCount: Int): RDD[UserItemFeature[Float]] = {

    val rddOfSample: RDD[UserItemFeature[Float]] = indexed
      .select("userId", "itemId", "label")
      .rdd.groupBy(_.getAs[Int]("userId")).map(data => {
      val uid = data._1
      val rows = data._2.toIterator
      val feature = Tensor(1000, 2).fill(uid)
      val label = Tensor(1000)

      var i = 1
      while(rows.hasNext) {
        val current = rows.next()
        val iid = current.getAs[Int]("itemId")
        val l = current.getAs[Int]("label")
        feature.setValue(i, 2, iid)
        label.setValue(i, l)

        i += 1
      }

      UserItemFeature(uid, -1, Sample(feature, label))
    })
    rddOfSample
  }


  class HitRate[T: ClassTag](k: Int = 10)(
                           implicit ev: TensorNumeric[T])
    extends ValidationMethod[T] {
    override def apply(output: Activity, target: Activity):
    ValidationResult = {
      val topk = findLargestK()

      new AccuracyResult(correct, count)
    }

    def findLargestK():Array[(Int, Double)] = {

    }

    override def format(): String = "HitRate@10"
  }

}
