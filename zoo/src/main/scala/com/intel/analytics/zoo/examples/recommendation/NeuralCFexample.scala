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
import com.intel.analytics.bigdl.utils.{Engine, T}
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
                          val dataset: String = "ml-1m",
                          val batchSize: Int = 256,
                          val nEpochs: Int = 2,
                          val learningRate: Double = 1e-3,
                          val learningRateDecay: Double = 0.0,
                          val iteration: Int = 0,
                          val trainNegtiveNum: Int = 4,
                          val valNegtiveNum: Int = 100,
                          val layers: String = "64,32,16,8",
                          val numFactors: Int = 8,
                          val core: Int = 4
                    )

case class Rating(userId: Int, itemId: Int, label: Int, timestamp: Int, train: Boolean)

object NeuralCFexample {

  def main(args: Array[String]): Unit = {

    val defaultParams = NeuralCFParams()

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
      opt[Int]('i', "iter")
        .text("iteration numbers")
        .action((x, c) => c.copy(iteration = x))
      opt[Double]('l', "lRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x))
      opt[Int]("trainNeg")
        .text("The Number of negative instances to pair with a positive train instance.")
        .action((x, c) => c.copy(trainNegtiveNum = x))
      opt[Int]("valNeg")
        .text("The Number of negative instances to pair with a positive validation instance.")
        .action((x, c) => c.copy(valNegtiveNum = x))
      opt[String]("layers")
        .text("The sizes of hidden layers for MLP. Default is 64,32,16,8")
        .action((x, c) => c.copy(layers = x))
      opt[Int]("numFactors")
        .text("The Embedding size of MF model.")
        .action((x, c) => c.copy(numFactors = x))
      opt[Int]("core")
        .text("Core per executor.")
        .action((x, c) => c.copy(core = x))
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

    val validateBatchSize = param.core

    val (ratings, userCount, itemCount, itemMapping) =
      loadPublicData(sqlContext, param.inputDir, param.dataset)
    println(s"${userCount} ${itemCount}")
    val hiddenLayers = param.layers.split(",").map(_.toInt)

    val isImplicit = false
    val ncf = NeuralCF[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 1,
      userEmbed = hiddenLayers(0) / 2,
      itemEmbed = hiddenLayers(0) / 2,
      hiddenLayers = hiddenLayers.slice(1, hiddenLayers.length),
      mfEmbed = param.numFactors)

    println(ncf)

    println(s"parameter length: ${ncf.parameters()._1.map(_.nElement()).sum}")

    val (trainDataFrame, valDataFrame) = generateTrainValData(ratings, userCount, itemCount,
      trainNegNum = param.trainNegtiveNum, valNegNum = param.valNegtiveNum)

//    val trainData = sc.textFile("/tmp/ncf_recommendation_buffer/")

    val trainpairFeatureRdds =
      assemblyFeature(isImplicit, trainDataFrame, userCount, itemCount)
    val validationpairFeatureRdds =
      assemblyValFeature(isImplicit, valDataFrame, userCount, itemCount, param.valNegtiveNum)

    val trainRdds = trainpairFeatureRdds.map(x => x.sample).cache()
    val validationRdds = validationpairFeatureRdds.map(x => x.sample).cache()
    println(s"Train set ${trainRdds.count()} records")
    println(s"Val set ${validationRdds.count()} records")

    val valDataset = DataSet.array(validationRdds.collect()) -> SampleToMiniBatch[Float](validateBatchSize)

    val sampleToMiniBatch = SampleToMiniBatch[Float](param.batchSize)
    val trainDataset = (DataSet.array[Sample[Float]](trainRdds.collect()) -> sampleToMiniBatch).toLocal()

    val optimizer = new LocalOptimizer[Float](ncf,
      trainDataset, BCECriterion[Float]())

//    val optimizer = Optimizer(
//      model = ncf,
//      sampleRDD = trainRdds,
//      criterion = BCECriterion[Float](),
//      batchSize = param.batchSize)

    val optimMethod = new Adam[Float](
      learningRate = param.learningRate,
      learningRateDecay = param.learningRateDecay)
//    val optimMethod = new ParallelAdam[Float](
//      learningRate = param.learningRate,
//      learningRateDecay = param.learningRateDecay)

    val endTrigger = if (param.iteration != 0) {
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
        .setValidation(Trigger.everyEpoch, valDataset,
          Array(new HitRate[Float](negNum = param.valNegtiveNum),
          new Ndcg[Float](negNum = param.valNegtiveNum)))
//      .setValidation(Trigger.everyEpoch, validationRdds, Array(new HitRate[Float](),
//      new Ndcg[Float]()), 4)
      .setTrainSummary(trainSummary)
      .setValidationSummary(valSummary)
      .setEndWhen(endTrigger)
      .optimize()

  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String,
                     dataset: String): (DataFrame, Int, Int, Map[Int, Int]) = {
    import sqlContext.implicits._
    val ratings = dataset match {
      case "ml-1m" =>
        loadMl1mData(sqlContext, dataPath)
      case "ml-20m" =>
        loadMl20mData(sqlContext, dataPath)
      case _ =>
        throw new IllegalArgumentException(s"Only support dataset ml-1m and ml-20m, but got ${dataset}")
    }

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

  def loadMl1mData(sqlContext: SQLContext, dataPath: String): DataFrame = {
    import sqlContext.implicits._
    sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::")
        Rating(line(0).toInt, line(1).toInt, 1, line(3).toInt, true)
      }).toDF()
  }

  def loadMl20mData(sqlContext: SQLContext, dataPath: String): DataFrame = {
    val ratings = sqlContext.read
      .option("inferSchema", "true")
      .option("header", "true")
      .option("delimiter", ",")
      .csv(dataPath + "/ratings.csv")
      .toDF()
    println(ratings.schema)
    val result = ratings.withColumnRenamed("movieId", "itemId").withColumn("rating", lit(1))
      .withColumnRenamed("rating", "label").withColumn("train", lit(true))
    println(result.schema)
    result
  }

  def generateTrainValData(rating: DataFrame, userCount: Int, itemCount: Int,
                           trainNegNum: Int = 4, valNegNum: Int = 100): (DataFrame, DataFrame) = {
    val maxTimeStep = rating.groupBy("userId").max("timestamp").collect().map(r => (r.getInt(0), r.getInt(1))).toMap
    val bcT = rating.sparkSession.sparkContext.broadcast(maxTimeStep)
    val evalPos = rating.filter(r => bcT.value.apply(r.getInt(0)) == r.getInt(3)).dropDuplicates("userId")
      .collect().toSet
    val bcEval = rating.sparkSession.sparkContext.broadcast(evalPos)

    val negDataFrame = rating.sqlContext.createDataFrame(
      rating.rdd.groupBy(_.getAs[Int]("userId")).flatMap{v =>
        val userId = v._1
        val items = scala.collection.mutable.Set(v._2.map(_.getAs[Int]("itemId")).toArray: _*)
        val itemNumOfUser = items.size
        val gen = new Random()
        gen.setSeed(userId + 1)
        var i = 0
        val totalNegNum = trainNegNum * (itemNumOfUser - 1) + valNegNum

        val negs = new Array[Rating](totalNegNum)
        // gen negative sample to validation
        while(i < valNegNum) {
          val negItem = Random.nextInt(itemCount) + 1
          if (!items.contains(negItem)) {
            negs(i) = Rating(userId, negItem, 0, 0, false)
            i += 1
          }
        }

        // gen negative sample for train
        while(i < totalNegNum) {
          val negItem = gen.nextInt(itemCount) + 1
          if (!items.contains(negItem)) {
            negs(i) = Rating(userId, negItem, 0, 0, true)
            i += 1
          }
        }
        negs.toIterator
    })
//    println("neg train" + negDataFrame.filter(_.getAs[Boolean]("train")).count())
//    println("neg eval" + negDataFrame.filter(!_.getAs[Boolean]("train")).count())

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
                      itemCount: Int,
                      negNum: Int = 100): RDD[UserItemFeature[Float]] = {

    val rddOfSample: RDD[UserItemFeature[Float]] = indexed
      .select("userId", "itemId", "label")
      .rdd.groupBy(_.getAs[Int]("userId")).map(data => {
      val totalNum = 1 + negNum
      val uid = data._1
      val rows = data._2.toIterator
      val feature = Tensor(totalNum, 2).fill(uid)
      val label = Tensor(totalNum)

      var i = 1
      while(rows.hasNext) {
        val current = rows.next()
        val iid = current.getAs[Int]("itemId")
        val l = current.getAs[Int]("label")
        feature.setValue(i, 2, iid)
        label.setValue(i, l)

        i += 1
      }
      require(i == totalNum + 1)

      UserItemFeature(uid, -1, Sample(feature, label))
    })
    rddOfSample
  }

}

class HitRate[T: ClassTag](k: Int = 10, negNum: Int = 100)(
    implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {
  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    val o = output.toTensor[T].resize(1 + negNum)
    val t = target.toTensor[T].resize(1 + negNum)
    var exceptedTarget = 0
    var i = 1
    while(i <= t.nElement()) {
      if (t.valueAt(i) == 1) {
        exceptedTarget = i
      }
      i += 1
    }
    require(exceptedTarget != 0, s"No positive sample")

    val hr = hitRate(exceptedTarget, o, k)

    new LossResult(hr, 1)
  }

  def hitRate(index: Int, o: Tensor[T], k: Int): Float = {
    var topK = 1
    var i = 1
    val precision = ev.toType[Float](o.valueAt(index))
    while (i < o.nElement() && topK <= k) {
      if (ev.toType[Float](o.valueAt(i)) > precision) {
        topK += 1
      }
      i += 1
    }

    if(topK <= k) {
      1
    } else {
      0
    }
  }

  override def format(): String = "HitRate@10"
}

class Ndcg[T: ClassTag](k: Int = 10, negNum: Int = 100)(
    implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {
  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    val o = output.toTensor[T].resize(1 + negNum)
    val t = target.toTensor[T].resize(1 + negNum)
    var exceptedTarget = 0
    var i = 1
    while(i <= t.nElement()) {
      if (t.valueAt(i) == 1) {
        exceptedTarget = i
      }
      i += 1
    }
    require(exceptedTarget != 0, s"No positive sample")

    val n = ndcg(exceptedTarget, o, k)

    new LossResult(n, 1)
  }

  def ndcg(index: Int, o: Tensor[T], k: Int): Float = {
    var ranking = 1
    var i = 1
    val precision = ev.toType[Float](o.valueAt(index))
    while (i < o.nElement() && ranking <= k) {
      if (ev.toType[Float](o.valueAt(i)) > precision) {
        ranking += 1
      }
      i += 1
    }

    if(ranking <= k) {
      (math.log(2) / math.log(ranking + 1)).toFloat
    } else {
      0
    }
  }

  override def format(): String = "HDCG"
}
