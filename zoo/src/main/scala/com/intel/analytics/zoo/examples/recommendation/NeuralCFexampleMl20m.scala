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

//package com.intel.analytics.zoo.examples.recommendation
//
//import com.intel.analytics.bigdl._
//import com.intel.analytics.bigdl.dataset.{DataSet, Sample, SampleToMiniBatch}
//import com.intel.analytics.bigdl.example.recommendation.NeuralCFV2
//import com.intel.analytics.bigdl.nn.{BCECriterion, ClassNLLCriterion}
//import com.intel.analytics.bigdl.numeric.NumericFloat
//import com.intel.analytics.bigdl.optim._
//import com.intel.analytics.bigdl.tensor.Tensor
//import com.intel.analytics.bigdl.utils.T
//import com.intel.analytics.zoo.common.NNContext
//import com.intel.analytics.zoo.models.recommendation.{NeuralCF, UserItemFeature, Utils}
//import org.apache.log4j.{Level, Logger}
//import org.apache.spark.SparkConf
//import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.{DataFrame, SQLContext}
//import org.apache.spark.sql.functions._
//import scopt.OptionParser
//
//// run with https://github.com/qiuxin2012/BigDL/tree/ncfPerf
//object NeuralCFexampleMl20m {
//
//  def main(args: Array[String]): Unit = {
//
//    val defaultParams = NeuralCFParams()
//
//    val parser = new OptionParser[NeuralCFParams]("NCF Example") {
//      opt[String]("inputDir")
//        .text(s"inputDir")
//        .action((x, c) => c.copy(inputDir = x))
//      opt[Int]('b', "batchSize")
//        .text(s"batchSize")
//        .action((x, c) => c.copy(batchSize = x.toInt))
//      opt[Int]('e', "nEpochs")
//        .text("epoch numbers")
//        .action((x, c) => c.copy(nEpochs = x))
//      opt[Int]('i', "iter")
//        .text("iteration numbers")
//        .action((x, c) => c.copy(iteration = x))
//      opt[Double]('l', "lRate")
//        .text("learning rate")
//        .action((x, c) => c.copy(learningRate = x.toDouble))
//    }
//
//    parser.parse(args, defaultParams).map {
//      params =>
//        run(params)
//    } getOrElse {
//      System.exit(1)
//    }
//  }
//
//  def run(param: NeuralCFParams): Unit = {
//    Logger.getLogger("org").setLevel(Level.ERROR)
//    Logger.getLogger("com.intel").setLevel(Level.DEBUG)
//    val conf = new SparkConf()
//    conf.set("spark.driver.maxResultSize", "2048")
//    conf.setAppName("NCFExample").set("spark.sql.crossJoin.enabled", "true")
//    val sc = NNContext.initNNContext(conf)
//    val sqlContext = SQLContext.getOrCreate(sc)
//
//    val (ratings, userCount, itemCount, userIdMapping) = loadMl20mData(sqlContext, param.inputDir)
//
//    val isImplicit = false
//    val ncf = NeuralCFV2[Float](
//      userCount = userCount,
//      itemCount = itemCount,
//      numClasses = 1,
//      userEmbed = 128,
//      itemEmbed = 128,
//      hiddenLayers = Array(256, 128, 64),
//      mfEmbed = 64).buildModel()
//
//    val pairFeatureRdds: RDD[UserItemFeature[Float]] =
//      assemblyFeature(isImplicit, ratings, userCount, itemCount, userIdMapping)
//
//    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
//      pairFeatureRdds.randomSplit(Array(0.8, 0.2))
//    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
//    val validationRdds = validationpairFeatureRdds.map(x => x.sample)
//
//    val sampleToMiniBatch = SampleToMiniBatch[Float](param.batchSize)
//
//
//    val ncfOptimzer = new NCFOptimizer[Float](ncf,
//      (DataSet.array[Sample[Float]](trainRdds.collect()) -> sampleToMiniBatch).toLocal(),
//      BCECriterion[Float]())
//
//    val optimMethod = new ParallelAdam[Float](
//      learningRate = param.learningRate,
//      learningRateDecay = param.learningRateDecay)
//
//    val endTrigger = if(param.iteration != 0 ) {
//      Trigger.maxIteration(param.iteration)
//    } else {
//      Trigger.maxEpoch(param.nEpochs)
//    }
//
//    ncfOptimzer
//      .setOptimMethod(optimMethod)
//      .setEndWhen(endTrigger)
//      .optimize()
//
////    val optimizer = Optimizer(
////      model = ncf,
////      sampleRDD = trainRdds,
////      criterion = BCECriterion[Float](),
////      batchSize = param.batchSize)
////    optimizer
////      .setOptimMethod(optimMethod)
////      .setEndWhen(Trigger.maxEpoch(param.nEpochs))
////      .optimize()
//
////    val results = ncf.predict(validationRdds)
////    results.take(5).foreach(println)
////    val resultsClass = ncf.predictClass(validationRdds)
////    resultsClass.take(5).foreach(println)
//
////    val userItemPairPrediction = ncf.predictUserItemPair(validationpairFeatureRdds)
////
////    userItemPairPrediction.take(5).foreach(println)
////
////    val userRecs = ncf.recommendForUser(validationpairFeatureRdds, 3)
////    val itemRecs = ncf.recommendForItem(validationpairFeatureRdds, 3)
////
////    userRecs.take(10).foreach(println)
////    itemRecs.take(10).foreach(println)
//  }
//
//  def loadMl20mData(sqlContext: SQLContext, dataPath: String): (DataFrame, Int, Int, Map[Int, Int]) = {
//    import sqlContext.implicits._
//    val ratings = sqlContext.read
//      .option("inferSchema", "true")
//      .option("header", "true")
//      .option("delimiter", ",")
//      .csv(dataPath + "/ratings.csv")
//      .toDF()
//    println(ratings.schema)
//
//    val minMaxRow = ratings.agg(max("userId")).collect()(0)
//    val usercount = minMaxRow.getInt(0)
//
//    val uniqueMovie = ratings.rdd.map(_.getAs[Int]("movieId")).distinct().collect()
//    val mapping = uniqueMovie.zip(1 to uniqueMovie.length).toMap
//
//    (ratings, usercount, uniqueMovie.length, mapping)
//  }
//
//  def assemblyFeature(isImplicit: Boolean = false,
//                      indexed: DataFrame,
//                      userCount: Int,
//                      itemCount: Int,
//                      movieMapping: Map[Int, Int]): RDD[UserItemFeature[Float]] = {
//    val bcMovieMapping = indexed.sparkSession.sparkContext.broadcast(movieMapping)
//
//    val unioned = if (isImplicit) {
//      val negativeDF = Utils.getNegativeSamples(indexed)
//      negativeDF.unionAll(indexed.withColumn("rating", lit(2)))
//    }
//    else indexed
//
//    val rddOfSample: RDD[UserItemFeature[Float]] = unioned
//      .select("userId", "movieId", "rating")
//      .rdd.map(row => {
//      val mapping = bcMovieMapping.value
//      val uid = row.getAs[Int](0)
//      val iid = mapping(row.getAs[Int](1))
//
//      val label = 1
//      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))
//
//      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
//    })
//    rddOfSample
//  }
//
//}
