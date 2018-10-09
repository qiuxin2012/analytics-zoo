package com.intel.analytics.zoo.models.recommendation

import com.intel.analytics.bigdl.nn.BCECriterion
import com.intel.analytics.bigdl.optim.{EmbeddingAdam2, NCFOptimizer2, ParallelAdam, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator}
import com.intel.analytics.zoo.examples.mlperf.recommendation.NCFDataSet
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper

class NcfDatasetSpec extends ZooSpecHelper{
  "dataset" should "generate right result" in {
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val valPos = Map(
      1 -> 1,
      2 -> 2,
      3 -> 3,
      4 -> 4,
      5 -> 5,
      6 -> 6,
      7 -> 7,
      8 -> 8
    )

    val trainNegatives = 3
    val batchSize = 3
    val userCount = 8
    val itemCount = 10

    val ncfD = new NCFDataSet(trainSet, valPos,
      trainNegatives, batchSize, userCount, itemCount)

    RandomGenerator.RNG.setSeed(10)
    ncfD.shuffle()
    val ite = ncfD.data(true)
    var count = 0
    while(ite.hasNext) {
      val batch = ite.next()
      val input = batch.getInput().toTensor[Float]
      val target = batch.getTarget().toTensor[Float]
      (1 to input.size(1)).foreach{i =>
        val userID = input.valueAt(i, 1)
        if (target.valueAt(i, 1) == 1) {
          Some(input.valueAt(i, 2)) should contain oneOf (userID + 1, userID + 2)
        } else {
          Some(input.valueAt(i, 2)) should not contain oneOf (userID, userID + 1, userID + 2)
        }
      }
      count += 1
    }
   count should be (22)

  }

  "dataset" should "generate right result 2" in {
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val valPos = Map(
      1 -> 1,
      2 -> 2,
      3 -> 3,
      4 -> 4,
      5 -> 5,
      6 -> 6,
      7 -> 7,
      8 -> 8
    )

    val trainNegatives = 3
    val batchSize = 4
    val userCount = 8
    val itemCount = 10

    val ncfD = new NCFDataSet(trainSet, valPos,
      trainNegatives, batchSize, userCount, itemCount)

    RandomGenerator.RNG.setSeed(10)
    ncfD.shuffle()
    var ite = ncfD.data(true)
    while(ite.hasNext) {
      val batch = ite.next()
      val input = batch.getInput().toTensor[Float]
      val target = batch.getTarget().toTensor[Float]
      (1 to input.size(1)).foreach{i =>
        val userID = input.valueAt(i, 1)
        if (target.valueAt(i, 1) == 1) {
          Some(input.valueAt(i, 2)) should contain oneOf (userID + 1, userID + 2)
        } else {
          Some(input.valueAt(i, 2)) should not contain oneOf (userID, userID + 1, userID + 2)
        }
      }
    }

    RandomGenerator.RNG.setSeed(12)
    ncfD.shuffle()
    ite = ncfD.data(true)
    while(ite.hasNext) {
      val batch = ite.next()
      val input = batch.getInput().toTensor[Float]
      val target = batch.getTarget().toTensor[Float]
      (1 to input.size(1)).foreach{i =>
        val userID = input.valueAt(i, 1)
        if (target.valueAt(i, 1) == 1) {
          Some(input.valueAt(i, 2)) should contain oneOf (userID + 1, userID + 2)
        } else {
          Some(input.valueAt(i, 2)) should not contain oneOf (userID, userID + 1, userID + 2)
        }
      }
    }
  }

  "dataset" should "generate right result 3" in {
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val valPos = Map(
      1 -> 1,
      2 -> 2,
      3 -> 3,
      4 -> 4,
      5 -> 5,
      6 -> 6,
      7 -> 7,
      8 -> 8
    )

    val trainNegatives = 3
    val batchSize = 4
    val userCount = 8
    val itemCount = 10

    val ncfD = new NCFDataSet(trainSet, valPos,
      trainNegatives, batchSize, userCount, itemCount)
    val userCounts = new Array[Int](8)

    RandomGenerator.RNG.setSeed(10)
    ncfD.shuffle()
    var ite = ncfD.data(true)
    while(ite.hasNext) {
      val batch = ite.next()
      val input = batch.getInput().toTensor[Float]
      val target = batch.getTarget().toTensor[Float]
      (1 to input.size(1)).foreach{i =>
        val userID = input.valueAt(i, 1)
        userCounts(userID.toInt - 1) += 1
        if (target.valueAt(i, 1) == 1) {
          Some(input.valueAt(i, 2)) should contain oneOf (userID + 1, userID + 2)
        } else {
          Some(input.valueAt(i, 2)) should not contain oneOf (userID, userID + 1, userID + 2)
        }
      }
    }

    userCounts should be (Array.tabulate(8)(_ => 8))

    ite = ncfD.data(true)
    while(ite.hasNext) {
      val batch = ite.next()
      val input = batch.getInput().toTensor[Float]
      val target = batch.getTarget().toTensor[Float]
      (1 to input.size(1)).foreach{i =>
        val userID = input.valueAt(i, 1)
        userCounts(userID.toInt - 1) += 1
        if (target.valueAt(i, 1) == 1) {
          Some(input.valueAt(i, 2)) should contain oneOf (userID + 1, userID + 2)
        } else {
          Some(input.valueAt(i, 2)) should not contain oneOf (userID, userID + 1, userID + 2)
        }
      }
    }

    userCounts should be (Array.tabulate(8)(_ => 16))
  }

  "dataset" should "generate right result 4" in {
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val valPos = Map(
      1 -> 1,
      2 -> 2,
      3 -> 3,
      4 -> 4,
      5 -> 5,
      6 -> 6,
      7 -> 7,
      8 -> 8
    )

    val trainNegatives = 3
    val batchSize = 4
    val userCount = 8
    val itemCount = 10

    val ncfD = new NCFDataSet(trainSet, valPos,
      trainNegatives, batchSize, userCount, itemCount)

    RandomGenerator.RNG.setSeed(10)
    ncfD.shuffle()
    val tensor1 = Tensor(ncfD.inputBuffer.clone(), Array(64, 2))
    RandomGenerator.RNG.setSeed(10)
    ncfD.shuffle()
    val tensor2 = Tensor(ncfD.inputBuffer.clone(), Array(64, 2))
    tensor1 should not be (tensor2)

  }

  "dataset" should "run with ncfoptimizer" in {
    System.setProperty("bigdl.localMode", "true")
    Engine.init(1, 1, false)
    val trainSet = Array(
      (1, Set(2, 3)),
      (2, Set(3, 4)),
      (3, Set(4, 5)),
      (4, Set(5, 6)),
      (5, Set(6, 7)),
      (6, Set(7, 8)),
      (7, Set(8, 9)),
      (8, Set(9, 10)))

    val valPos = Map(
      1 -> 1,
      2 -> 2,
      3 -> 3,
      4 -> 4,
      5 -> 5,
      6 -> 6,
      7 -> 7,
      8 -> 8
    )

    val trainNegatives = 3
    val batchSize = 11
    val userCount = 8
    val itemCount = 10
    val numFactors = 8
    val learningRate = 1e-3

    val trainDataset = new NCFDataSet(trainSet, valPos,
      trainNegatives, batchSize, userCount, itemCount)
    trainDataset.shuffle()

    val hiddenLayers = Array(16, 16, 8, 4)

    val optimMethod = Map(
      "embeddings" -> new EmbeddingAdam2[Float](
        learningRate = learningRate,
        userCount = userCount,
        itemCount = itemCount,
        embedding1 = hiddenLayers(0) / 2,
        embedding2 = numFactors),
      "linears" -> new ParallelAdam[Float](
        learningRate = learningRate))

    val ncf = NeuralCFV2[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 1,
      userEmbed = hiddenLayers(0) / 2,
      itemEmbed = hiddenLayers(0) / 2,
      hiddenLayers = hiddenLayers.slice(1, hiddenLayers.length),
      mfEmbed = numFactors)

    val optimizer = new NCFOptimizer2[Float](ncf,
      trainDataset, BCECriterion[Float]())

    optimizer
        .setEndWhen(Trigger.maxEpoch(1))
      .setOptimMethods(optimMethod)
      .optimize()

    trainDataset.shuffle()

    optimizer
      .setEndWhen(Trigger.maxEpoch(2))
      .optimize()

    trainDataset.shuffle()

    optimizer
      .setEndWhen(Trigger.maxEpoch(3))
      .optimize()

  }
}
