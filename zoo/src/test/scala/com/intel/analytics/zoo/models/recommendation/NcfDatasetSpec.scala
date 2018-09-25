package com.intel.analytics.zoo.models.recommendation

import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.zoo.examples.recommendation.NCFDataSet
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
    val batchSize = 4
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

}
