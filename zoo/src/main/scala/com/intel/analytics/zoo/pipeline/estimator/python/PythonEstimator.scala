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

package com.intel.analytics.zoo.pipeline.estimator.python

import java.util.{List => JList, Map => JMap}

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.optim.{OptimMethod, Trigger, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFeatureToMiniBatch}
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.feature.{FeatureSet, PythonLoaderFeatureSet}
import com.intel.analytics.zoo.pipeline.estimator.Estimator
import jep.{JepConfig, NamingConventionClassEnquirer, SharedInterpreter}
import org.apache.spark.SparkContext

import scala.reflect.ClassTag
import scala.collection.JavaConverters._

object PythonEstimator {
  def ofFloat(): PythonEstimator[Float] = new PythonEstimator[Float]()

  def ofDouble(): PythonEstimator[Double] = new PythonEstimator[Double]()
}

class PythonEstimator[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T]{
  def createEstimator(model: Module[T],
                      optimMethod: OptimMethod[T],
                      modelDir: String): Estimator[T] = {
    Estimator(model, optimMethod, modelDir)
  }

  def createEstimator(model: Module[T],
                      optimMethods: JMap[String, OptimMethod[T]],
                      modelDir: String): Estimator[T] = {
    require(optimMethods != null, "optimMethods cannot be null")
    Estimator(model, optimMethods.asScala.toMap, modelDir)
  }

  def createPytorchEstimator(model: Array[Byte],
                             optimMethod: OptimMethod[T],
                             modelDir: String): Unit = {
    Estimator(model, optimMethod, modelDir)
  }

  def estimatorEvaluate(estimator: Estimator[T], validationSet: FeatureSet[Sample[T]],
                        validationMethod: JList[ValidationMethod[T]], batchSize: Int
                       ): Map[ValidationMethod[T], ValidationResult] = {
    val sample2batch = SampleToMiniBatch(batchSize)
    val validationMiniBatch = validationSet -> sample2batch
    estimator.evaluate(validationMiniBatch, validationMethod.asScala.toArray)
  }

  def estimatorEvaluateImageFeature(estimator: Estimator[T],
                                    validationSet: FeatureSet[ImageFeature],
                                    validationMethod: JList[ValidationMethod[T]],
                                    batchSize: Int
                                   ): Map[ValidationMethod[T], ValidationResult] = {
    val imageFeature2batch = ImageFeatureToMiniBatch(batchSize)
    val validationMiniBatch = validationSet -> imageFeature2batch
    estimator.evaluate(validationMiniBatch, validationMethod.asScala.toArray)
  }

  def estimatorTrain(estimator: Estimator[T], trainSet: FeatureSet[Sample[T]],
                     criterion: Criterion[T],
                     endTrigger: Trigger = null,
                     checkPointTrigger: Trigger = null,
                     validationSet: FeatureSet[Sample[T]] = null,
                     validationMethod: JList[ValidationMethod[T]] = null,
                     batchSize: Int)
  : estimator.type = {
    val sample2batch = SampleToMiniBatch(batchSize)
    val trainMiniBatch = trainSet -> sample2batch
    val validationMiniBatch = if (validationSet != null) {
      validationSet -> sample2batch
    } else {
      null
    }

    estimator.train(trainMiniBatch, criterion,
      Option(endTrigger), Option(checkPointTrigger),
      validationMiniBatch, Option(validationMethod).map(_.asScala.toArray).orNull)
  }

  def createSampleToMiniBatch(batchSize: Int): SampleToMiniBatch[T] = {
    SampleToMiniBatch(batchSize)
  }

  def estimatorTrainMiniBatch(
      estimator: Estimator[T],
      trainSet: FeatureSet[MiniBatch[T]],
      criterion: Criterion[T],
      endTrigger: Trigger = null,
      checkPointTrigger: Trigger = null,
      validationSet: FeatureSet[MiniBatch[T]] = null,
      validationMethod: JList[ValidationMethod[T]] = null) : estimator.type = {
    estimator.train(trainSet, criterion,
      Option(endTrigger), Option(checkPointTrigger),
      validationSet, Option(validationMethod).map(_.asScala.toArray).orNull)
  }

  def estimatorEvaluateMiniBatch(
      estimator: Estimator[T],
      validationMiniBatch: FeatureSet[MiniBatch[T]],
      validationMethod: JList[ValidationMethod[T]]
      ): Map[ValidationMethod[T], ValidationResult] = {
    estimator.evaluate(validationMiniBatch, validationMethod.asScala.toArray)
  }

  def printOmpThread(): String = {
    val sc = SparkContext.getOrCreate()
    val rdd = sc.parallelize(0 to 100, 1)
    rdd.mapPartitions{iter =>
      val c = PythonLoaderFeatureSet.getOrCreateInterpreter()
      val str =
        s"""
           |import os
           |""".stripMargin
      c.exec(str)
      val v = c.getValue("os.environ['OMP_NUM_THREADS']").asInstanceOf[String]
      println(v)
      Iterator.single(v)
    }.reduce(_ + _)

  }

  def estimatorTest(): Double = {
    val sc = SparkContext.getOrCreate()
    val rdd = sc.parallelize(0 to 100, 1)
    rdd.mapPartitions{iter =>
      val c = PythonLoaderFeatureSet.getOrCreateInterpreter()
      val str =
        s"""
           |import torch
           |import torch.nn as nn
           |import torchvision
           |import torch.nn.functional as F
           |import torch.optim as optim
           |from torchvision import datasets, transforms
           |from zoo.pipeline.api.net.torch_net import TorchNet2
           |from zoo.pipeline.api.net.torch_criterion import TorchCriterion2
           |from zoo.pipeline.estimator import *
           |
           |normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           |train_dataset = datasets.ImageFolder(
           |    '/home/xin/datasets/imagenet-small/train',
           |    transforms.Compose([
           |        transforms.RandomResizedCrop(224),
           |        transforms.RandomHorizontalFlip(),
           |        transforms.ToTensor(),
           |        normalize,
           |    ]))
           |
           |train_loader = torch.utils.data.DataLoader(
           |    train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=1)
           |
           |model = torchvision.models.resnet50()
           |model.train()
           |criterion = nn.CrossEntropyLoss()
           |import time
           |for i, (images, target) in enumerate(train_loader):
           |    s = time.time()
           |    output = model(images)
           |    loss = criterion(output, target)
           |    loss.backward()
           |    print(str(i) + ": " + str(loss.data.item()) + " " + str(time.time() - s))
           |""".stripMargin
      val start = System.nanoTime()
      c.exec(str)
      println((System.nanoTime() - start) / 1e9)
      Iterator.single((System.nanoTime() - start) / 1e9)
    }.reduce(_ + _)
  }

  def estimatorTrainImageFeature(estimator: Estimator[T],
                                 trainSet: FeatureSet[ImageFeature],
                                 criterion: Criterion[T],
                                 endTrigger: Trigger = null,
                                 checkPointTrigger: Trigger = null,
                                 validationSet: FeatureSet[ImageFeature] = null,
                                 validationMethod: JList[ValidationMethod[T]] = null,
                                 batchSize: Int)
  : estimator.type = {
    val imageFeature2batch = ImageFeatureToMiniBatch(batchSize)
    val trainMiniBatch = trainSet -> imageFeature2batch
    val validationMiniBatch = if (validationSet != null) {
      validationSet -> imageFeature2batch
    } else {
      null
    }
    val valMethods = if (validationMethod != null) {
      validationMethod.asScala.toArray
    } else {
      null
    }

    estimator.train(trainMiniBatch, criterion,
      Option(endTrigger), Option(checkPointTrigger),
      validationMiniBatch, valMethods)
  }

  def clearGradientClipping(estimator: Estimator[T]): Unit = {
    estimator.clearGradientClipping()
  }

  def setConstantGradientClipping(estimator: Estimator[T], min: Double, max: Double): Unit = {
    estimator.setConstantGradientClipping(min, max)
  }

  def setGradientClippingByL2Norm(estimator: Estimator[T], clipNorm: Double): Unit = {
    estimator.setGradientClippingByL2Norm(clipNorm)
  }
}
