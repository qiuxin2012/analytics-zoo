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

package com.intel.analytics.zoo.common

import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator}
import jep._
import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.zoo.common.NNContext.initNNContext
import com.intel.analytics.zoo.pipeline.estimator.python.PythonEstimator


class UtilsSpec extends FlatSpec with Matchers {
  val path: String = getClass.getClassLoader.getResource("qa").getPath
  val txtRelations: String = path + "/relations.txt"

  "Utils listFiles" should "work properly" in {
    val files = Utils.listPaths(path)
    assert(files.size == 3)
    val recursiveFiles = Utils.listPaths(path, true)
    assert(recursiveFiles.size == 13)
  }

  "Utils readBytes" should "work properly" in {
    val inputStream = Utils.open(txtRelations)
    val fileLen = inputStream.available()
    inputStream.close()
    val bytes = Utils.readBytes(txtRelations)
    assert(bytes.length == fileLen)
  }

  "Utils saveBytes" should "work properly" in {
    val fs = Utils.getFileSystem(path)
    // Generate random file
    val tmpFile = System.currentTimeMillis()
    val randomContent = new Array[Byte](1000)
    Utils.saveBytes(randomContent, path + "/" + tmpFile)
    // Delete random file
    fs.deleteOnExit(new Path(path + "/" + tmpFile))
    fs.close()
  }

  "123" should "work" in {
    val config: JepConfig = new JepConfig()
    config.setClassEnquirer(new NamingConventionClassEnquirer())
    SharedInterpreter.setConfig(config)
    val c = new SharedInterpreter()
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
         |    print(str(i) + ": " + str(loss.data.item()) + " " + str(time.time() - s))
         |""".stripMargin
       val start = System.nanoTime()
       c.exec(str)
       println((System.nanoTime() - start) / 1e9)
  }

  "12345" should "work" in {
    import com.intel.analytics.bigdl.utils.Engine
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("spark.master", "local[4]")
    val sc =initNNContext("1234")
//    val sc = new SparkContext(new SparkConf().setAppName("1234").setMaster("local[4]")
//    .set("spark.shuffle.reduceLocality.enabled", "false"))
//    Engine.init
   new PythonEstimator[Float]().estimatorTest()

  }
}
