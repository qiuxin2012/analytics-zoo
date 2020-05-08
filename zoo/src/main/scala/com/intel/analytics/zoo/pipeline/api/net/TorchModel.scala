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

package com.intel.analytics.zoo.pipeline.api.net

import java.util.UUID

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, QuantizedType, Storage, Tensor}
import com.intel.analytics.zoo.common.PythonInterpreter
import com.intel.analytics.zoo.feature.PythonLoaderFeatureSet
import com.intel.analytics.zoo.pipeline.api.net.TorchModel.TorchModel2Holder
import jep.{Jep, NDArray}

import scala.reflect.ClassTag

/***
 * TorchModel wraps a pytorch model as a single layer, thus this Pytorch model
 * can be used for distributed inference or training.
 * @param modelHolder model bytes
 * @param initWeights initial weights
 */
class TorchModel private(private val modelHolder: TorchModel2Holder, initWeights: Array[Float])
  extends AbstractModule[Activity, Activity, Float]{
  import TorchModel._

  protected lazy val loaded = {
    PythonInterpreter.set("model_bytes", modelHolder.torchBytes)
    val loadModelCode =
      s"""
         |import torch
         |import torch.nn as nn
         |import torch.nn.functional as F
         |import torchvision
         |def tensor_to_numpy(elements):
         |    if isinstance(elements, np.ndarray):
         |        return elements
         |    elif isinstance(elements, list):
         |        return tensor_to_list_of_numpy(elements)
         |    elif isinstance(elements, str):
         |        return elements
         |    else:
         |        return elements.numpy()
         |    results = []
         |    for element in elements:
         |        results += tensor_to_list_of_numpy(element)
         |    return results
         |
         |def tuple_to_numpy(data):
         |    return tuple([tensor_to_numpy(d) for d in data])
         |
         |from pyspark.serializers import CloudPickleSerializer
         |by = bytes(b % 256 for b in model_bytes)
         |${getName()} = CloudPickleSerializer.loads(CloudPickleSerializer, by)
         |""".stripMargin
    PythonInterpreter.exec(loadModelCode)
    true
  }

  val weights: Tensor[Float] = Tensor[Float](Storage[Float](initWeights))
  val gradients: Tensor[Float] = Tensor[Float](weights.size())

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weights), Array(gradients))
  }

  // TODO: this take about 1 second while running resnet50.
  val setWeightCode =
    s"""
        |w = torch.Tensor(list(newWeight))
        |torch.nn.utils.vector_to_parameters(w, ${getName()}.parameters())
        |
        |""".stripMargin

  val forwardCode =
    s"""
       |input = data[0]
       |target = data[1]
       |output = ${getName()}(input)
       |""".stripMargin

  override def updateOutput(input: Activity): Activity = {
    loaded
    val forwardCode = if (train) {
      PythonInterpreter.set("newWeight", weights.storage().array())
      PythonInterpreter.exec(setWeightCode)
      this.forwardCode
    } else {
      this.forwardCode
    }
    PythonInterpreter.exec(forwardCode)
    val outputNd = PythonInterpreter.getValue[NDArray[_]]("tensor_to_numpy(output.data.numpy())")
    output = PythonLoaderFeatureSet.ndArrayToTensor(outputNd)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    loaded
    val startTime = System.nanoTime()
    val backwardCode =
      s"""
        |loss.backward(retain_graph=True)
        |""".stripMargin
    PythonInterpreter.exec(backwardCode)
    val getWeightCode =
      s"""
        |grads=[]
        |for param in ${getName()}.parameters():
        |    grads.append(param.grad.view(-1))
        |grad=torch.nn.utils.parameters_to_vector(grads)
        |""".stripMargin
    PythonInterpreter.exec(getWeightCode)
    // TODO: just do a copy
    val grad = PythonLoaderFeatureSet.ndArrayToTensor(
      PythonInterpreter.getValue("grad.data.numpy()").asInstanceOf[NDArray[_]])
    gradients.copy(grad)
    println(s"backward total cost: ${(System.nanoTime() - startTime) / 1e9}")
    gradInput
  }

  override def zeroGradParameters(): Unit = {
    val zeroGradCode =
      s"""
        |for param in ${this.getName()}.parameters():
        |    param.grad.fill_(0)
        |""".stripMargin
    PythonInterpreter.exec(zeroGradCode)
    super.zeroGradParameters()
  }

  override def evaluate(): this.type = {
    super.evaluate()
    this
  }

}

object TorchModel {
  private val modelBytesRegistry = new RegistryMap[Array[Byte]]()

  @transient
  private lazy val inDriver = NetUtils.isDriver

  class TorchModel2Holder(@transient var torchBytes: Array[Byte], private var id: String)
    extends SerializationHolder {

    override def writeInternal(out: CommonOutputStream): Unit = {
      val (graphDef, _) = modelBytesRegistry.getOrCreate(id) {
        torchBytes
      }
      val len = graphDef.length
      out.writeString(id)
      if (inDriver) {
        out.writeInt(len)
        timing(s"writing ${len / 1024 / 1024}Mb torch model to stream") {
          out.write(graphDef)
        }
      } else {
        out.writeInt(0)
      }
    }

    override def readInternal(in: CommonInputStream): Unit = {
      id = in.readString()
      val (graph, _) = modelBytesRegistry.getOrCreate(id) {
        val len = in.readInt()
        assert(len >= 0, "GraphDef length should be an non-negative integer")
        val graphDef = new Array[Byte](len)
        timing("reading graph def from stream") {
          var numOfBytes = 0
          while (numOfBytes < len) {
            val read = in.read(graphDef, numOfBytes, len - numOfBytes)
            numOfBytes += read
          }
        }
        graphDef
      }

      torchBytes = graph
      id = id
    }

  }

  def apply(modelBytes: Array[Byte], weights: Array[Float]): TorchModel = {
    new TorchModel(new TorchModel2Holder(modelBytes, UUID.randomUUID().toString), weights)
  }
}

