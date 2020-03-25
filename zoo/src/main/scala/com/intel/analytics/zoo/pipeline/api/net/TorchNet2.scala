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

import java.io.{File, IOException}
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.utils.{ModelBroadcast, ModelBroadcastFactory, ModelBroadcastImp}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.{QuantizedTensor, QuantizedType, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.PythonInterpreter
import com.intel.analytics.zoo.feature.PythonLoaderFeatureSet
import com.intel.analytics.zoo.pipeline.api.Predictable
import com.intel.analytics.zoo.pipeline.api.net.TorchNet2.TorchModelHolder2
import jep.{Jep, NDArray}
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.slf4j.LoggerFactory

import scala.reflect.ClassTag
//TODO parameter length optional? Train function
class TorchNet2 private(private val modelHolder: TorchModelHolder2, init_weights: Array[Float])
  extends AbstractModule[Activity, Activity, Float]{
  import TorchNet2._

  protected lazy val loaded = {
    println(Thread.currentThread())
    PythonInterpreter.set("model_bytes", modelHolder.torchBytes)
    val loadModelCode =
      s"""
         |import torch
         |import torch.nn as nn
         |import torch.nn.functional as F
         |import torchvision
         |by = bytes(b % 256 for b in model_bytes)
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
         |${getName()} = CloudPickleSerializer.loads(CloudPickleSerializer, by)
         |criterion = nn.CrossEntropyLoss()
         |""".stripMargin
    println(Thread.currentThread())
    PythonInterpreter.exec(loadModelCode)
    true
  }

  val weights: Tensor[Float] = Tensor[Float](Storage[Float](init_weights))
  val gradients: Tensor[Float] = Tensor[Float](weights.size())

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(weights), Array(gradients))
  }

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
    // TODO: parameter from python
    loaded
    println(Thread.currentThread())
    val startTime = System.nanoTime()
    val forwardCode = if (train) {
      PythonInterpreter.set("newWeight", weights.storage().array())
      PythonInterpreter.exec(setWeightCode)
      println(s"setWeight time is ${(System.nanoTime() - startTime) / 1e9}")
      this.forwardCode +
        "\nloss = criterion(output, target)\n"
    } else {
      this.forwardCode
    }
    PythonInterpreter.exec(forwardCode)
    println(s"run forward cost: ${(System.nanoTime() - startTime) / 1e9}")
    val outputNd = PythonInterpreter.getValue[NDArray[_]]("tensor_to_numpy(output.data.numpy())")
    output = PythonLoaderFeatureSet.ndArrayToTensor(outputNd)
    println(s"forward total cost: ${(System.nanoTime() - startTime) / 1e9}")
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
    println(s"run backward cost: ${(System.nanoTime() - startTime) / 1e9}")
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

object TorchNet2 {
  private val modelBytesRegistry = new RegistryMap[Array[Byte]]()

  @transient
  private lazy val inDriver = NetUtils.isDriver

  class TorchModelHolder2(@transient var torchBytes: Array[Byte], private var id: String)
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

  def apply(modelBytes: Array[Byte], weights: Array[Float]): TorchNet2 = {
    new TorchNet2(new TorchModelHolder2(modelBytes, UUID.randomUUID().toString), weights)
  }
}

/**
 * ModelBroadcast is used to broadcast model.
 *
 * Note: If you want to use this to broadcast training model, please use value(true) to get
 * the model. And before broadcasting please make sure the model's parameter is compacted.
 *
 * @tparam T data type
 */
class TorchNet2Broadcast[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends ModelBroadcast[T] {
  // TODO:
  println("----------create TorchNet2 Broadcast")

  private var broadcastModel: Broadcast[Module[T]] = _
  private var broadcastParameters: Broadcast[Array[Tensor[T]]] = _

  /**
   * broadcast the model
   * first get and clear the weight and bias parameters from the model
   * then broadcast the parameters and model(without parameters) separately
   * @param sc    SparkContext
   * @param model model to broadcast
   * @return this
   */
  override def broadcast(sc: SparkContext, model: Module[T]): this.type = {
    val weightsBias = getAndClearWeightBias(model.parameters())
    broadcastModel = sc.broadcast(model.cloneModule())
    broadcastParameters = sc.broadcast(weightsBias)
    putWeightBias(weightsBias, model)
    initGradWeightBias(weightsBias, model)
    this
  }

  /**
   * get the broadcast model
   * put the weight and bias back to the model
   *
   * @param initGradient if init gradParameter.
   * @return model
   */
  override def value(initGradient: Boolean = false, shareWeight: Boolean = true): Module[T] = {
    val localModel = broadcastModel.value.cloneModule()
    putWeightBias(broadcastParameters.value, localModel)
    if (initGradient) {
      initGradWeightBias(broadcastParameters.value, localModel)
    }
    localModel
  }


  private def getAndClearWeightBias(parameters: (Array[Tensor[T]], Array[Tensor[T]]))
  : Array[Tensor[T]] = {
    if (parameters._1.length != 0) {
      var i = 0
      val weightsBias = new Array[Tensor[T]](parameters._1.length)
      val isQuantized = parameters._1.exists(_.getTensorType == QuantizedType)
      val (isCompacted, storage) = if (!isQuantized) {
        val storage = Storage(parameters._1(0).storage.array())
        (parameters._1.map(_.nElement()).sum == storage.length(), storage)
      } else {
        (false, null)
      }

      // get weight and bias
      while (i < parameters._1.length) {
        if (parameters._1(i) != null) {
          val wb = parameters._1(i)
          weightsBias(i) = if (isCompacted) {
            Tensor[T](storage, wb.storageOffset(), wb.size(), wb.stride())
          } else {
            Tensor[T](Storage(wb.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
          }
          i += 1
        }
      }
      // clear parameters
      clearTensor(parameters._1)
      clearTensor(parameters._2)

      weightsBias
    } else {
      // just return an empty array when parameters is empty.
      Array()
    }
  }

  private def clearTensor(tensors: Array[Tensor[T]]): Unit = {
    var i = 0
    while (i < tensors.length) {
      if (tensors(i) != null) {
        tensors(i).set()
      }
      i += 1
    }
  }

  private def putWeightBias(
                               broadcastWeightBias: Array[Tensor[T]],
                               localModel: Module[T]): Unit = {
    val localWeightBias = localModel.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        localWeightBias(i).set(broadcastWeightBias(i))
      }
      i += 1
    }
  }

  private def initGradWeightBias(
                                    broadcastWeightBias: Array[Tensor[T]],
                                    localModel: Module[T]): Unit = {
    val (localWeightBias, localGradWeightBias) = localModel.parameters()
    // init gradient with a compacted storage
    val storage = Storage[T](localGradWeightBias.map(_.nElement()).sum)
    val isQuantized = broadcastWeightBias.exists(_.getTensorType == QuantizedType)
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        val wb = broadcastWeightBias(i)
        wb.getTensorType match {
          case QuantizedType =>
            localGradWeightBias(i).set(Tensor(1))
          case _ =>
            localGradWeightBias(i).set(storage, wb.storageOffset(), wb.size(), wb.stride())
        }
      }
      i += 1
    }
  }
}


object TorchNet2Broadcast {
  def apply[@specialized(Float, Double) T: ClassTag]()
        (implicit ev: TensorNumeric[T]) : ModelBroadcast[T] = {
    new TorchNet2Broadcast[T]()
  }
}

class TorchNet2BroadcastFactory extends ModelBroadcastFactory {
  override def create[T: ClassTag]()(implicit ev: TensorNumeric[T]): ModelBroadcast[T] = {
    TorchNet2Broadcast()
  }
}
