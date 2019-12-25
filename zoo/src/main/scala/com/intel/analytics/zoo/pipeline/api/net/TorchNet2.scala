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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.feature.PythonLoaderFeatureSet
import com.intel.analytics.zoo.pipeline.api.Predictable
import com.intel.analytics.zoo.pipeline.api.net.TorchNet2.TorchModelHolder2
import jep.{Jep, NDArray}
import org.apache.commons.io.FileUtils
import org.slf4j.LoggerFactory

import scala.reflect.ClassTag
//TODO parameter length optional? Train function
class TorchNet2 private(private val modelHolder: TorchModelHolder2, parameterLength: Int)
  extends AbstractModule[Activity, Activity, Float]{
  import TorchNet2._
  sharedJep.set("model_bytes", modelHolder.torchBytes)
  val loadModelCode =
    s"""
      |by = bytes(b % 256 for b in pyjarray)
      |${getName()} = pickle.loads(by)
      |""".stripMargin
  sharedJep.exec(loadModelCode)

  val weights: Tensor[Float] = Tensor(parameterLength)
  val gradients: Tensor[Float] = Tensor(parameterLength)

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
       |output = ${getName()}(data)
       |""".stripMargin

  override def updateOutput(input: Activity): Activity = {
    // TODO: parameter from python
    sharedJep.set("newWeight", weights.storage().array())
    val forwardCode = if (train) {
      setWeightCode +
        "\nloss = F.nll_loss(output, target)\n" +
        this.forwardCode
    } else {
      this.forwardCode
    }
    sharedJep.exec(forwardCode)
    val outputNd = sharedJep.getValue("tensor_to_numpy(output)").asInstanceOf[NDArray[_]]
    output = PythonLoaderFeatureSet.ndArrayToTensor(outputNd)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    val backwardCode =
      s"""
        |loss.backward()
        |grad=torch.nn.utils.parameters_to_vector(${getName()}.parameters())
        |""".stripMargin
    sharedJep.exec(backwardCode)
    // TODO: just do a copy
    val grad = PythonLoaderFeatureSet.ndArrayToTensor(
      sharedJep.getValue("grad.numpy()").asInstanceOf[NDArray[_]])
    gradients.copy(grad)

    gradInput
  }

  override def zeroGradParameters(): Unit = {
    val zeroGradCode =
      """
        |for param in model.parameters():
        |    param.grad.fill_(0)
        |""".stripMargin
    sharedJep.exec(zeroGradCode)
    super.zeroGradParameters()
  }

  override def evaluate(): this.type = {
    super.evaluate()
    this
  }

}

object TorchNet2 {
  private val modelBytesRegistry = new RegistryMap[Array[Byte]]()
  protected val sharedJep = PythonLoaderFeatureSet.getOrCreateInterpreter()

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

  def apply(modelBytes: Array[Byte], parameterLength: Int): TorchNet2 = {
    new TorchNet2(new TorchModelHolder2(modelBytes, UUID.randomUUID().toString), parameterLength)
  }
}
