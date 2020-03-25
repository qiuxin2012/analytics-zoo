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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.common.PythonInterpreter
import com.intel.analytics.zoo.feature.PythonLoaderFeatureSet
import com.intel.analytics.zoo.pipeline.api.net.TorchCriterion2.TorchCriterion2Holder


class TorchCriterion2(private val criterionHolder: TorchCriterion2Holder)
  extends AbstractCriterion[Activity, Activity, Float]() {
  import TorchCriterion2._

  protected lazy val loaded = {
    PythonInterpreter.set("model_bytes", criterionHolder.torchBytes)
    val loadModelCode =
      s"""
         |from pyspark.serializers import CloudPickleSerializer
         |${getName()} = CloudPickleSerializer.loads(CloudPickleSerializer, by)
         |""".stripMargin
    PythonInterpreter.exec(loadModelCode)
    true
  }

  override def updateOutput(input: Activity, target: Activity): Float = {
    PythonInterpreter.exec(s"loss = ${getName()}(output, target)")
    output = PythonInterpreter.getValue("loss.item()").asInstanceOf[Double].toFloat
    output
  }

  override def updateGradInput(input: Activity, target: Activity): Activity = {
    //TODO: return a empty result
    Tensor[Float]()
  }

  final def getName() : String = {
    s"${this.getClass.getSimpleName}${Integer.toHexString(java.util.UUID.randomUUID().hashCode())}"
  }

}

object TorchCriterion2{

  private val modelBytesRegistry = new RegistryMap[Array[Byte]]()

  @transient
  private lazy val inDriver = NetUtils.isDriver

  class TorchCriterion2Holder(@transient var torchBytes: Array[Byte], private var id: String)
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

  def apply(modelBytes: Array[Byte]): TorchCriterion2 = {
    new TorchCriterion2(new TorchCriterion2Holder(modelBytes, UUID.randomUUID().toString))
  }
}


