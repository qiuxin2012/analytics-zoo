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


class TorchCriterion2(private val criterionHolder: Array[Byte])
  extends AbstractCriterion[Activity, Activity, Float]() {
  import TorchCriterion2._

  protected lazy val loaded = {
    PythonInterpreter.set("criterion_bytes", criterionHolder)
    val loadModelCode =
      s"""
         |from pyspark.serializers import CloudPickleSerializer
         |c_by = bytes(b % 256 for b in criterion_bytes)
         |${getName()} = CloudPickleSerializer.loads(CloudPickleSerializer, c_by)
         |""".stripMargin
    PythonInterpreter.exec(loadModelCode)
    true
  }

  override def updateOutput(input: Activity, target: Activity): Float = {
    loaded
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
  def apply(modelBytes: Array[Byte]): TorchCriterion2 = {
    new TorchCriterion2(modelBytes)
  }
}


