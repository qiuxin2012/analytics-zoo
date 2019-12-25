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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.feature.PythonLoaderFeatureSet


class TorchCriterion2 extends AbstractCriterion[Activity, Activity, Float]() {
  import TorchCriterion2._

  override def updateOutput(input: Activity, target: Activity): Float = {
    output = sharedJep.getValue("loss.item()").asInstanceOf[Float]
    output
  }

  override def updateGradInput(input: Activity, target: Activity): Activity = {
    //TODO: return a empty result
    Tensor[Float]()
  }

}

object TorchCriterion2{
  protected val sharedJep = PythonLoaderFeatureSet.getOrCreateInterpreter()
}


