package com.intel.analytics.zoo.pipeline.api.net

import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.common.PythonInterpreter
import com.intel.analytics.zoo.feature.PythonFeatureSet
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalOptimizerUtil
import jep.NDArray
import org.apache.spark.TaskContext

import scala.reflect.ClassTag

class MultiTorchOptim[@specialized(Float, Double) T: ClassTag](
        torchOptim: Array[TorchOptim[T]],
        epochs: Array[Int] = Array(),
        iterations: Array[Int] = Array())(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {
    override def optimize(
                             feval: Tensor[T] => (T, Tensor[T]),
                             parameter: Tensor[T]): (Tensor[T], Array[T]) = {

    }

    override def clearHistory(): Unit = {

    }

    override def getLearningRate(): Double = {
    }

    override def loadFromTable(config: Table): this.type = {
      this
    }

    override def updateHyperParameter(): Unit = {
    }

    override def getHyperParameter(): String = {
    }

}


object MultiTorchOptim{
    def apply[T: ClassTag](
            torchBytes: Array[Array[Byte]],
            epochs: Array[Int],
            iterations: Array[Int])(implicit ev: TensorNumeric[T]): MultiTorchOptim[T] = {
        new MultiTorchOptim[T](torchBytes.map(TorchOptim[T]))
    }
}
