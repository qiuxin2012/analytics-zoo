package com.intel.analytics.zoo.pipeline.api.net

import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{EngineType, Table}
import com.intel.analytics.zoo.common.PythonInterpreter
import com.intel.analytics.zoo.feature.PythonFeatureSet
import jep.NDArray
import org.apache.spark.TaskContext

import scala.reflect.ClassTag

class TorchOptim[@specialized(Float, Double) T: ClassTag](
    torchOptim: Array[Byte])(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {
  import TorchOptim._

  protected val postfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode())
  @transient
  protected lazy val optimType: OptimType = {
    val partId = TaskContext.getPartitionId()
    name = s"optim_${postfix}_${partId}"
    PythonInterpreter.set("optim_bytes", torchOptim)
    val loadModelCode =
      s"""
         |import torch
         |from torch.optim.optimizer import Optimizer
         |from torch.optim.lr_scheduler import _LRScheduler
         |from zoo.pipeline.api.torch import zoo_pickle_module
         |
         |optim_by = bytes(b % 256 for b in optim_bytes)
         |$name = torch.load(io.BytesIO(optim_by), pickle_module=zoo_pickle_module)
         |""".stripMargin
    PythonInterpreter.exec(loadModelCode)
    if (PythonInterpreter.getValue[Boolean](s"isinstance($name, Optimizer)")) {
      Optim
    } else if (PythonInterpreter.getValue[Boolean](s"isinstance($name, _LRScheduler)")) {
      LrSchedule
    } else {
      throw new IllegalArgumentException(s"Unknown optimizer type")
    }
  }

  var name = ""
  var init = false

  override def optimize(feval: Tensor[T] => (T, Tensor[T]), parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    optimType match {
      case Optim =>
        val (fx, dfdx) = feval(parameter)
        val weightName = "weight"
        if (!init) {
          PythonInterpreter.set(weightName, new NDArray[Array[Float]](
            parameter.toTensor[Float].storage().array()))
          val initCode =
            s"""
               |$weightName = torch.tensor($weightName, requires_grad=True)
               |$weightName = torch.autograd.Variable($weightName)
               |${name}.__init__([${weightName}], **${name}.defaults)
               |""".stripMargin
          PythonInterpreter.exec(initCode)
        }
        val gradientName = "gradient"
        PythonInterpreter.set("gradient", new NDArray[Array[Float]](
          dfdx.toTensor[Float].storage().array()))
        val stepCode =
          s"""
             |${weightName}.grad = torch.tensor(${gradientName})
             |${name}.step()
             |""".stripMargin
        PythonInterpreter.exec(stepCode)
        val updatedParameter = PythonFeatureSet.ndArrayToTensor(
          PythonInterpreter.getValue(s"${weightName}.data.numpy()").asInstanceOf[NDArray[_]])
        parameter.copy(updatedParameter.toTensor[T])
        (parameter, Array(fx))
      case LrSchedule =>
        throw new IllegalArgumentException()
    }

  }

  override def clearHistory(): Unit = ???

  override def getLearningRate(): Double = ???

  override def loadFromTable(config: Table): TorchOptim.this.type = {
    throw new UnsupportedOperationException()
  }

}

object TorchOptim{
  sealed trait OptimType

  case object LrSchedule extends OptimType
  case object Optim extends OptimType

  def apply[T](optimBytes: Array[Byte])(implicit ev: TensorNumeric[T]): TorchOptim[T] = {
    new TorchOptim[T](optimBytes)
  }
}
