#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import sys
import io
import torch
from importlib.util import find_spec
from zoo.pipeline.api.torch import zoo_pickle_module
from bigdl.optim.optimizer import OptimMethod

if find_spec('jep') is None:
    raise Exception("jep not found, please install jep first.")


class TorchOptim(OptimMethod):
    """
    TorchOptim wraps a torch optimizer or a learning rate scheduler for distributed training.
    """

    def __init__(self, optim_bytes, decayType, bigdl_type="float"):
        """
        :param bigdl_type:
        """
        super(TorchOptim, self).__init__(None, bigdl_type, optim_bytes, decayType)

    @staticmethod
    def from_pytorch(optim, decayType="EpochDecay"):
        """
        :param optim: Pytorch optimizer or LrScheduler.
        :param decayType: one string of EpochDecay, IterationDecay, EpochDecayByScore if
                          the optim is LrScheduler.
                          EpochDecay: call LrScheduler.step() every epoch.
                          IterationDecay: call LrScheduler.step() every iteration.
                          EpochDecayByScore: call LrScheduler.step(val_score) every epoch,
                              val_score is the return value of the first validation method.
                              ReduceLROnPlateau must use this EpochDecayByScore.
        """
        bys = io.BytesIO()
        torch.save(optim, bys, pickle_module=zoo_pickle_module)
        zoo_optim = TorchOptim(bys.getvalue(), decayType)
        return zoo_optim


class MultiStepTorchOptim(OptimMethod):
    """
    MultiStepTorchOptim wraps few TorchOptims for distributed training.
    """

    def __init__(self, torch_optims, end_epochs, bigdl_type="float"):
        """
        :param bigdl_type:
        """
        super(MultiStepTorchOptim, self).__init__(None, bigdl_type, torch_optims, end_epochs)

    @staticmethod
    def from_torch_optims(torch_optims, end_epochs):
        """
        :param torch_optims: a list of TorchOptim
        :param end_epochs: end epochs of each TorchOptim
        """
        zoo_optim = MultiStepTorchOptim(torch_optims, end_epochs)
        return zoo_optim

