
# coding: utf-8

# In[1]:


import os
import heapq
import math
import time
from functools import partial
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torch import multiprocessing as mp

import utils
from neumf import NeuMF


# In[2]:


torch.__version__


# In[3]:


import sys
print(sys.version)


# In[4]:


torch.manual_seed(1)
np.random.seed(seed=1)

userCount = 128
itemCount = 100
layers = [256,256,128,64]
mfDim = 64
model1 = NeuMF(userCount, itemCount,
                   mf_dim=mfDim, mf_reg=0.,
                   mlp_layer_sizes=layers,
                   mlp_layer_regs=layers)


# In[5]:


from bigdl.nn.layer import *
from bigdl.nn.initialization_method import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.util.common import _py2java
from bigdl.nn.initialization_method import *
from bigdl.dataset import movielens
from numpy.testing import assert_allclose, assert_array_equal
from bigdl.util.engine import compare_version
from bigdl.transform.vision.image import *
from bigdl.models.utils.model_broadcast import broadcast_model
from bigdl.dataset.dataset import *
from bigdl.util.tf_utils import *


# In[6]:


def save_param(model, path):
    param = model.named_parameters()
    pytorch_weights = []
    pytorch_grads = []
    for p in param:
        name = p[0]
        pytorch_weights += [p[1].data.numpy()]
        pytorch_grads += [p[1].grad.numpy()]
    
    weights = {
        "mfUserEmbedding_weight": pytorch_weights[0],
        "mfItemEmbedding_weight": pytorch_weights[1],
        "mlpUserEmbedding_weight": pytorch_weights[2],
        "mlpItemEmbedding_weight": pytorch_weights[3],
        "fc256->256_weight": pytorch_weights[4],
        "fc256->256_bias": pytorch_weights[5],
        "fc256->128_weight": pytorch_weights[6],
        "fc256->128_bias": pytorch_weights[7],
        "fc128->64_weight": pytorch_weights[8],
        "fc128->64_bias": pytorch_weights[9],
        "fc128->1_weight": pytorch_weights[10],
        "fc128->1_bias": pytorch_weights[11],
        "mfUserEmbedding_gradWeight": pytorch_grads[0],
        "mfItemEmbedding_gradWeight": pytorch_grads[1],
        "mlpUserEmbedding_gradWeight": pytorch_grads[2],
        "mlpItemEmbedding_gradWeight": pytorch_grads[3],
        "fc256->256_gradWeight": pytorch_grads[4],
        "fc256->256_gradBias": pytorch_grads[5],
        "fc256->128_gradWeight": pytorch_grads[6],
        "fc256->128_gradBias": pytorch_grads[7],
        "fc128->64_gradWeight": pytorch_grads[8],
        "fc128->64_gradBias": pytorch_grads[9],
        "fc128->1_gradWeight": pytorch_grads[10],
        "fc128->1_gradBias": pytorch_grads[11]
    }

    save_variable_bigdl(weights, path)


# In[7]:


def save_weights(model, path):
    param = model.named_parameters()
    pytorch_weights = []
    pytorch_grads = []
    for p in param:
        name = p[0]
        pytorch_weights += [p[1].data.numpy()]
    
    weights = {
        "mfUserEmbedding_weight": pytorch_weights[0],
        "mfItemEmbedding_weight": pytorch_weights[1],
        "mlpUserEmbedding_weight": pytorch_weights[2],
        "mlpItemEmbedding_weight": pytorch_weights[3],
        "fc256->256_weight": pytorch_weights[4],
        "fc256->256_bias": pytorch_weights[5],
        "fc256->128_weight": pytorch_weights[6],
        "fc256->128_bias": pytorch_weights[7],
        "fc128->64_weight": pytorch_weights[8],
        "fc128->64_bias": pytorch_weights[9],
        "fc128->1_weight": pytorch_weights[10],
        "fc128->1_bias": pytorch_weights[11]
    }
    save_variable_bigdl(weights, path)
    
save_weights(model1, "/tmp/pyBigDL/init_model.obj")


# In[8]:


batchSize = 64
n_iteration = 10
n_epoch = 10

data = {}
for ite in range(n_iteration): 
    user_id = np.random.randint(0, userCount, size = [batchSize])
    item_id = np.random.randint(0, itemCount, size = [batchSize])
    label_ = np.random.randint(0, 2, size = [batchSize, 1]).astype(dtype=np.float32)
    data["user" + str(ite)] = user_id
    data["item" + str(ite)] = item_id
    data["label" + str(ite)] = label_
    
optimizer = torch.optim.Adam(model1.parameters(), lr=0.01, betas = (0.8, 0.9))
for e in range(n_epoch):
    for ite in range(n_iteration):
        user_id = data["user" + str(ite)]
        item_id = data["item" + str(ite)]
        label_ = data["label" + str(ite)]

        user = torch.autograd.Variable(torch.from_numpy(user_id))
        item = torch.autograd.Variable(torch.from_numpy(item_id))
        label = torch.autograd.Variable(torch.from_numpy(label_))

        criterion = nn.BCEWithLogitsLoss()
        losses = utils.AverageMeter()

        outputs = model1(user, item, sigmoid=False)
        loss = criterion(outputs, label)
        print(loss.item())
        losses.update(loss.data.item(), user.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_param(model1, "/tmp/pyBigDL/e" + str(e) + "i" + str(ite) + ".obj")
    print("\n")
    
save_variable_bigdl(data, "/tmp/pyBigDL/data.obj")    


# In[9]:


get_ipython().system('scp /tmp/pyBigDL/* xin@10.239.12.13:~/IntelAnalytics/analytics-zoo5')

