# example copy from https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
"""
Training a Classifier
=====================
This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.
Now you might be thinking,
What about data?
----------------
Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.
-  For images, packages such as Pillow, OpenCV are useful
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful
Specifically for vision, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.
This provides a huge convenience and avoids writing boilerplate code.
For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.
.. figure:: /_static/img/cifar10.png
   :alt: cifar10
   cifar10
Training an image classifier
----------------------------
We will do the following steps in order:
1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch
import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
from zoo.pipeline.api.net.torch_net import TorchNet2
from zoo.pipeline.api.net.torch_criterion import TorchCriterion2
from zoo.pipeline.estimator import *
from bigdl.optim.optimizer import SGD, Adam
from zoo.common.nncontext import *
from zoo.feature.common import FeatureSet
from bigdl.optim.optimizer import Loss
from zoo.pipeline.api.keras.metrics import Accuracy

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import numpy as np

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

num_executors = 1
num_cores_per_executor = 4
hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
sc = init_spark_on_local(cores=16, conf={"spark.driver.memory": "10g"})
#sc = init_spark_on_yarn(
#   hadoop_conf=hadoop_conf_dir,
#   conda_name=os.environ["ZOO_CONDA_NAME"],  # The name of the created conda-env
#   num_executor=num_executors,
#   executor_cores=num_cores_per_executor,
#   executor_memory="10g",
#   driver_memory="10g",
#   driver_cores=1,
#   spark_conf={"spark.rpc.message.maxSize": "1024",
#               "spark.task.maxFailures":  "1",
#               "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})

zooOptim = SGD(learningrate=0.001, momentum=0.9)
zooModel = TorchNet2.from_pytorch(net)
zooCriterion = TorchCriterion2(criterion)
estimator = Estimator(zooModel, optim_methods=zooOptim)
train_featureSet = FeatureSet.data_loader(trainloader)
test_featureSet = FeatureSet.data_loader(testloader)

from bigdl.optim.optimizer import MaxEpoch, EveryEpoch, MaxIteration
from zoo.pipeline.api.keras.metrics import Accuracy

estimator.train_minibatch(train_featureSet, zooCriterion, end_trigger=MaxEpoch(2), checkpoint_trigger=EveryEpoch(),
                          validation_set=test_featureSet, validation_method=[Accuracy()])

