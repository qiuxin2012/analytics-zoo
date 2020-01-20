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
import tensorflow as tf
from zoo import init_nncontext, init_spark_on_yarn
from zoo.tfpark import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
import numpy as np
import sys
from bigdl.dataset import mnist

sys.path.append("/tmp/models/research/slim")  # add the slim library
from nets import lenet

slim = tf.contrib.slim


def _normalize_train_img(img, label):
  img = (tf.cast(img, tf.float32) - mnist.TRAIN_MEAN) / mnist.TRAIN_STD
  return (img, label)


def _normalize_test_img(img, label):
  img = (tf.cast(img, tf.float32) - mnist.TEST_MEAN) / mnist.TEST_STD
  return (img, label)



def trainfunc():
    import tensorflow_datasets as tfds
    t = tfds.load(data_dir="/tmp/tfds/mnist", name="mnist", split=tfds.Split.TRAIN, as_supervised=True)
    return t.map(_normalize_train_img).shuffle(1024) \
        .batch(512).prefetch(tf.data.experimental.AUTOTUNE)


def testfunc():
    import tensorflow_datasets as tfds
    t = tfds.load(data_dir="/tmp/tfds/mnist", name="mnist", split=tfds.Split.TEST, as_supervised=True)
    return t.map(_normalize_test_img) \
        .batch(128).prefetch(tf.data.experimental.AUTOTUNE)


def main(max_epoch, data_num):
    trainfunc()

    num_executors = 2
    num_cores_per_executor = 1
    hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
    sc = init_spark_on_yarn(
        hadoop_conf=hadoop_conf_dir,
        conda_name=os.environ["ZOO_CONDA_NAME"],  # The name of the created conda-env
        num_executor=num_executors,
        executor_cores=num_cores_per_executor,
        executor_memory="3g",
        driver_memory="10g",
        driver_cores=1,
        spark_conf={"spark.rpc.message.maxSize": "1024",
                    "spark.task.maxFailures":  "1",
                    "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})

    dataset = TFDataset.from_dataset(trainfunc,
                                     train_dataset_size=60000,
                                     features=(tf.float32, [28, 28, 1]),
                                     labels=(tf.int32, []),
                                     val_dataset=testfunc,
                                     val_dataset_size=10000
                                     )

    # construct the model from TFDataset
    images, labels = dataset.tensors

    with slim.arg_scope(lenet.lenet_arg_scope()):
        logits, end_points = lenet.lenet(images, num_classes=10, is_training=True)

    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

    # create a optimizer
    optimizer = TFOptimizer.from_loss(loss, Adam(1e-3),
                                      val_outputs=[logits],
                                      val_labels=[labels],
                                      val_method=Top1Accuracy(), model_dir="/tmp/lenet/")
    # kick off training
    optimizer.optimize(end_trigger=MaxEpoch(max_epoch))

    saver = tf.train.Saver()
    saver.save(optimizer.sess, "/tmp/lenet/model")

if __name__ == '__main__':

    max_epoch = 5
    data_num = 60000

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
        data_num = int(sys.argv[2])
    main(max_epoch, data_num)
