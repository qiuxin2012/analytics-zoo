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
from zoo import init_spark_on_yarn
from zoo.tfpark import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
import sys
from tensorflow.keras.models import Model


def main(max_epoch, data_num):
    num_executors = 1
    num_cores_per_executor = 1
    hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
    sc = init_spark_on_yarn(
        hadoop_conf=hadoop_conf_dir,
        conda_name=os.environ["ZOO_CONDA_NAME"],  # The name of the created conda-env
        num_executor=num_executors,
        executor_cores=num_cores_per_executor,
        executor_memory="2g",
        driver_memory="10g",
        driver_cores=1,
        spark_conf={"spark.rpc.message.maxSize": "1024",
                    "spark.task.maxFailures":  "1",
                    "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})


    # dataset = TFDataset.from_dataset(train_dataset, 20, validation_set=valid_dataset)

    from zoo.examples.tensorflow.tfpark.tf_optimizer.train_lenet import trainfunc, testfunc
    dataset = TFDataset.from_dataset(trainfunc,
                                 features=(tf.float32, [28, 28, 1]),
                                 labels=(tf.int32, []),
                                 batch_size=280,
                                 val_dataset=testfunc
                                 )

    from tensorflow.keras.layers import Input, Flatten, Dense
    data = Input(shape=[28, 28, 1])

    x = Flatten()(data)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=data, outputs=predictions)

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    optimizer = TFOptimizer.from_keras(model, dataset, model_dir="/tmp/mnist_keras")

    # kick off training
    optimizer.optimize(end_trigger=MaxEpoch(max_epoch))

    model.save_weights("/tmp/mnist_keras/mnist_keras.h5")


if __name__ == '__main__':
    max_epoch = 5
    data_num = 60000

    if len(sys.argv) > 1:
        max_epoch = int(sys.argv[1])
        data_num = int(sys.argv[2])
    main(max_epoch, data_num)
