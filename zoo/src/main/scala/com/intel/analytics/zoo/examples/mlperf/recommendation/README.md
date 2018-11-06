# 1. Problem
This task benchmarks recommendation with implicit feedback on the [MovieLens 20 Million (ml-20m) dataset](https://grouplens.org/datasets/movielens/20m/) with a [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569) model.
The model trains on binary information about whether or not a user interacted with a specific item.

# 2. Directions
### Steps to configure machine

1. Install [JDK 1.8](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html), [Maven 3.3.9](https://archive.apache.org/dist/maven/maven-3/3.3.9/binaries/), and [Spark 2.1.0](https://archive.apache.org/dist/spark/spark-2.1.0/spark-2.1.0-bin-hadoop2.7.tgz). Make sure `mvn`, `java`, `spark-submit` works in shell.
2. Install `unzip` and `curl`

```bash
sudo apt-get install unzip curl
```
3. Checkout the repo
```bash
git clone -b ncf-new https://github.com/qiuxin2012/analytics-zoo.git
```

4. Build from source

```bash
cd analytics-zoo
./make-dist.sh -P spark_2.x
```
Maven will take a few minutes to download dependencies the first time.

### Steps to download and verify data

You can download and verify the dataset by running the `download_dataset.sh` and `verify_dataset.sh` scripts in the parent directory:

```bash
# Creates ml-20.zip
bash download_dataset.sh
# Confirms the MD5 checksum of ml-20.zip
bash verify_dataset.sh
```

### Steps to run and time


Run the `run_and_time.sh` script with an integer parallelism and seed.
CORE is the parallel number of BigDL, the best practice is half of the physical core number of the machine. 28 for skylake 8180. SEED is an interger value.

```bash
bash run_and_time.sh PARALLELISM SEED
```

### Run on Skylake 8180

Run `run.sh` script with an integer `n times`, it will run `n times` of ncf training with the seed of time seconds, and write logs to folder `./logs`.
```bash
bash run.sh <run times>
```

# 3. Dataset/Environment
### Publication/Attribution
Harper, F. M. & Konstan, J. A. (2015), 'The MovieLens Datasets: History and Context', ACM Trans. Interact. Intell. Syst. 5(4), 19:1--19:19.

### Data preprocessing

1. Unzip
2. Remove users with less than 20 reviews
3. Create training and test data separation described below

### Training and test data separation
Positive training examples are all but the last item each user rated.
Negative training examples are randomly selected from the unrated items for each user.

The last item each user rated is used as a positive example in the test set.
A fixed set of 999 unrated items are also selected to calculate hit rate at 10 for predicting the test item.

### Training data order
Data is traversed randomly with 4 negative examples selected on average for every positive example.


# 4. Model
### Publication/Attribution
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569). In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

The author's original code is available at [hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering).

# 5. Quality
### Quality metric
Hit rate at 10 (HR@10) with 999 negative items.

### Quality target
HR@10: 0.635

### Evaluation frequency
After every epoch through the training data.

### Evaluation thoroughness

Every users last item rated, i.e. all held out positive examples.

# 6. Optimizations
Compare to reference code, we have done some optimizations. See [Optimization](Optimization.md) for details.
