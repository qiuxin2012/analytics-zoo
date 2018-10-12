#!/bin/bash
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <core number> <random seed>

THRESHOLD=0.635
BASEDIR=$(dirname -- "$0")

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Get command line parallelism
parallelism=${1:-56}

# Get command line seed
seed=${2:-1}

echo "unzip ml-20m.zip"
if unzip -o ml-20m.zip
then
    echo "Start training"
    t0=$(date +%s)
    spark-submit --master "local[$parallelism]" --driver-memory 40g \
      --conf "spark.driver.extraJavaOptions=-Dbigdl.utils.Engine.defaultPoolSize=$core" \
      --class com.intel.analytics.zoo.examples.mlperf.recommendation.NeuralCFexample \
      dist/lib/analytics-zoo-bigdl_0.7.0-SNAPSHOT-spark_2.1.0-0.3.0-SNAPSHOT-jar-with-dependencies.jar \
      --inputDir ml-20m -b 2048 -e 7 --valNeg 999 --layers 256,256,128,64 --numFactors 64 \
      --dataset ml-20m -l 0.0005 --seed $seed --threshold $THRESHOLD
    t1=$(date +%s)
	delta=$(( $t1 - $t0 ))
    echo "Finish training in $delta seconds"

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="recommendation"


	echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
else
	echo "Problem unzipping ml-20.zip"
	echo "Please run 'download_data.sh && verify_datset.sh' first"
fi

