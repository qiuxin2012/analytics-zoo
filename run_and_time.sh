#!/bin/bash
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

THRESHOLD=0.635
BASEDIR=$(dirname -- "$0")

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Get command line seed
seed=${1:-1}

echo "unzip ml-20m.zip"
if unzip -o ml-20m.zip
then
    echo "Start training"
    t0=$(date +%s)
	java -Xmx40g  -Dspark.master="local[56]" -Dbigdl.utils.Engine.defaultPoolSize=56 \
	 --class com.intel.analytics.zoo.examples.mlperf.recommendation.NeuralCFexample \
	 zoo/target/lib/analytics-zoo-bigdl_0.7.0-SNAPSHOT-spark_2.1.0-0.3.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
	 --inputDir ml-20m -b 2048 -e 7 --valNeg 999 --layers 256,256,128,64 --numFactors 64 \
	 --dataset ml-20m -l 0.0005 --seed $seed
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





