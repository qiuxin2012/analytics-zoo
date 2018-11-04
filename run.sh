#!/bin/bash
times=${1:-100}
mkdir logs
for i in $(seq 1 $times)
do
  seed=`date +%s`
  echo "sh run_and_time.sh 22 $seed"
  sh run_and_time.sh 22 $seed > logs/seed$seed.log
done
