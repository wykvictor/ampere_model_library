#!/user/bin/env bash

set -euxo pipefail


function run_fp16()
{
  local bs=$1
  local thread_num=$2

  AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" AIO_NUM_THREADS=${thread_num} python3 run.py \
  -b ${bs} -f pytorch --num_runs 50 -p fp32 --csv_path perf.csv
}

function run_fp32()
{
  local bs=$1
  local thread_num=$2

  AIO_NUM_THREADS=${thread_num} python3 run.py \
  -b ${bs} -f pytorch --num_runs 50 -p fp32 --csv_path perf.csv
}

run_fp32 16 128
run_fp16 16 128
exit 0

for aio_num_threads in {16,32,64,128}
do
  for bs in {1,2,4,8,16,32}
    do
      echo "run_fp16" ${bs} ${aio_num_threads}
      run_fp16 ${bs} ${aio_num_threads}
    done
done


for aio_num_threads in {16,32,64,128}
do
  for bs in {1,2,4,8,16,32}
    do
      echo "run_fp32" ${bs} ${aio_num_threads}
      run_fp32 ${bs} ${aio_num_threads}
    done
done
