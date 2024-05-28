#!/bin/bash

for offload_size in 32 48 56
do
    for batch_size in 8 16 32 64 128
    do
    python examples/benchmark_offload.py --model_path=/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-64/snapshots/b9fae596288a4a26289db5b28ab014b93fe5b9a9 \
                                        --offload_size=$offload_size \
                                        --batch_size=$batch_size \
                                        --max_new_tokens=16 \
                                        --num_batches=20 \
                                        --is_baseline 
    done
done   