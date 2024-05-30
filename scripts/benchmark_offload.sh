#!/bin/bash

# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {switch-32|switch-64|switch-128}"
    exit 1
fi

# Read the input argument
switch=$1

# Execute commands based on the input argument
case $switch in
    switch-32)
        for offload_size in 32 48 56
        do
            for batch_size in 8 16 32 64 128
            do
            python examples/benchmark_offload.py --model_path=/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-32/snapshots/2018338b8dad760fa7a35a754d532486ef3942f9 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                --max_new_tokens=16 \
                                                --num_batches=20 \
                                                --top_n=1
            done
        done   
        ;;
    switch-64)
        for offload_size in 32 48 56
        do
            for batch_size in 8 16 32 64 128
            do
            python examples/benchmark_offload.py --model_path=/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-64/snapshots/b9fae596288a4a26289db5b28ab014b93fe5b9a9 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                --max_new_tokens=16 \
                                                --num_batches=20 \
                                                --top_n=1 
            done
        done
        ;;
    switch-128)
        for offload_size in 64
        do
            for batch_size in 256
            do
            python examples/benchmark_offload.py --model_path=/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-128/snapshots/86c815ec05361a33a8b49fc717277da9c0a4e711 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                --max_new_tokens=16 \
                                                --num_batches=20 \
                                                --top_n=1
            done
        done
        ;;
    *)
        echo "Invalid argument: $switch"
        echo "Usage: $0 {switch-16|switch-32|switch-64}"
        exit 1
        ;;
esac