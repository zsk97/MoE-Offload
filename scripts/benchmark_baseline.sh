#!/bin/bash
export PATH=/mnt/raid/tangzhenheng/anaconda3/envs/moe/bin:$PATH

# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {switch-32|switch-64|switch-128}"
    exit 1
fi

# Read the input argument
switch=$1

# Default values for parameters
is_predict=false
in_order=false
top_n=0
seed=1234
max_new_tokens=16
num_batches=20

# Define log file and CSV file with dynamic names
log_file="baseline_experiment_log.txt"
csv_file="baseline_experiment_data.csv"
raw_log_file="baseline_raw_experiment_log.txt"


# Initialize CSV file with headers
echo "Switch,Offload Size,Batch Size,Elapsed Time,Forward Computation Time" >> $csv_file

# Function to log and append to CSV
log_experiment() {
    local switch=$1
    local offload_size=$2
    local batch_size=$3
    local elapsed_time=$4
    local forward_time=$5
    local max_gpu_mem=$6

    # Log to text file
    echo "----------------------------------------" >> $log_file
    echo "Switch: $switch" >> $log_file
    echo "Offload Size: $offload_size" >> $log_file
    echo "Batch Size: $batch_size" >> $log_file
    echo "Elapsed Time: $elapsed_time" >> $log_file
    echo "Forward Computation Time: $forward_time" >> $log_file
    echo "Max GPU memory usage: $max_gpu_mem" >> $log_file
    echo "----------------------------------------" >> $log_file

    # Append to CSV file
    echo "$switch,$offload_size,$batch_size,$elapsed_time,$forward_time,$max_gpu_mem" >> $csv_file
}

# Execute commands based on the input argument
case $switch in
    switch-32)
        for offload_size in 16 24 28
        do
            for batch_size in 8 16 32 64
            do
                output=$(python examples/benchmark_offload.py --model_path=/home/tangzhenheng/hexin/data/switch-base-finetuned-wmt16/switch-base-32 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                --max_new_tokens=16 \
                                                --num_batches=20 \
                                                --is_baseline 2>&1 | tee -a $raw_log_file)
                elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
                forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
                max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
                log_experiment "$switch" "$offload_size" "$batch_size" "$elapsed_time" "$forward_time" "$max_gpu_mem"
            done
        done   
        ;;
    switch-64)
        for offload_size in 32 48 56
        do
            for batch_size in 8 16 32 64 128
            do
                output=$(python examples/benchmark_offload.py --model_path=/home/tangzhenheng/hexin/data/switch-base-finetuned-wmt16/switch-base-64 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                --max_new_tokens=16 \
                                                --num_batches=20 \
                                                --is_baseline 2>&1 | tee -a $raw_log_file)
                elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
                forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
                max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
                log_experiment "$switch" "$offload_size" "$batch_size" "$elapsed_time" "$forward_time" "$max_gpu_mem"
            done
        done
        ;;
    switch-128)
        for offload_size in 64 96 112
        do
            for batch_size in 4 8 16 32 64 128 256
            do
                output=$(python examples/benchmark_offload.py --model_path=/home/tangzhenheng/hexin/data/switch-base-finetuned-wmt16/switch-base-128 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                --max_new_tokens=16 \
                                                --num_batches=20 \
                                                --is_baseline 2>&1 | tee -a $raw_log_file)
                elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
                forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
                max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
                log_experiment "$switch" "$offload_size" "$batch_size" "$elapsed_time" "$forward_time" "$max_gpu_mem"
            done
        done
        ;;
    *)
        echo "Invalid argument: $switch"
        echo "Usage: $0 {switch-32|switch-64|switch-128}"
        exit 1
        ;;
esac