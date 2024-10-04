#!/bin/bash
export PATH=/mnt/raid/tangzhenheng/anaconda3/envs/moe/bin:$PATH
BASE_PATH=/home/xinmatrix/hexin/datasets

# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {switch-32|switch-64|switch-128}"
    exit 1
fi

# Read the input argument
switch=$1

# Default values for parameters
data_name="wmt16"
is_profile=false
seed=1234
max_new_tokens=16
num_batches=20

# Define log file and CSV file with dynamic names
log_file="no_offload_experiment_log.txt"
csv_file="no_offload_experiment_data.csv"
raw_log_file="no_offload_raw_experiment_log.txt"


# Initialize CSV file with headers
echo "Switch,Batch Size,Max New Tokens,Num Batches,Elapsed Time,Forward Computation Time,GPU Mem(GB)" >> $csv_file

# Function to log and append to CSV
log_experiment() {
    local switch=$1
    local batch_size=$2
    local max_new_tokens=$3
    local num_batches=$4
    local elapsed_time=$5
    local forward_time=$6
    local max_gpu_mem=$7

    # Log to text file
    echo "----------------------------------------" >> $log_file
    echo "Switch: $switch" >> $log_file
    echo "Batch Size: $batch_size" >> $log_file
    echo "Max New Tokens: $max_new_tokens" >> $log_file
    echo "Num Batches: $num_batches" >> $log_file
    echo "Elapsed Time: $elapsed_time" >> $log_file
    echo "Forward Computation Time: $forward_time" >> $log_file
    echo "Max GPU memory usage: $max_gpu_mem" >> $log_file
    echo "----------------------------------------" >> $log_file

    # Append to CSV file
    echo "$switch,$batch_size,$max_new_tokens,$num_batches,$elapsed_time,$forward_time,$max_gpu_mem" >> $csv_file
}

# Execute commands based on the input argument
case $switch in
    switch-32)
        for batch_size in 8 16 32
        do
            cmd="python examples/benchmark_no_offload.py --model_path=$BASE_PATH/switch-base-32 \
                                            --batch_size=$batch_size \
                                            --max_new_tokens=$max_new_tokens \
                                            --num_batches=$num_batches \
                                            --data_name=$data_name \
                                            2>&1 | tee -a $raw_log_file"
            echo $cmd
            output=$(eval $cmd)
            elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
            forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
            max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
            log_experiment "$switch" "$batch_size" "$max_new_tokens" "$num_batches" "$elapsed_time" "$forward_time" "$max_gpu_mem"
        done   
        ;;
    switch-64)
        for batch_size in 8 16 32
        do
            cmd="python examples/benchmark_no_offload.py --model_path=$BASE_PATH/switch-base-64 \
                                            --batch_size=$batch_size \
                                            --max_new_tokens=$max_new_tokens \
                                            --num_batches=$num_batches \
                                            --data_name=$data_name \
                                            2>&1 | tee -a $raw_log_file"
            echo $cmd
            output=$(eval $cmd)
            elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
            forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
            max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
            log_experiment "$switch" "$batch_size" "$max_new_tokens" "$num_batches" "$elapsed_time" "$forward_time" "$max_gpu_mem"
        done
        ;;
    switch-128)
        for batch_size in 8 16 32
        do
            cmd="python examples/benchmark_no_offload.py --model_path=$BASE_PATH/switch-base-128 \
                                            --batch_size=$batch_size \
                                            --max_new_tokens=$max_new_tokens \
                                            --num_batches=$num_batches \
                                            --data_name=$data_name \
                                            2>&1 | tee -a $raw_log_file"
            echo $cmd
            output=$(eval $cmd)
            elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
            forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
            max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
            log_experiment "$switch" "$batch_size" "$max_new_tokens" "$num_batches" "$elapsed_time" "$forward_time" "$max_gpu_mem"
        done
        ;;
    *)
        echo "Invalid argument: $switch"
        echo "Usage: $0 {switch-32|switch-64|switch-128}"
        exit 1
        ;;
esac