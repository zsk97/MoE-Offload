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
is_predict=true
in_order=false
top_n=1
seed=1234
max_new_tokens=16
num_batches=20
schedule_size=0

# Define log file and CSV file with dynamic names
log_file="offload_overlap_experiment_log.txt"
csv_file="offload_overlap_experiment_data.csv"
raw_log_file="offload_overlap_raw_experiment_log.txt"

# Initialize CSV file with headers
echo "Switch,Offload Size,Batch Size,Is Predict,In Order,Top N,Schedule Size,Seed,Max New Tokens,Num Batches,Elapsed Time,Forward Computation Time,GPU Mem(GB)" >> $csv_file

# Function to log and append to CSV
log_experiment() {
    local switch=$1
    local offload_size=$2
    local batch_size=$3
    local is_predict=$4
    local in_order=$5
    local top_n=$6
    local schedule_size=$7
    local seed=$8
    local max_new_tokens=$9
    local num_batches=${10}
    local elapsed_time=${11}
    local forward_time=${12}
    local max_gpu_mem=${13}

    # Log to text file
    echo "----------------------------------------" >> $log_file
    echo "Switch: $switch" >> $log_file
    echo "Offload Size: $offload_size" >> $log_file
    echo "Batch Size: $batch_size" >> $log_file
    echo "Is Predict: $is_predict" >> $log_file
    echo "In Order: $in_order" >> $log_file
    echo "Top N: $top_n" >> $log_file
    echo "Schedule Size: $schedule_size" >> $log_file
    echo "Seed: $seed" >> $log_file
    echo "Max New Tokens: $max_new_tokens" >> $log_file
    echo "Num Batches: $num_batches" >> $log_file
    echo "Elapsed Time: $elapsed_time" >> $log_file
    echo "Forward Computation Time: $forward_time" >> $log_file
    echo "Max GPU memory usage: $max_gpu_mem" >> $log_file
    echo "----------------------------------------" >> $log_file

    # Append to CSV file
    echo "$switch,$offload_size,$batch_size,$is_predict,$in_order,$top_n,$schedule_size,$seed,$max_new_tokens,$num_batches,$elapsed_time,$forward_time,$max_gpu_mem" >> $csv_file
}

# Execute commands based on the input argument
case $switch in
    switch-32)
        for offload_size in 16 24 # 0 8 
        do
            for batch_size in 4 8 16 32 # 4 8 16 32 64
            do
                if [ "$is_predict" = true ]; then
                    predict_arg="--is_predict"
                else
                    predict_arg=""
                fi

                if [ "$in_order" = true ]; then
                    in_order_arg="--in_order"
                else
                    in_order_arg=""
                fi

                output=$(python examples/benchmark_offload.py --model_path=/home/tangzhenheng/hexin/data/switch-base-finetuned-wmt16/switch-base-32 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                $predict_arg \
                                                $in_order_arg \
                                                --top_n=$top_n \
                                                --schedule_size=$schedule_size \
                                                --seed=$seed \
                                                --max_new_tokens=$max_new_tokens \
                                                --num_batches=$num_batches 2>&1 | tee -a $raw_log_file)
                elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
                forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
                max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
                log_experiment "$switch" "$offload_size" "$batch_size" "$is_predict" "$in_order" "$top_n" "$schedule_size" "$seed" "$max_new_tokens" "$num_batches" "$elapsed_time" "$forward_time" "$max_gpu_mem"
            done
        done   
        ;;
    switch-64)
        for offload_size in 32 48 # 0 8 16
        do
            for batch_size in 8 16 32 64
            do
                if [ "$is_predict" = true ]; then
                    predict_arg="--is_predict"
                else
                    predict_arg=""
                fi

                if [ "$in_order" = true ]; then
                    in_order_arg="--in_order"
                else
                    in_order_arg=""
                fi

                output=$(python examples/benchmark_offload.py --model_path=/home/tangzhenheng/hexin/data/switch-base-finetuned-wmt16/switch-base-64 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                $predict_arg \
                                                $in_order_arg \
                                                --top_n=$top_n \
                                                --schedule_size=$schedule_size \
                                                --seed=$seed \
                                                --max_new_tokens=$max_new_tokens \
                                                --num_batches=$num_batches 2>&1 | tee -a $raw_log_file)
                elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
                forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
                max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
                log_experiment "$switch" "$offload_size" "$batch_size" "$is_predict" "$in_order" "$top_n" "$schedule_size" "$seed" "$max_new_tokens" "$num_batches" "$elapsed_time" "$forward_time" "$max_gpu_mem"
            done
        done
        ;;
    switch-128)
        for offload_size in 64 96 112 # 16 32 
        do
            for batch_size in 16 32 64 128 # 8 16 32 64
            do
                if [ "$is_predict" = true ]; then
                    predict_arg="--is_predict"
                else
                    predict_arg=""
                fi

                if [ "$in_order" = true ]; then
                    in_order_arg="--in_order"
                else
                    in_order_arg=""
                fi

                output=$(python examples/benchmark_offload.py --model_path=/home/tangzhenheng/hexin/data/switch-base-finetuned-wmt16/switch-base-128 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                $predict_arg \
                                                $in_order_arg \
                                                --top_n=$top_n \
                                                --schedule_size=$schedule_size \
                                                --seed=$seed \
                                                --max_new_tokens=$max_new_tokens \
                                                --num_batches=$num_batches 2>&1 | tee -a $raw_log_file)
                elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
                forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
                max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
                log_experiment "$switch" "$offload_size" "$batch_size" "$is_predict" "$in_order" "$top_n" "$schedule_size" "$seed" "$max_new_tokens" "$num_batches" "$elapsed_time" "$forward_time" "$max_gpu_mem"
            done
        done
        ;;
    *)
        echo "Invalid argument: $switch"
        echo "Usage: $0 {switch-32|switch-64|switch-128}"
        exit 1
        ;;
esac