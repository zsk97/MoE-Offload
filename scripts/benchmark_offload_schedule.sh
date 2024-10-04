#!/bin/bash
export PATH=/mnt/raid/tangzhenheng/anaconda3/envs/moe/bin:$PATH
BASE_PATH=/home/xinmatrix/hexin/datasets

# Check if exactly one argument is provided
if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: $0 {switch-32|switch-64|switch-128} [top_n]"
    exit 1
fi

if [ "$#" -eq 2 ]; then
    top_n=$2
else
    top_n=1
fi

# Read the input argument
switch=$1

# Default values for parameters
data_name="wmt16"
is_predict=true
in_order=false
seed=1234
max_new_tokens=16
num_batches=10
schedule_size=0

# Define log file and CSV file with dynamic names
log_file="offload_schedule_experiment_log.txt"
csv_file="offload_schedule_experiment_data.csv"
raw_log_file="offload_schedule_raw_experiment_log.txt"

# Initialize CSV file with headers
echo "Switch,Offload Size,Batch Size,Is Predict,In Order,Top N,Schedule Size,Seed,Max New Tokens,Num Batches,Elapsed Time,Forward Computation Time,GPU Mem(GB), Hit Rate" >> $csv_file

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
    local final_hit_rate=${14}

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
    echo "Final hit rate: $final_hit_rate" >> $log_file
    echo "----------------------------------------" >> $log_file

    # Append to CSV file
    echo "$switch,$offload_size,$batch_size,$is_predict,$in_order,$top_n,$schedule_size,$seed,$max_new_tokens,$num_batches,$elapsed_time,$forward_time,$max_gpu_mem,$final_hit_rate" >> $csv_file
}

# Execute commands based on the input argument
case $switch in
    switch-32)
        for offload_size in 28 24 16
        do
            cache_size=$((32 - offload_size))
            max_batch_size=$((cache_size * 2))
            batch_size=$max_batch_size
            while [ $batch_size -le $max_batch_size ]
            do
                schedule_size=$((2 * batch_size))
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

                cmd="python examples/benchmark_schedule.py --model_path=$BASE_PATH/switch-base-32 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                $predict_arg \
                                                $in_order_arg \
                                                --top_n=$top_n \
                                                --schedule_size=$schedule_size \
                                                --seed=$seed \
                                                --max_new_tokens=$max_new_tokens \
                                                --data_name=$data_name \
                                                --num_batches=$num_batches 2>&1 | tee -a $raw_log_file"
                echo $cmd
                output=$(eval $cmd)
                elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
                forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
                max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
                final_hit_rate=$(echo "$output" | grep "Final hit rate" | awk '{print $4}')
                log_experiment "$switch" "$offload_size" "$batch_size" "$is_predict" "$in_order" "$top_n" "$schedule_size" "$seed" "$max_new_tokens" "$num_batches" "$elapsed_time" "$forward_time" "$max_gpu_mem" "$final_hit_rate"
                batch_size=$((batch_size * 2))
            done
        done   
        ;;
    switch-64)
        for offload_size in 60 56 48
        do
            cache_size=$((64 - offload_size))
            max_batch_size=$((cache_size * 2))
            batch_size=$max_batch_size
            while [ $batch_size -le $max_batch_size ]
            do
                schedule_size=$((2 * batch_size))
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

                cmd="python examples/benchmark_schedule.py --model_path=$BASE_PATH/switch-base-64 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                $predict_arg \
                                                $in_order_arg \
                                                --top_n=$top_n \
                                                --schedule_size=$schedule_size \
                                                --seed=$seed \
                                                --max_new_tokens=$max_new_tokens \
                                                --data_name=$data_name \
                                                --num_batches=$num_batches 2>&1 | tee -a $raw_log_file"
                echo $cmd
                output=$(eval $cmd)
                elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
                forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
                max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
                final_hit_rate=$(echo "$output" | grep "Final hit rate" | awk '{print $4}')
                log_experiment "$switch" "$offload_size" "$batch_size" "$is_predict" "$in_order" "$top_n" "$schedule_size" "$seed" "$max_new_tokens" "$num_batches" "$elapsed_time" "$forward_time" "$max_gpu_mem" "$final_hit_rate"
                batch_size=$((batch_size * 2))
            done
        done
        ;;
    switch-128)
        for offload_size in 124 120 112
        do
            cache_size=$((128 - offload_size))
            max_batch_size=$((cache_size * 2))
            batch_size=$max_batch_size
            while [ $batch_size -le $max_batch_size ]
            do
                schedule_size=$((2 * batch_size))
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

                cmd="python examples/benchmark_schedule.py --model_path=$BASE_PATH/switch-base-128 \
                                                --offload_size=$offload_size \
                                                --batch_size=$batch_size \
                                                $predict_arg \
                                                $in_order_arg \
                                                --top_n=$top_n \
                                                --schedule_size=$schedule_size \
                                                --seed=$seed \
                                                --max_new_tokens=$max_new_tokens \
                                                --data_name=$data_name \
                                                --num_batches=$num_batches 2>&1 | tee -a $raw_log_file"
                echo $cmd
                output=$(eval $cmd)
                elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
                forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
                max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
                final_hit_rate=$(echo "$output" | grep "Final hit rate" | awk '{print $4}')
                log_experiment "$switch" "$offload_size" "$batch_size" "$is_predict" "$in_order" "$top_n" "$schedule_size" "$seed" "$max_new_tokens" "$num_batches" "$elapsed_time" "$forward_time" "$max_gpu_mem" "$final_hit_rate"
                batch_size=$((batch_size * 2))
            done
        done
        ;;
    *)
        echo "Invalid argument: $switch"
        echo "Usage: $0 {switch-32|switch-64|switch-128}"
        exit 1
        ;;
esac