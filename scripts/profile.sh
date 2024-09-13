#!/bin/bash

# 默认参数
model_path="/home/tangzhenheng/hexin/data/switch-base-finetuned-wmt16/switch-base-32"
offload_size=16
batch_size=16
max_new_tokens=16
num_batches=20
is_profile=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      experiment="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 根据实验类型选择不同的输出文件名和脚本
case "$experiment" in
  baseline)
    output_file="offload_switch_baseline"
    script="examples/benchmark_offload.py"
    extra_args="--is_baseline"
    ;;
  overlap)
    output_file="offload_switch_overlap"
    script="examples/benchmark_offload.py"
    extra_args="--is_predict"
    ;;
  schedule)
    output_file="offload_switch_schedule"
    script="examples/benchmark_schedule.py"
    schedule_size=$((2 * $batch_size))
    extra_args="--is_predict --schedule_size $schedule_size"
    ;;
  *)
    echo "Unknown experiment type: $experiment"
    exit 1
    ;;
esac

# 构建命令
command="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -f true -x true -o $output_file python $script"

# 添加通用参数
command+=" --model_path $model_path --offload_size $offload_size --batch_size $batch_size --max_new_tokens $max_new_tokens --num_batches $num_batches --is_profile"

# 添加特定实验参数
command+=" $extra_args"

# 执行命令
echo $command
eval $command

# bash scripts/profile.sh --experiment baseline
# bash scripts/profile.sh --experiment overlap
# bash scripts/profile.sh --experiment schedule