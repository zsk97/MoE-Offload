CUDA_VISIBLE_DEVICES=5 python benchmark_offload.py --model_path="/home/nus-hx/.cache/huggingface/hub/models--google--switch-base-32/snapshots/2018338b8dad760fa7a35a754d532486ef3942f9" \
--offload_size=16 \
--batch_size=16 \
--max_new_tokens=8 \
$@