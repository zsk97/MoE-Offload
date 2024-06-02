nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx --capture-range=cudaProfilerApi \
--capture-range-end=stop -f true -x true -o offload_switch \
python benchmark_offload.py \
--model_path="/home/nus-hx/.cache/huggingface/hub/models--google--switch-base-32/snapshots/2018338b8dad760fa7a35a754d532486ef3942f9" \
--offload_size=16 \
--batch_size=16 \
--max_new_tokens=7 \
--is_profile \ 
--is_predict