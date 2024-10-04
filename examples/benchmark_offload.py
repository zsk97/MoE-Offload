import os
from glob import glob
import torch
import torch.nn as nn
import re
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import concurrent.futures
import fairscale.nn.model_parallel.initialize as fs_init

from MoEOffload.load_utils import process_dataset
from MoEOffload.generate import fix_decode_generate
from MoEOffload.build_model import build_offload_model
from MoEOffload.args import parse_args
from MoEOffload.utils import init_distributed_mode

def benchmark_offload(state_path, 
                      device, 
                      offload_size,
                      batch_size,
                      max_new_tokens,
                      top_n,
                      num_batches,
                      is_baseline=False,
                      is_profile=False,
                      is_predict=False):

    memory_function = lambda: torch.cuda.max_memory_allocated(0) / 1024 ** 3
    # memory_function = lambda: torch.cuda.max_memory_reserved(0) / 1024 ** 3
    prev_memory = memory_function()
    data_name = args.data_name # 'wmt16' by default or 'xsum'
    print(f"Data name: {data_name}")
    if 'switch' in state_path:
        pattern = r'switch-base-(16|32|64|128)'
        match = re.search(pattern, state_path)
        if match:
            # Output the captured number
            logging.info(f"Running model {match.group(0)}")
        else:
            logging.error("No match model found")
            exit(0)

        model_name = "google/" + match.group(0)
        num_decoder_sparse_layer = 6 # switch-32/64/128/256
        num_experts_per_layer = int(match.group(1))
        NUM_LABELS = num_decoder_sparse_layer * num_experts_per_layer
        hf_dataset_name = f"marsggbo/{data_name}_switch{num_experts_per_layer}_token_real_and_predicted_patterns_t5-small_dff2048_dmodel32"
        predictor_model_name = f'marsggbo/t5-small_dff2048_dmodel32_token-pattern-predictor_switch{num_experts_per_layer}_{data_name}'

    else:
        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        num_experts_per_layer = 8
        num_layers = 32
        NUM_LABELS = num_experts_per_layer * num_layers
        hf_dataset_name = f"marsggbo/{data_name}_mixtral_token_real_and_predicted_patterns_t5-small_dff2048_dmodel32"
        predictor_model_name = f'marsggbo/t5-small_dff2048_dmodel32_token-pattern-predictor_mixtral_{data_name}'
    offload_model, cache_engine = build_offload_model(
        offload_per_layer=offload_size,
        state_path=state_path,
        model_name=model_name,
        is_baseline=is_baseline,
        is_profile=is_profile
    )
    offload_model = offload_model.bfloat16().to(device)
    print(f"Memory for offload model: {memory_function() - prev_memory} GB")
    prev_memory = memory_function()

    hf_dataset_name = f"marsggbo/xsum_switch32_token_real_and_predicted_patterns_t5-small_dff2048_dmodel32" # for quick test
    dataset = load_dataset(hf_dataset_name)['train']
    dataset.shuffle(seed=1234)
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-32")
    tokenizer.padding_side = 'left'
    compute_stream = torch.cuda.Stream()
    predict_stream = torch.cuda.Stream()

    assert num_batches < len(dataset) // batch_size

    predictor_model_name = f'marsggbo/t5-small_dff2048_dmodel32_token-pattern-predictor_switch32_xsum' # for quick test
    predictor = AutoModelForSeq2SeqLM.from_pretrained(
        predictor_model_name, ignore_mismatched_sizes=True, use_safetensors=False
    ) # to download the model weights (binary file) from huggingface
    home_path = os.path.expanduser('~')
    ckpt_path = f"{home_path}/.cache/huggingface/hub/*{predictor_model_name.split('/')[-1]}/snapshots/*/*bin"
    ckpt_path = glob(ckpt_path)[0]
    model_config = AutoConfig.from_pretrained(predictor_model_name)
    predictor = AutoModelForSeq2SeqLM.from_config(config=model_config)
    predictor.lm_head = torch.nn.Linear(predictor.config.hidden_size, NUM_LABELS, bias=False)
    predictor.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False) # for quick test
    # predictor.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)

    predictor = predictor.bfloat16().to(device)
    print(f"Memory for predictor: {memory_function() - prev_memory} GB")
    prev_memory = memory_function()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    if is_profile:
        torch.cuda.cudart().cudaProfilerStart()
    batch = 0
    hit_rate = []
    forward_time = 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for input_data, decode_id, pattern in process_dataset(dataset, tokenizer, batch_size, num_experts_per_layer, top_n):
        torch.cuda.nvtx.range_push(f"Batch {batch}")
        if batch == 1:
            start_event.record()
        input_ids = input_data.input_ids.to(device)
        attention_mask = input_data.attention_mask.to(device)
        decode_input_id = decode_id.to(device)
        predict_pattern = pattern.to(device)

        forward_time += fix_decode_generate(input_ids, decode_input_id, attention_mask, predict_pattern, offload_model, predictor, executor, cache_engine, 
                                            is_baseline=is_baseline, is_predict=is_predict, compute_stream=compute_stream, predict_stream=predict_stream, max_new_tokens=max_new_tokens)

        batch += 1
        torch.cuda.nvtx.range_pop()
        crt_hit_rate = cache_engine.get_hit_rate()
        print('Hit rate:', crt_hit_rate)
        hit_rate.append(crt_hit_rate)
        if is_profile and batch == 8:
            break
        
        if batch == num_batches:
            break

    if is_profile:
        torch.cuda.cudart().cudaProfilerStop()
    end_event.record()
    torch.cuda.synchronize()
    final_hit_rate = sum(hit_rate) / len(hit_rate)
    print(f"Final hit rate: {final_hit_rate}")
    memory_allocated = torch.cuda.max_memory_reserved(0) / 1024 ** 3
    print(f"Memory for generation: {memory_function() - prev_memory} GB")
    prev_memory = memory_function()
    print(f"Memory in the end: {memory_function()} GB")

    # Calculate the elapsed time in milliseconds
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time: {elapsed_time_ms} ms")
    print(f"Forward computation time: {forward_time*1000} ms")
    print(f"Max GPU memory usage: {memory_allocated} GB")

def init_env():
    # define the model
    init_distributed_mode()
    fs_init.initialize_model_parallel(torch.distributed.get_world_size())

if __name__ == "__main__":
    args = parse_args()
    if args.ipdb:
        from ipdb import set_trace
        set_trace()
    
    torch.manual_seed(args.seed)

    init_env()

    rank = torch.distributed.get_rank()
    device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'

    if 'mixtral' in args.model_path.lower():
        from glob import glob
        state_path = glob(f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/*")[0]
        args.model_path = state_path
    print(args)

    benchmark_offload(args.model_path,
                      device,
                      args.offload_size,
                      args.batch_size,
                      args.max_new_tokens,
                      args.top_n,
                      args.num_batches,
                      args.is_baseline,
                      args.is_profile,
                      args.is_predict)