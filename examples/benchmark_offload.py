import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import concurrent.futures
from accessory.util import misc
import fairscale.nn.model_parallel.initialize as fs_init

from MoEOffload.load_utils import process_dataset
from MoEOffload.generate import fix_decode_generate
from MoEOffload.build_model import build_offload_switch
from MoEOffload.args import parse_args

def benchmark_offload(state_path, 
                      device, 
                      offload_size,
                      batch_size,
                      max_new_tokens,
                      is_baseline=False,
                      is_profile=False):
    offload_model, cache_engine = build_offload_switch(offload_per_layer=offload_size, state_path=state_path)
    offload_model = offload_model.bfloat16().to(device)

    dataset = load_dataset("marsggbo/bigbench4switch32_pattern_predictor_tmp")
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-32")
    tokenizer.padding_side = 'left'
    compute_stream = torch.cuda.Stream()
    predict_stream = torch.cuda.Stream()

    num_decoder_sparse_layer = 6 # switch-32/64/128/256
    num_experts_per_layer = 32
    NUM_LABELS = num_decoder_sparse_layer * num_experts_per_layer

    predictor = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
    predictor.lm_head = nn.Linear(predictor.config.hidden_size, NUM_LABELS, bias=False)
    predictor = predictor.bfloat16().to(device)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    if is_profile:
        torch.cuda.cudart().cudaProfilerStart()
    batch = 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for input_data, decode_id, pattern in process_dataset(dataset, tokenizer, batch_size):
        torch.cuda.nvtx.range_push(f"Batch {batch}")
        if batch == 1:
            start_event.record()
        input_ids = input_data.input_ids.to(device)
        attention_mask = input_data.attention_mask.to(device)
        decode_input_id = decode_id.to(device)
        predict_pattern = pattern.to(device)

        print("predict pattern shape ", predict_pattern.shape)

        fix_decode_generate(input_ids, decode_input_id, attention_mask, predict_pattern, offload_model, predictor, executor, cache_engine, 
                            is_baseline=is_baseline, compute_stream=compute_stream, predict_stream=predict_stream, max_new_tokens=max_new_tokens)

        batch += 1
        torch.cuda.nvtx.range_pop()

        if batch == 8:
            break
    
    if is_profile:
        torch.cuda.cudart().cudaProfilerStop()
    end_event.record()
    torch.cuda.synchronize()

    # Calculate the elapsed time in milliseconds
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time: {elapsed_time_ms} ms")
    torch.cuda.cudart().cudaProfilerStop()
    return

def init_env():
    # define the model
    misc.init_distributed_mode()
    fs_init.initialize_model_parallel(torch.distributed.get_world_size())

if __name__ == "__main__":
    args = parse_args()
    
    init_env()

    rank = torch.distributed.get_rank()
    device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'

    benchmark_offload(args.model_path,
                      device,
                      args.offload_size,
                      args.batch_size,
                      args.max_new_tokens,
                      args.is_baseline,
                      args.is_profile)