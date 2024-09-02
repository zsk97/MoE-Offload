import torch
import torch.nn as nn
import re
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import concurrent.futures
import fairscale.nn.model_parallel.initialize as fs_init

from MoEOffload.load_utils import process_dataset
from MoEOffload.generate import fix_decode_generate
from MoEOffload.build_model import build_offload_switch
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

    pattern = r'switch-base-(16|32|64|128)'
    match = re.search(pattern, state_path)
    if match:
        # Output the captured number
        logging.info(f"Running model {match.group(0)}")
    else:
        logging.error("No match model found")
        exit(0)

    model_name = "google/" + match.group(0)
    offload_model, cache_engine = build_offload_switch(offload_per_layer=offload_size, state_path=state_path, model_name=model_name, is_baseline=is_baseline, is_profile=is_profile)
    offload_model = offload_model.bfloat16().to(device)

    dataset_name = f"marsggbo/bigbench4switch{int(match.group(1))}_patternand_pattern_predictor_gen"
    logging.info("Load dataset " + dataset_name)
    
    dataset = load_dataset(dataset_name)['train']
    dataset.shuffle(seed=1234)
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-32")
    tokenizer.padding_side = 'left'
    compute_stream = torch.cuda.Stream()
    predict_stream = torch.cuda.Stream()

    assert num_batches < len(dataset) // batch_size

    num_decoder_sparse_layer = 6 # switch-32/64/128/256
    num_experts_per_layer = int(match.group(1))
    NUM_LABELS = num_decoder_sparse_layer * num_experts_per_layer

    predictor = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
    predictor.lm_head = nn.Linear(predictor.config.hidden_size, NUM_LABELS, bias=False)
    predictor = predictor.bfloat16().to(device)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    if is_profile:
        torch.cuda.cudart().cudaProfilerStart()
    batch = 0
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

        if is_profile and batch == 8:
            break
        
        if batch == num_batches:
            break

    if is_profile:
        torch.cuda.cudart().cudaProfilerStop()
    end_event.record()
    torch.cuda.synchronize()

    # Calculate the elapsed time in milliseconds
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time: {elapsed_time_ms} ms")
    print(f"Forward computation time: {forward_time*1000} ms")

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