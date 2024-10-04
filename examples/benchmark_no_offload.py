import os
import time
from glob import glob
import torch
import torch.nn as nn
import re
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.modeling_outputs import MoEModelOutput
import fairscale.nn.model_parallel.initialize as fs_init

from MoEOffload.load_utils import process_dataset
from MoEOffload.args import parse_args
from MoEOffload.utils import init_distributed_mode

def fix_decode_generate(input_ids,
                        decode_ids,
                        attention_mask,
                        model,
                        max_new_tokens=128,
                        past_key_values=None,
                        device=torch.device("cuda:0")):
    # 初始化生成的令牌列表和past_key_values（用于存储注意力层的状态，加速和优化生成）
    generated_tokens = []
    past = past_key_values

    decoder_input_ids = torch.tensor([[0]]*len(input_ids)).int().to(device)
    encoder_outputs = None
    
    duration = 0

    # lengths = attention_mask.sum(-1)
    # max_length = lengths.max()
    # print("Max length in current batch ", max_length)
    # print(f"Start inference")
    model.eval()  # Put model in evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for step in range(1, max_new_tokens):
            torch.cuda.nvtx.range_push(f"Step {step}")
            torch.cuda.synchronize()
            if step > 1:
                start = time.time()
            
            torch.cuda.nvtx.range_push(f"Compute")
            outputs = model(input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past,
                            encoder_outputs=encoder_outputs,
                            output_router_logits=True,
                            use_cache=True)  # use_cache允许模型返回past_key_values
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()

            if step > 1:
                duration += time.time() - start
            # print(f"Step{step}: encoder-{outputs.encoder_router_logits[1][0].shape} decoder-{outputs.decoder_router_logits[1][0].shape}")
            
            # Select the next token based on the decode_id
            next_token = decode_ids[:, step]
            next_token = torch.unsqueeze(next_token, dim=-1).to(torch.int)

            # 应用temperature来调整预测分布
            generated_tokens.append(next_token)
            
            # decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            decoder_input_ids = next_token

            # Update Key-Value cache
            past = outputs.past_key_values

            # Update encoder outputs
            if encoder_outputs is None:
                encoder_outputs = MoEModelOutput(last_hidden_state=outputs.encoder_last_hidden_state,
                                                hidden_states=outputs.encoder_hidden_states,
                                                attentions=outputs.encoder_attentions,
                                                router_probs=outputs.encoder_router_logits)
            torch.cuda.nvtx.range_pop()
    
    return duration

def benchmark_no_offload(
    state_path, 
    device, 
    batch_size,
    max_new_tokens,
    num_batches,
    is_profile=False
    ):

    pattern = r'switch-base-(16|32|64|128)'
    match = re.search(pattern, state_path)
    if match:
        # Output the captured number
        logging.info(f"Running model {match.group(0)}")
    else:
        logging.error("No match model found")
        exit(0)

    memory_function = lambda: torch.cuda.max_memory_reserved(0) / 1024 ** 3
    # memory_function = lambda: torch.cuda.memory_allocated(0) / 1024 ** 3
    prev_memory = memory_function()
    model_name = "google/" + match.group(0)
    moe_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=False)
    moe_model = moe_model.bfloat16().to(device)
    print(f"Memory for moe model: {memory_function() - prev_memory} GB")
    prev_memory = memory_function()

    num_experts_per_layer = int(match.group(1))
    # dataset = load_dataset(f"marsggbo/bigbench4switch{int(match.group(1))}_patternand_pattern_predictor_gen")['train']
    data_name = args.data_name # 'wmt16' by default or 'xsum'
    dataset = load_dataset(f"marsggbo/{data_name}_switch{num_experts_per_layer}_token_real_and_predicted_patterns_t5-small_dff2048_dmodel32")['train']
    dataset.shuffle(seed=1234)
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-32")
    tokenizer.padding_side = 'left'

    assert num_batches < len(dataset) // batch_size

    if is_profile:
        torch.cuda.cudart().cudaProfilerStart()
    batch = 0
    forward_time = 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    top_n = 1
    for input_data, decode_id, pattern in process_dataset(dataset, tokenizer, batch_size, num_experts_per_layer, top_n):
        torch.cuda.nvtx.range_push(f"Batch {batch}")
        if batch == 1:
            start_event.record()
        input_ids = input_data.input_ids.to(device)
        attention_mask = input_data.attention_mask.to(device)
        decode_input_id = decode_id.to(device)
        predict_pattern = pattern.to(device)

        forward_time += fix_decode_generate(
            input_ids, decode_input_id,
            attention_mask,
            moe_model,
            max_new_tokens=max_new_tokens
        )

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
    memory_allocated = memory_function()
    print(f"Memory for generation: {memory_allocated - prev_memory} GB")
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

    print(args)

    benchmark_no_offload(
        args.model_path,
        device,
        args.batch_size,
        args.max_new_tokens,
        args.num_batches,
        args.is_profile
    )