import sys
from typing import List, Optional, Tuple, Union, Dict
import time
import os
import torch
import json

import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset
from transformers.modeling_outputs import MoEModelOutput
from peft import PeftModel

from MoEOffload.utils import init_distributed_mode
from MoEOffload.build_model import build_offload_switch


def prepare_data(dataset_name, dataset_list: Dict[str,int]):
    dataset_name = "tasksource/bigbench"
    names = list(dataset_list.keys())
    all_inputs = []
    for name in names:
        print(name)
        all_inputs.append(load_dataset(dataset_name, name))
    train_all_inputs = []
    # valid_all_inputs = []
    for dataset in all_inputs:
        train_all_inputs += [text for text in dataset["train"]["inputs"]]
        # valid_all_inputs += [text for text in dataset["validation"]["inputs"]]
    return train_all_inputs


def main(args):
    if args.ipdb:
        from ipdb import set_trace
        set_trace()
    dataset_name = args.dataset_name
    if dataset_name == 'tasksource/bigbench':
        dataset_list = {
            # "auto_categorization": 328,
            # "disfl_qa": 8000,
            # "linguistics_puzzles": 2000,
            "tense": 2,
            "semantic_parsing_in_context_sparc": 1160,
            "word_sorting": 1900,
        }
        print(f'Building dataset including {dataset_list}')
        data = prepare_data(dataset_name, dataset_list)
    elif dataset_name == 'wmt16':
        num_samples = 10000
        dataset = load_dataset("wmt16", "de-en", split={'train': f'train[:{num_samples}]'})['train']
        data = [text['de'] for text in dataset["translation"]]
    
    data = data*3
    ###### random order
    # indices = list(range(len(data)))
    # np.random.shuffle(indices)
    # data = np.array(data)[indices]
    ###### length-sorted order
    data = np.array(sorted(data, key=len))
    batch_size = 8
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    device = torch.device("cuda:0")
    num_experts = args.num_experts
    model_name = f"google/switch-base-{num_experts}"
    if args.state_path is None:
        state_path = f'~/.cache/huggingface/hub/models--google--switch-base-{num_experts}/snapshots/*'
        state_path = os.path.expanduser(state_path)
    else:
        state_path = args.state_path
    offload_size = 0
    is_baseline = False
    is_profile = False
    # base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # model = PeftModel.from_pretrained(base_model, state_path)
    cache_engine = None
    model, cache_engine = build_offload_switch(offload_per_layer=offload_size, state_path=state_path, model_name=model_name, is_baseline=is_baseline, is_profile=is_profile)
    model = model.bfloat16().to(device)
    max_new_tokens = 32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'

    pattern_matrices = get_pattern_matrices(model, tokenizer, cache_engine, batches, max_new_tokens, device, num_experts, dataset_name)
    return pattern_matrices


def custom_generate(
    input_ids,
    decoder_input_ids,
    attention_mask,
    model,
    cache_engine,
    num_experts,
    pattern=None,
    max_new_tokens=128,
    past_key_values=None,
    temperature=0.9,
    top_p=0.9,
    
    predictor=None
):
    if pattern is None:
        pattern = torch.zeros((24, num_experts), dtype=torch.int).to(input_ids.device)
    # 初始化生成的令牌列表和past_key_values（用于存储注意力层的状态，加速和优化生成）
    generated_tokens = [decoder_input_ids]
    past = past_key_values
    model.eval()  # Put model in evaluation mode
    with torch.inference_mode():  # Disable gradient calculation
        encoder_outputs = None
        encoder_router_indices = None
        decoder_router_indices_list = []
        for step in range(max_new_tokens):
            if cache_engine is not None: cache_engine.prefetch(pattern)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            encoder_outputs=encoder_outputs,
                            decoder_input_ids=decoder_input_ids,
                            past_key_values=past,
                            output_router_logits=True,
                            use_cache=True)  # use_cache允许模型返回past_key_values
            # 获取输出中的下一个token logits和更新past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values

            # 应用temperature来调整预测分布
            next_token_logits = next_token_logits / temperature
            filtered_logits = top_p_filtering(next_token_logits, top_p)
            probs = torch.nn.functional.softmax(filtered_logits, dim=-1)

            # 随机选择一个令牌
            next_token = torch.multinomial(probs, 1) # (batch_size , 1)
            # 将生成的令牌添加到列表和解码器输入中
            generated_tokens.append(next_token)
            decoder_input_ids = next_token
            encoder_outputs = MoEModelOutput(
                last_hidden_state=outputs.encoder_last_hidden_state,
                hidden_states=outputs.encoder_hidden_states,
                attentions=outputs.encoder_attentions,
                router_probs=outputs.encoder_router_logits,
            )
            if encoder_router_indices is None:
                encoder_router_logits = outputs.encoder_router_logits
                encoder_router_indices = [x[1] if len(x)==2 else None for x in encoder_router_logits]
            decoder_router_indices_list.append(outputs.decoder_router_logits)
        generated_tokens = torch.cat(generated_tokens, dim=-1) # (batch_size, seq_len)
        decoder_router_indices = []
        num_layers = len(decoder_router_indices_list[0])
        for i in range(num_layers):
            crt_layer_indices = None
            if i%2 ==1:
                crt_layer_indices = torch.cat([x[i][1] for x in decoder_router_indices_list], dim=1) # (batch_size, seq_len)
            decoder_router_indices.append(crt_layer_indices)
        return generated_tokens[:,:-1], (encoder_router_indices, decoder_router_indices)

def top_p_filtering(logits, top_p=0.9):
    """
    Filter a distribution of logits using nucleus (top-p) sampling

    Args:
    logits (torch.Tensor): The logits output by the model.
    top_p (float): The cumulative probability cutoff for nucleus sampling.

    Returns:
    torch.Tensor: The filtered logits.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits

def get_pattern_matrices(model, tokenizer, cache_engine, batches, max_new_tokens, device, num_experts, dataset_name):
    # Initialize a dictionary to hold the activations
    pattern_matrices = {
        # 0: {
        #     'prompt_text': "this is a prompt",
        #     'prompt_ids': [], # prompt token list
        #     'prompt_pattern': , # 大小为(seq_len, num_layers, num_experts)的 one-hot 矩阵
        #     'decode_ids': [], # deocde token list
        #     'decode_pattern': , # 大小为(seq_len, num_layers, num_experts)的 one-hot 矩阵
        # },
        # 1: {},
        # ...
    }
    for batch_idx, batch in enumerate(batches):
        if batch_idx % 100==0:
            print(f'Processing batch {batch_idx}/{len(batches)} for switch-{num_experts}')
        batch = batch.tolist()
        data = tokenizer(
            batch, return_tensors="pt", return_attention_mask=True, padding=True, truncation=True, max_length=256)
        data = {key: val.to(device) for key, val in data.items()}
        data['decoder_input_ids'] = torch.zeros(
            (data['input_ids'].shape[0], 1), dtype=torch.long, device=device)
        generated_token_ids, router_indices = custom_generate(
            **data, model=model, max_new_tokens=max_new_tokens, cache_engine=cache_engine, num_experts=num_experts
        )
        (encoder_router_indices, decoder_router_indices) = router_indices

        for i, text in enumerate(batch):
            prompt_ids = data['input_ids'][i].cpu()[data['attention_mask'][i].cpu()==1]
            decode_ids = generated_token_ids[i].detach().cpu()
            pad_len = (data['attention_mask'][i]==0).sum().item()
            pattern_matrices[len(batch)*batch_idx+i] = {
                'prompt_text': text,
                'prompt_ids': prompt_ids.tolist(),
                'decode_ids': decode_ids.tolist(),
                'prompt_pattern': [x[i, pad_len:].tolist() for x in encoder_router_indices if x is not None],
                'decode_pattern': [x[i].tolist() for x in decoder_router_indices if x is not None]
            }
    torch.save(pattern_matrices, 'switch_pattern_matrices.pt')
    hf_pattern_matrices = {
        'prompt_text': [],
        'prompt_ids': [],
        'decode_ids': [],
        'prompt_pattern': [],
        'decode_pattern': []
    }
    for i in range(len(pattern_matrices)):
        hf_pattern_matrices['prompt_text'].append(pattern_matrices[i]['prompt_text'])
        hf_pattern_matrices['prompt_ids'].append(pattern_matrices[i]['prompt_ids'])
        hf_pattern_matrices['decode_ids'].append(pattern_matrices[i]['decode_ids'])
        hf_pattern_matrices['prompt_pattern'].append(pattern_matrices[i]['prompt_pattern'])
        hf_pattern_matrices['decode_pattern'].append(pattern_matrices[i]['decode_pattern'])
    hf_pattern_matrices_dataset = Dataset.from_dict(hf_pattern_matrices)
    hf_pattern_matrices_dataset.push_to_hub(f'marsggbo/{dataset_name}_switch{num_experts}_token_patterns')
    return pattern_matrices



if __name__ == '__main__':
    import argparse
    import torch.distributed as dist
    import fairscale.nn.model_parallel.initialize as fs_init
    
    def init_env():
        # define the model
        init_distributed_mode()
        fs_init.initialize_model_parallel(dist.get_world_size())

    init_env()
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Benchmark on a single GPU')

    # 添加参数
    parser.add_argument('--dataset_name', type=str, help='dataset name', default='wmt16')
    parser.add_argument('--num_experts', type=int, help='number of experts', default=64)
    parser.add_argument('--ipdb', action='store_true', help='Enable ipdb on error')
    parser.add_argument('--state_path', type=str, help='Path to the state file', default=None)

    # 解析命令行输入
    args = parser.parse_args()
    main(args)

# torchrun --nproc_per_node=1 --master_port=26173  build_pattern_datasets.py --num_experts 64 --state_path /path/to/switch-64