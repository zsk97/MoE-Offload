import sys
from typing import List, Optional, Tuple, Union, Dict
import time
import os
import torch
import json
from types import SimpleNamespace
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import MoEModelOutput
from datasets import load_dataset, Dataset


def load_batch(dataset, batch_indices, tokenizer):
    samples = dataset.select(batch_indices)
    prompts = samples["prompt_text"]
    prompt_data = tokenizer(
        prompts, padding=True, return_tensors="pt", return_attention_mask=True, max_length=512, truncation=True)
    prompt_ids, attention_mask = list(prompt_data.values())
    decode_ids = torch.tensor(samples["decode_ids"]).long()
    return prompt_ids, attention_mask, decode_ids


def main(args):
    if args.ipdb:
        from ipdb import set_trace
        set_trace()
    device = torch.device("cuda")
    num_experts = args.num_experts
    NUM_LABELS = 6 * num_experts
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    tokenizer.padding_side='left'
    predictor = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
    predictor.lm_head = torch.nn.Linear(predictor.config.hidden_size, NUM_LABELS, bias=False)
    predictor.load_state_dict(torch.load(args.predictor_ckpt, map_location=torch.device('cpu')))
    # predictor = predictor.to(device).eval()
    predictor = predictor.to(device).bfloat16().eval()

    dataset_name = f"marsggbo/xsum_switch{num_experts}_token_patterns"
    new_dataset_name = dataset_name.replace('_token_patterns', '_token_real_and_predicted_patterns')
    origin_dataset = load_dataset(dataset_name)['train']
    dataset = origin_dataset
    # dataset = origin_dataset.shard(num_shards=len(origin_dataset), index=0)
    indices = list(range(len(dataset)))
    batch_size = 16
    batch_indices_list = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    decoder_patterns = []
    for j, batch_indices in enumerate(batch_indices_list):
        if j%100==0:
            print(f"{j}/{len(batch_indices_list)}")
        batch_data = load_batch(dataset, batch_indices, tokenizer)
        batch_data = [x.to(device) for x in batch_data]
        prompt_ids, attention_mask, decode_ids = batch_data
        past_key_values = None
        encoder_outputs = None
        num_steps = decode_ids.shape[1]
        batch_decode_patterns = []
        topk = 3
        with torch.inference_mode():
            for step in range(num_steps):
                outputs = predictor(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decode_ids[:, step].view(-1,1),
                    past_key_values=past_key_values,
                    encoder_outputs=encoder_outputs,
                )
                past_key_values = outputs.past_key_values
                if encoder_outputs is None:
                    encoder_outputs = MoEModelOutput(
                        last_hidden_state=outputs.encoder_last_hidden_state,
                        hidden_states=outputs.encoder_hidden_states,
                        attentions=outputs.encoder_attentions,
                    )
                logits = outputs.logits # (bs, 1, NUM_LABELS)
                logits = logits.view(len(batch_indices), 1, -1, num_experts) # (bs, 1, 6, num_experts)
                top_indices = logits.topk(topk, dim=-1)[1] # (bs, 1, 6, topk)
                batch_decode_patterns.append(top_indices.cpu())
            batch_decode_patterns = torch.cat(batch_decode_patterns, dim=1) # (bs, seq_len, 6, topk)
            decoder_patterns.append(batch_decode_patterns)
    decoder_patterns = torch.cat(decoder_patterns, dim=0) # (num_samples, seq_len, 6, topk)
    new_dataset = dataset.add_column('predictor_pattern', decoder_patterns.tolist())
    new_dataset.push_to_hub(new_dataset_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark on a single GPU')

    # 添加参数
    parser.add_argument('--predictor_ckpt', type=str, help='Path to predictor checkpoint')
    parser.add_argument('--num_experts', type=int, default=32, help='number of experts per MoE layer')
    parser.add_argument('--ipdb', action='store_true', help='Enable ipdb on error')

    # 解析命令行输入
    args = parser.parse_args()
    main(args)

# CUDA_VISIBLE_DEVICES=1 python gen_predictor_pattern.py --num_experts 32 --predictor_ckpt /path/to/*pytorch_model.bin