from MoEOffload.expert_cache import ExpertCache, ExpertCacheV1
from MoEOffload.custom_layers import MixtralMoeWrapper
from MoEOffload.expert_wrapper import MixtralExpertWrapper
from MoEOffload.utils import (nested_flatten, nested_pack, with_default_dtype,
                              forward_pre_hook, forward_post_hook)

from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from MoEOffload.models.mixtral import MixtralForCausalLM, MixtralBlockSparseTop2MLP

from dataclasses import dataclass
from transformers import AutoConfig
from safetensors.torch import load_file

from tqdm.auto import trange
from glob import glob
import json
import os
import typing as tp
import torch
import logging


@dataclass(frozen=True)
class OffloadConfig:
    main_size: int
    offload_size: int
    buffer_size: int
    offload_per_layer: int

def make_empty_expert(
        model_config: MixtralConfig
    ) -> MixtralBlockSparseTop2MLP:
        return MixtralBlockSparseTop2MLP(
            model_config,
        )

def make_and_load_expert_wrapper(
    config: MixtralConfig,
    states_dir: str,
    expert_uid: tuple[int, int],
    device: torch.device,
) -> MixtralExpertWrapper:
    layer_idx, expert_idx = expert_uid

    index_path = os.path.join(states_dir, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        module_idx = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
        weight_map = json.load(f)["weight_map"]
    state_fpaths = []
    module_names = []
    for i in range(1, 4):
        module_name = f"{module_idx}.w{i}.weight"
        module_names.append(module_name)
        path = weight_map[module_name]
        if path not in state_fpaths:
            state_fpaths.append(path)

    state_dict = {}
    for state_fpath in state_fpaths:
        state_dict.update(load_file(os.path.join(states_dir, state_fpath), device=str(device)))
    module_state_dict = {module_name.replace(f"{module_idx}.", ''): state_dict[module_name] for module_name in module_names}

    expert = make_empty_expert(config).bfloat16()
    expert.load_state_dict(module_state_dict, strict=True)

    return MixtralExpertWrapper(expert, device)

def build_offload_mixtral(
        offload_per_layer: int=4,
        state_path: str='/home/nus-hx/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/*',
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        buffer_size:int = 4,
        device = torch.device("cuda:0"),
        config=None,
        is_baseline=False,
        is_profile=False,
    ):
 
    if config is None:
        config = AutoConfig.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        config.offload = True

    # config.num_hidden_layers = 2
    num_layers = config.num_hidden_layers
    num_experts = config.num_local_experts
    num_expert_layers = config.num_hidden_layers
    offload_config = OffloadConfig(
        main_size=num_expert_layers * (num_experts - offload_per_layer),
        offload_size=num_expert_layers * num_experts,
        buffer_size=buffer_size,
        offload_per_layer=offload_per_layer,
    )

    def _make_module():
        config = AutoConfig.from_pretrained(model_name)
        expert = make_empty_expert(config).bfloat16()
        return MixtralExpertWrapper(expert, device=device)
    
    MoEWrapper = None
    expertCache = None

    if is_baseline:
        expertCache = ExpertCache
    else:
        expertCache = ExpertCacheV1
    MoEWrapper = MixtralMoeWrapper
    
    with with_default_dtype(torch.bfloat16):
        model = MixtralForCausalLM(config)
    
    model_config = config
    expert_cache = expertCache(
        make_module=_make_module,
        main_size=offload_config.main_size,
        offload_size=offload_config.offload_size,
        buffer_size=offload_config.buffer_size,
        num_layer=num_layers
    )

    state_index_path = os.path.join(state_path, "model.safetensors.index.json")
    with open(state_index_path) as f:
        weight_map = json.load(f)["weight_map"]
    trunk_state_dict = {}
    state_file_paths = set(weight_map.values())
    for state_file_path in state_file_paths:
        state_dict = load_file(os.path.join(state_path, state_file_path))
        trunk_state_dict.update({k:v for k,v in state_dict.items() if 'expert' not in k})
    model.load_state_dict(trunk_state_dict, strict=False)

    for layer_idx in trange(model_config.num_hidden_layers, desc="Loading experts"):
        curr_layer = model.model.layers[layer_idx]
        curr_layer.block_sparse_moe = MixtralMoeWrapper(
            model_config,
            layer_idx,
            curr_layer.block_sparse_moe.gate,
            expert_cache,
        )

        for expert_idx in range(model_config.num_local_experts):
            do_offload = expert_idx < offload_config.offload_per_layer

            expert_wrapper = make_and_load_expert_wrapper(
                config=model_config,
                states_dir=state_path,
                expert_uid=(layer_idx, expert_idx),
                device="cpu",
            )

            expert_cache.add_expert(
                uid=(layer_idx, expert_idx),
                module=expert_wrapper,
                eviction_group=layer_idx,
                offload=do_offload,
            )

            del expert_wrapper
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    if is_profile:
        logging.info("Add model hooking for profiling")
        for module in model.modules():
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_post_hook)
    return model, expert_cache

def custom_generate(
    input_ids,
    attention_mask,
    model,
    max_new_tokens=32,
    past_key_values=None,
    temperature=0.9,
    top_p=0.9
):
    """
    Generate text from an input using caching and sampling techniques.

    Args:
    input_ids (torch.Tensor): Tensor of token ids to be fed to the model.
    attention_mask (torch.Tensor): Tensor representing the attention mask.
    model (transformers.PreTrainedModel): The model to use for generating text.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer associated with the model.
    max_new_tokens (int): Maximum number of tokens to generate.
    temperature (float): Sampling temperature for controlling generation randomness.
    top_p (float): Nucleus sampling cutoff probability.

    Returns:
    torch.Tensor: Tensor containing the generated token ids.
    """
    model.eval()  # Put model in evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        # Initialize variables to store outputs and past_key_values
        generated_token_ids = []
        crt_tokens = input_ids
        router_logits = []

        for _ in range(max_new_tokens+1):
            outputs = model(
                input_ids=crt_tokens,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_router_logits=True,
                use_cache=True  # Informs the model to return past key-values
            )

            # Update past_key_values for the next iteration
            past_key_values = outputs.past_key_values

            # Obtain logits
            logits = outputs.logits[:, -1, :] / temperature

            # Apply top-p nucleus sampling
            if top_p is not None:
                filtered_logits = top_p_filtering(logits, top_p=top_p)
            else:
                filtered_logits = logits
            probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)

            # Sample from the filtered distribution
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            crt_tokens = next_token_id
            generated_token_ids.append(next_token_id)

            # Update the attention_mask for new token
            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=attention_mask.device)], dim=-1)
            router_logits.append(outputs.router_logits) # outputs.router_logits 的长度是 num_layers, 每个元素是 tuple,维度是(num_tokens, num_experts)

        prompt_token_ids = input_ids
        generated_token_ids = torch.cat(generated_token_ids[:-1], dim=1)
        
        num_layers = len(router_logits[0])
        bs, input_len = input_ids.shape
        prompt_router_logits = router_logits[0] # (num_layers, num_tokens, num_experts)
        prompt_pattern = torch.stack(prompt_router_logits, dim=0) # (num_layers, num_tokens, num_experts), num_tokens=bs*input_len
        prompt_pattern = prompt_pattern.view(num_layers, bs, input_len, -1)
        prompt_pattern = torch.permute(prompt_pattern, (1, 0, 2, 3)) # (bs, num_layers, input_len, num_experts)，后面需要通过top-1转换成(bs, num_layers, input_len)
        
        decode_router_logits = router_logits[1:] # (num_steps, num_layers, num_tokens, num_experts), num_tokens=bs*1
        decode_pattern = []
        for step_idx in range(len(decode_router_logits)):
            step_logits = decode_router_logits[step_idx] # tuple： num_layers个维度(bs, num_experts)的元素
            step_decode_pattern = torch.stack(step_logits, dim=0) # 转换成 tensor，维度是(num_layers, bs, num_experts)
            step_decode_pattern = step_decode_pattern.permute(1, 0, 2) # (bs, num_layers, num_experts)
            decode_pattern.append(step_decode_pattern) # (bs, num_layers, num_experts)
        decode_pattern = torch.stack(decode_pattern, dim=0) # (num_steps, bs, num_layers, num_experts)
        decode_pattern = decode_pattern.permute(1, 2, 0, 3) # (bs, num_layers, num_steps, num_experts)
        return prompt_token_ids, generated_token_ids, prompt_pattern.topk(2)[1], decode_pattern.topk(2)[1]


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


if __name__ == "__main__":
    import os
    from glob import glob
    from MoEOffload.utils import init_env
    init_env()
    state_path = glob(f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/*")[0]
    print(state_path)
    model, expert_cache = build_offload_mixtral(
        offload_per_layer=7,
        buffer_size=4,
        state_path=state_path,
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )
    model = model.bfloat16().cuda()
    num_layers = 32
    pattern = torch.zeros((num_layers, 8), dtype=torch.int).cuda()
    expert_cache.prefetch(pattern)
    # model = AutoModelForCausalLM.from_pretrained(
    #     "mistralai/Mixtral-8x7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map=torch.device("cuda:0"))
    model.eval()

    #########################################################
    # 构建routing path dataset
    #########################################################
    import datasets
    from transformers import AutoConfig, AutoTokenizer

    for data_name in ['xsum', 'wmt16']:
        if data_name == 'xsum':
            prefix = "summarize: "
        elif data_name == 'wmt16':
            prefix = "translate French to English: "
        dataset_name = f"marsggbo/{data_name}_switch32_token_real_and_predicted_patterns_t5-small_dff2048_dmodel32"
        ds = datasets.load_dataset(dataset_name)
        all_prompt_text = ds['train']['prompt_text']
        bs = 16
        batch_text = [prefix + all_prompt_text[i] for i in range(0, len(all_prompt_text), bs)]

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        
        dataset_for_predictor = {
            "prompt_text": [],
            "prompt_ids": [],
            "decode_ids": [],
            "prompt_pattern": [],
            "decode_pattern": []
        }

        max_tokens = 16
        rank = 0
        device = torch.device(f"cuda:{rank}")
        for batch_idx, batch_data in enumerate(batch_text):
            batch_data = [prefix+x for x in batch_data]
            # if batch_idx == 2:
            #     break
            data = tokenizer(batch_data, return_tensors="pt", padding=True, return_attention_mask=True)
            input_ids = data.input_ids.to(rank)
            attention_mask = data.attention_mask.to(rank)
            decoder_input_ids = torch.tensor([[0]]*len(input_ids)).int().to(device)
            prompt_token_ids, generated_ids, prompt_patterns, decode_patterns = custom_generate(
                input_ids, attention_mask, model, max_new_tokens=max_tokens
            )
            attention_mask = attention_mask.cpu().bool() # (bs, input_len)
            for i in range(len(generated_ids)):
                dataset_for_predictor['prompt_text'].append(batch_data[i])
                unpadded_input_ids = input_ids[i][attention_mask[i]]
                dataset_for_predictor['prompt_ids'].append(unpadded_input_ids.cpu())
                dataset_for_predictor['decode_ids'].append(generated_ids[i].cpu())
                pattern_shape = prompt_patterns[0].shape # (num_layers, input_len, top2)
                prompt_pattern = prompt_patterns[i][:,attention_mask[i]]
                dataset_for_predictor['prompt_pattern'].append(prompt_pattern)
                dataset_for_predictor['decode_pattern'].append(decode_patterns[i])
                print(f"{i} Q: {tokenizer.decode(input_ids[i].cpu().numpy().tolist(), skip_special_tokens=True)}")
                print(f"{i} A: {tokenizer.decode(generated_ids[i].cpu().numpy().tolist(), skip_special_tokens=True)}")
        dataset_for_predictor = datasets.Dataset.from_dict(dataset_for_predictor)
        dataset_for_predictor.push_to_hub(f'marsggbo/{data_name}_mixtral8x7bInstructv0.1_token_patterns')
