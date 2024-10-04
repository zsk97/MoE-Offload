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

MODEL_STATE_DICT = None

pretrained_switch_weights_map = {
    'google/switch-base-8': {
        'file_type': 'bin',
        'index_file': None
    },
    'google/switch-base-16': {
        'file_type': 'bin',
        'index_file': None
    },
    'google/switch-base-32': {
        'file_type': 'bin',
        'index_file': 'pytorch_model.bin.index.json'
    },
    'google/switch-base-64': {
        'file_type': 'bin',
        'index_file': 'pytorch_model.bin.index.json'
    },
    'google/switch-base-128': {
        'file_type': 'bin',
        'index_file': 'pytorch_model.bin.index.json'
    },
    'google/switch-base-256': {
        'file_type': 'bin',
        'index_file': 'pytorch_model.bin.index.json'
    },
    'google/switch-large-128': {
        'file_type': 'safetensors',
        'index_file': 'model.safetensors.index.json'
    },
}

global_state_cache = {}

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

    if MODEL_STATE_DICT is not None:
        assert len(global_state_cache) == 0
        non_expert_dict = {}
        for key, val in MODEL_STATE_DICT.items():
            if 'expert' not in key:
                non_expert_dict[key] = val
        model.load_state_dict(non_expert_dict, True)

    if len(global_state_cache) != 0:
        assert MODEL_STATE_DICT is None
        non_expert_dict = {}
        for _, state_dict in global_state_cache.items():
            for key, value in state_dict.items():
                if 'expert' not in key:
                    non_expert_dict[key] = value
        model.load_state_dict(non_expert_dict, True)

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
        generated_token_ids = input_ids
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
            generated_token_ids = torch.cat((generated_token_ids, next_token_id), dim=1)

            # Update the attention_mask for new token
            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=attention_mask.device)], dim=-1)
            router_logits.append(outputs.router_logits)

        merged_router_logits = []
        num_layers = len(router_logits[0])
        for i in range(num_layers):
            layer_logits = [logit[i] for logit in router_logits]
            merged_logits = torch.cat(layer_logits, dim=0)
            merged_router_logits.append(merged_logits)
        return generated_token_ids, merged_router_logits


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
        offload_per_layer=4,
        buffer_size=4,
        state_path=state_path,
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     "mistralai/Mixtral-8x7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map=torch.device("cuda:0"))
    model.eval()

    #########################################################
    # 构建routing path dataset
    #########################################################
    import datasets
    from transformers import AutoModelForSeq2SeqLM, AutoConfig, AutoTokenizer

    # data_name = 'xsum'
    data_name = 'wmt16'
    dataset_name = f"marsggbo/{data_name}_switch32_token_real_and_predicted_patterns_t5-small_dff2048_dmodel32"
    ds = datasets.load_dataset(dataset_name)
    all_prompt_text = ds['train']['prompt_text']
    bs = 16
    batch_text = [all_prompt_text[i:i+bs] for i in range(0, len(all_prompt_text), bs)]

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    tokenizer.padding_side = 'left'
    
    dataset_for_predictor = {
        "prompt_text": [],
        "prompt_ids": [],
        "decode_ids": [],
        "prompt_pattern": [],
        "decode_pattern": []
    }
    def parse_router_logits(router_logits_tuples):
        encoder_router, decoder_router = router_logits_tuples
        num_layers = len(encoder_router)
        encoder_pattern = torch.stack([encoder_router[i][1] for i in range(num_layers) if i%2==1], dim=-2) # bs,seq_len, num_layer, num_experts
        decoder_pattern = torch.stack([decoder_router[i][1] for i in range(num_layers) if i%2==1], dim=-2) # bs,seq_len, num_layer, num_experts
        return encoder_pattern.cpu(), decoder_pattern.cpu()


    max_tokens = 32
    rank = 0
    device = torch.device(f"cuda:{rank}")
    for batch_idx, batch_data in enumerate(batch_text):
        if batch_idx == 2:
            break
        data = tokenizer(batch_data, return_tensors="pt", padding=True, return_attention_mask=True)
        input_ids = data.input_ids.to(rank)
        attention_mask = data.attention_mask.to(rank)
        decoder_input_ids = torch.tensor([[0]]*len(input_ids)).int().to(device)
        generated_ids, router_logits_tuples = custom_generate(
            input_ids, attention_mask, model, max_new_tokens=max_tokens
        )
        encoder_pattern, decoder_pattern = parse_router_logits(router_logits_tuples)
        attention_mask = attention_mask.cpu()
        for i in range(len(generated_ids)):
            dataset_for_predictor['prompt_text'].append(batch_data[i])
            unpadded_input_ids = input_ids[i][attention_mask[i].bool()]
            dataset_for_predictor['prompt_ids'].append(unpadded_input_ids.cpu())
            dataset_for_predictor['decode_ids'].append(generated_ids[i].cpu())
            pattern_shape = encoder_pattern[0].shape
            prompt_pattern = encoder_pattern[i][attention_mask[i].repeat(pattern_shape[0],1).bool()].view(pattern_shape[0],-1)
            dataset_for_predictor['prompt_pattern'].append(prompt_pattern)
            dataset_for_predictor['decode_pattern'].append(decoder_pattern[i])
            print(f"{i} Q: {tokenizer.decode(input_ids[i].cpu().numpy().tolist(), skip_special_tokens=True)}")
            print(f"{i} A: {tokenizer.decode(generated_ids[i].cpu().numpy().tolist(), skip_special_tokens=True)}")
    # dataset_for_predictor = datasets.Dataset.from_dict(dataset_for_predictor)
    # dataset_for_predictor.push_to_hub(f'marsggbo/{data_name}_mixtral8x7bInstructv0.1_token_patterns')
