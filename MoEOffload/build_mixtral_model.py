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

    num_layers = 1 # config.num_hidden_layers
    num_experts = 8 # config.num_local_experts
    num_expert_layers = 1 # config.num_hidden_layers
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
    
    config.num_hidden_layers = 1
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
    