from MoEOffload.expert_cache import ExpertCache, ExpertCacheV1
from MoEOffload.custom_layers import SwitchMoeWrapper, SwitchMoeWrapperV1
from MoEOffload.expert_wrapper import SwitchExpertWrapper
from MoEOffload.utils import (nested_flatten, nested_pack, with_default_dtype,
                              forward_pre_hook, forward_post_hook)

from transformers.models.switch_transformers.configuration_switch_transformers import SwitchTransformersConfig
from MoEOffload.models.switch_transformer import (
    SwitchTransformersDenseActDense,
    SwitchTransformersForConditionalGeneration)

from dataclasses import dataclass
from transformers import AutoConfig
from safetensors.torch import load_file

from tqdm.auto import trange
from glob import glob
import json
import os
import typing as tp
import torch

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
        'index_file': None
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

@dataclass(frozen=True)
class OffloadConfig:
    main_size: int
    offload_size: int
    buffer_size: int
    offload_per_layer: int

def make_empty_expert(
        model_config: SwitchTransformersConfig
    ) -> SwitchTransformersDenseActDense:
        return SwitchTransformersDenseActDense(
            model_config,
        )

def make_and_load_expert_wrapper(
        config: SwitchTransformersConfig,
        states_dir: str,
        expert_prefix: str, # 'encoder' or 'decoder
        expert_uid: tuple[int, int],
        device: torch.device,
        base_layer_idx: int,
        is_baseline: bool = False,
        pretrained_weights_map: dict=pretrained_switch_weights_map
    ):
        assert expert_prefix in ['encoder', 'decoder']
        layer_idx, expert_idx = expert_uid
        if expert_prefix == 'decoder':
            layer_idx -= base_layer_idx
        sub_layer_id = 1 if expert_prefix=='encoder' else 2

        weight_info = pretrained_weights_map[config._name_or_path]
        weight_file_type = weight_info['file_type']
        if weight_file_type=='bin':
            weight_load_func = lambda filepath, device: torch.load(filepath, map_location=str(device))
        else:
            weight_load_func = lambda filepath, device: load_file(filepath, device=str(device))
        weight_index_file = weight_info['index_file']
        if weight_index_file is not None:
            index_path = glob(f"{states_dir}/{weight_index_file}")[0]
        else:
            index_path = None
        module_idx = f"{expert_prefix}.block.{layer_idx}.layer.{sub_layer_id}.mlp.experts.expert_{expert_idx}"
        if index_path is not None:
            # 多文件权重
            with open(index_path) as f:
                # example: encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight
                # example: encoder.block.1.layer.1.mlp.experts.expert_0.wo.weight
                # example: decoder.block.1.layer.2.mlp.experts.expert_7.wi.weight
                # example: decoder.block.1.layer.2.mlp.experts.expert_7.wo.weight
                weight_map = json.load(f)["weight_map"]
                state_fpaths = [weight_map[f"{module_idx}.w{i}.weight"] for i in ['i', 'o']]
                state_fpaths = list(set(state_fpaths))
            for state_fpath in state_fpaths:
                state_dict = weight_load_func(os.path.join(states_dir, state_fpath), device)
                expert = make_empty_expert(config).bfloat16()
                for idx in ['i', 'o']:
                    layer = getattr(expert, f"w{idx}")
                    w_to_load = state_dict[f'{module_idx}.w{idx}.weight']
                    layer.weight.data.copy_(w_to_load)
        else:
            # 单文件权重
            if weight_file_type == 'bin':
                state_fpaths = glob(f"{states_dir}/pytorch_model.bin")
            else:
                state_fpaths = glob(f"{states_dir}/*.safetensors")
            assert len(state_fpaths)==1
            global MODEL_STATE_DICT
            if MODEL_STATE_DICT is None:
                MODEL_STATE_DICT = weight_load_func(state_fpaths[0], device)
            expert = make_empty_expert(config).bfloat16()
            for idx in ['i', 'o']:
                layer = getattr(expert, f"w{idx}")
                w_to_load = MODEL_STATE_DICT[f'{module_idx}.w{idx}.weight']
                layer.weight.data.copy_(w_to_load)
        
        return SwitchExpertWrapper(expert, device)

def build_offload_switch(
    offload_per_layer: int=16,
    buffer_size:int = 6,
    state_path: str='/home/nus-hx/.cache/huggingface/hub/models--google--switch-base-16/snapshots/0ef7d88ed50ec5f2cfdc019e81cef04d19700f8f',
    model_name="google/switch-base-32",
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

    num_layers = config.num_hidden_layers + config.num_decoder_layers
    num_experts = config.num_experts
    num_expert_layers = config.num_hidden_layers//config.encoder_sparse_step+config.num_decoder_layers//config.decoder_sparse_step
    offload_config = OffloadConfig(
        main_size=num_expert_layers * (num_experts - offload_per_layer),
        offload_size=num_expert_layers * offload_per_layer,
        buffer_size=buffer_size,
        offload_per_layer=offload_per_layer,
    )

    def _make_module():
        config = AutoConfig.from_pretrained(model_name)
        expert = make_empty_expert(config).bfloat16()
        return SwitchExpertWrapper(expert, device=device)
    
    MoEWrapper = None
    expertCache = None

    if is_baseline:
        expertCache = ExpertCache
        MoEWrapper = SwitchMoeWrapper
    else:
        expertCache = ExpertCacheV1
        MoEWrapper = SwitchMoeWrapperV1
    
    with device, with_default_dtype(torch.bfloat16):
        model = SwitchTransformersForConditionalGeneration(config)
    
    model_config = config
    expert_cache = expertCache(
        make_module=_make_module,
        main_size=offload_config.main_size,
        offload_size=offload_config.offload_size,
        buffer_size=offload_config.buffer_size,
        num_layer=num_layers
    )

    for block_type in ['encoder', 'decoder']:
        if block_type == 'encoder':
            num_block_layers = model_config.num_layers
            sparse_step = model_config.encoder_sparse_step
            block_inner_layer_id = 1
            base_layer_idx = 0
        else:
            num_block_layers = model_config.num_decoder_layers
            sparse_step = model_config.decoder_sparse_step
            block_inner_layer_id = 2
            base_layer_idx = model_config.num_layers
        for block_idx in list(range(num_block_layers))[1:][::sparse_step]:
            curr_layer = getattr(model, block_type).block[block_idx].layer[block_inner_layer_id]
            curr_layer.mlp = MoEWrapper(
                config=model_config,
                layer_id=block_idx+base_layer_idx,
                gate=curr_layer.mlp.router,
                expert_cache=expert_cache,
            )

            for expert_idx in range(model_config.num_experts):
                do_offload = expert_idx < offload_config.offload_per_layer

                expert_wrapper = make_and_load_expert_wrapper(
                    config=model_config,
                    states_dir=state_path,
                    expert_prefix=block_type,
                    expert_uid=(base_layer_idx+block_idx, expert_idx),
                    base_layer_idx=base_layer_idx,
                    device=device,
                )

                expert_cache.add_expert(
                    uid=(base_layer_idx+block_idx, expert_idx),
                    module=expert_wrapper,
                    eviction_group=base_layer_idx+block_idx,
                    offload=do_offload,
                )

                del expert_wrapper
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    if MODEL_STATE_DICT is not None:
        non_expert_dict = {}
        for key, val in MODEL_STATE_DICT.items():
            if 'expert' not in key:
                non_expert_dict[key] = val
        model.load_state_dict(non_expert_dict, True)

    if is_profile:
        for module in model.modules():
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_post_hook)
    return model, expert_cache
