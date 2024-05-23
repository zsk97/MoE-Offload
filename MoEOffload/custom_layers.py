from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.activations import ACT2FN
from typing import Dict, Any

import torch
from torch import nn
from torch.nn import functional as F
from fairscale.nn.model_parallel.layers import reduce_from_model_parallel_region


class SwitchMoeWrapper(nn.Module):
    def __init__(self, config, layer_id, gate, expert_cache):
        config.num_experts_per_tok = config.num_selected_experts
        config.intermediate_size = config.d_ff
        config.num_local_experts = config.num_experts
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id
        self.router = gate
        self.experts = expert_cache

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1) # shape is (batch, seq_len), dtype is int64
        active_experts = expert_index.flatten().unique().tolist()
        next_states = torch.zeros_like(hidden_states)
        for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(
            *((self.layer_id, expert_idx) for expert_idx in active_experts), unordered=True):
            token_indices = router_mask[:, :, expert_idx].bool()
            if torch.any(token_indices):
                expert_out = expert_layer(hidden_states[token_indices]).to(next_states.dtype)
                next_states[token_indices] = expert_out * router_probs[token_indices]
        hidden_states = reduce_from_model_parallel_region(next_states)
        return hidden_states, (router_logits, expert_index)

class SwitchMoeWrapperV1(nn.Module):
    def __init__(self, config, layer_id, gate, expert_cache):
        config.num_experts_per_tok = config.num_selected_experts
        config.intermediate_size = config.d_ff
        config.num_local_experts = config.num_experts
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id
        self.router = gate
        self.experts = expert_cache

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1) # shape is (batch, seq_len), dtype is int64
        active_experts = expert_index.flatten().unique().tolist()
        next_states = torch.zeros_like(hidden_states)

        # print(f"{len(active_experts)} should be loaded")
        # print("Hidden shape ", hidden_states.shape)

        self.experts.sync_layer(self.layer_id)
        for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(
            *((self.layer_id, expert_idx) for expert_idx in active_experts), unordered=True):
            token_indices = router_mask[:, :, expert_idx].bool()
            if torch.any(token_indices):
                expert_out = expert_layer(hidden_states[token_indices]).to(next_states.dtype)
                next_states[token_indices] = expert_out * router_probs[token_indices]
        hidden_states = reduce_from_model_parallel_region(next_states)
        return hidden_states, (router_logits, expert_index)