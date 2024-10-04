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

class MixtralMoeWrapper(nn.Module):
    def __init__(self, config, layer_id, gate, expert_cache):
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = gate
        self.experts = expert_cache

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        active_experts = selected_experts.flatten().unique().tolist()

        # Loop over all available experts in the model and perform the computation on each expert
        for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(
                *((self.layer_id, expert_idx) for expert_idx in active_experts), unordered=True):
            idx, top_x = torch.where(expert_mask[expert_idx])
            assert top_x.shape[0] > 0

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
