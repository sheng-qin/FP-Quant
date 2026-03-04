from typing import Tuple, Optional, Callable, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb, eager_attention_forward
from transformers.activations import ACT2FN

from ..quantization.qlinear import QLinear
from ..quantization.quantizer import Quantizer
from ..transforms.transforms import BaseTransform, IdentityTransform


class QuantizedQwen3MoeMLP(nn.Module):
    def __init__(
            self, 
            config, 
            intermediate_size=None,
            weight_quantizer_kwargs: Dict[str, Any] | None = None,
            act_quantizer_kwargs: Dict[str, Any] | None = None,
            gate_up_in_transform: BaseTransform = IdentityTransform(),
            down_in_transform: BaseTransform = IdentityTransform()
        ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.moe_intermediate_size
        self.gate_up_in_transform = gate_up_in_transform
        self.down_in_transform = down_in_transform
        self.gate_up_proj = QLinear(
                                    self.hidden_size,
                                    self.intermediate_size * 2,  # gate and up combined
                                    bias=False,
                                    weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
                                    act_quantizer=Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
                                )
        self.down_proj = QLinear(
                                    self.intermediate_size,
                                    self.hidden_size,
                                    bias=False,
                                    weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
                                    act_quantizer=Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
                                )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate_up_output = self.gate_up_proj(x, self.gate_up_in_transform)
        gate_output, up_output = gate_up_output.chunk(2, dim=-1)
        down_proj = self.down_proj(self.act_fn(gate_output) * up_output, self.down_in_transform)
        return down_proj

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """
        Load state dict by fusing gate_proj and up_proj weights into gate_up_proj.
        
        Original state_dict has:
            - gate_proj.weight: (intermediate_size, hidden_size)
            - up_proj.weight: (intermediate_size, hidden_size)
        
        Fused layer expects:
            - gate_up_proj.weight: (2 * intermediate_size, hidden_size)
        """
        new_state_dict = {}
        
        for key, value in state_dict.items():
            if '.gate_proj.' in key:
                # Replace gate_proj with gate_up_proj (first half)
                new_key = key.replace('.gate_proj.', '.gate_up_proj.')
                # Store gate_proj weight for later concatenation
                gate_key = key.replace('gate_proj', 'gate_proj')
                if 'weight' in key:
                    # Will concatenate with up_proj
                    gate_weight = value
                    up_key = key.replace('gate_proj', 'up_proj')
                    if up_key in state_dict:
                        up_weight = state_dict[up_key]
                        # Concatenate: gate_proj.weight (intermediate_size, hidden_size) 
                        #            + up_proj.weight (intermediate_size, hidden_size)
                        #            -> gate_up_proj.weight (2*intermediate_size, hidden_size)
                        new_state_dict[new_key] = torch.cat([gate_weight, up_weight], dim=0)
                        continue
            elif '.up_proj.' in key:
                # up_proj will be handled when processing gate_proj
                continue
            else:
                new_state_dict[key] = value
        
        super().load_state_dict(new_state_dict, strict=strict)


class QuantizedQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
                self, 
                config,
                weight_quantizer_kwargs: Dict[str, Any] | None = None,
                act_quantizer_kwargs: Dict[str, Any] | None = None,
                gate_up_in_transform: BaseTransform = IdentityTransform(),
                down_in_transform: BaseTransform = IdentityTransform(),
                ep_size: int = 1,
                ep_rank: int = 0,
                ep_group = None,
                use_dp_ep = True,
            ):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        
        # Expert parallel configuration
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group
        self.use_dp_ep = use_dp_ep
        
        # Calculate number of local experts for this rank
        # Each rank holds num_experts // ep_size experts
        self.num_local_experts = self.num_experts // self.ep_size if self.ep_size > 1 else self.num_experts
        self.experts_per_rank = self.num_local_experts

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [QuantizedQwen3MoeMLP(
                    config, 
                    config.moe_intermediate_size, 
                    weight_quantizer_kwargs, 
                    act_quantizer_kwargs,
                    gate_up_in_transform,
                    down_in_transform
                ) for _ in range(self.num_local_experts)]
        )
        self.amax_calib = False

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """
        Load state dict by fusing gate_proj and up_proj weights into gate_up_proj for each expert.
        
        Original state_dict has:
            - experts.{i}.gate_proj.weight: (moe_intermediate_size, hidden_size)
            - experts.{i}.up_proj.weight: (moe_intermediate_size, hidden_size)
        
        Fused layer expects:
            - experts.{i}.gate_up_proj.weight: (2 * moe_intermediate_size, hidden_size)
        """
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Handle gate_proj and up_proj fusion
            if '.gate_proj.' in key:
                # Replace gate_proj with gate_up_proj
                new_key = key.replace('.gate_proj.', '.gate_up_proj.')
                if 'weight' in key:
                    # Get corresponding up_proj weight
                    up_key = key.replace('.gate_proj.', '.up_proj.')
                    if up_key in state_dict:
                        gate_weight = value
                        up_weight = state_dict[up_key]
                        # Concatenate along output dimension (dim=0)
                        # gate_proj: (moe_intermediate_size, hidden_size)
                        # up_proj: (moe_intermediate_size, hidden_size)
                        # gate_up_proj: (2 * moe_intermediate_size, hidden_size)
                        new_state_dict[new_key] = torch.cat([gate_weight, up_weight], dim=0)
                        continue
            elif '.up_proj.' in key:
                # Skip - will be merged with gate_proj
                continue
            else:
                new_state_dict[key] = value
        
        super().load_state_dict(new_state_dict, strict=strict)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Expert parallel forward pass.
        Each rank only processes experts it owns (expert_idx in [ep_rank * num_local_experts, (ep_rank+1) * num_local_experts))
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.ep_size > 1 and self.use_dp_ep:
            gathered_states = [torch.zeros_like(hidden_states) for _ in range(self.ep_size)]
            dist.all_gather(gathered_states, hidden_states, group=self.ep_group)
            hidden_states = torch.cat(gathered_states, dim=0)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        # HACK route all tokens to all expert to search activation global scale (refer to modelopt) 
        top_k = self.num_experts if self.amax_calib else self.top_k
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * self.ep_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # With expert parallel, each rank only handles its local experts
        # expert_idx in this rank: [ep_rank * num_local_experts, (ep_rank+1) * num_local_experts)
        if self.ep_size > 1:
            # Filter to only experts owned by this rank
            start_expert_idx = self.ep_rank * self.num_local_experts
            end_expert_idx = start_expert_idx + self.num_local_experts
            expert_mask = expert_mask[start_expert_idx:end_expert_idx]
        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            # print(f"eid {expert_idx} {current_state.shape} {current_state}")
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        if self.ep_size > 1:
            # All-reduce to aggregate results across all expert parallel ranks
            # Each rank has computed part of the output (for its local experts), now we combine them
            assert self.ep_group is not None
            if self.use_dp_ep:
                # Reshape to (ep_size, batch*seq, hidden_dim) then split into list
                final_hidden_states = final_hidden_states.view(self.ep_size, -1, hidden_dim)
                # Split into list of tensors, one per rank
                output_list = [final_hidden_states[i] for i in range(self.ep_size)]
                output = torch.empty((batch_size * sequence_length, hidden_dim), 
                             device=hidden_states.device, dtype=hidden_states.dtype)
                dist.reduce_scatter(output, output_list, op=dist.ReduceOp.SUM, group=self.ep_group)
                final_hidden_states = output
            else:
                dist.all_reduce(final_hidden_states, op=dist.ReduceOp.SUM, group=self.ep_group)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # print(final_hidden_states)
        return final_hidden_states, router_logits