import numpy as np
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from transformers import AutoConfig
import torch.distributed as dist

from .common_utils import to
from .llama_utils import QuantizedLlamaMLP, QuantizedLlamaAttention
from .qwen3_utils import QuantizedQwen3MLP, QuantizedQwen3Attention
from .qwen3_moe_utils import QuantizedQwen3MoeSparseMoeBlock

def _get_ep_size():
    """Get expert parallel size from distributed context."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def _get_ep_rank():
    """Get expert parallel rank from distributed context."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

### Calibration utils and modules

LINEAR_LAYERS = (nn.Linear, _ConvNd)


class ForwardInterrupt(Exception):
    pass


class InputCollector(nn.Module):

    def __init__(self, module: nn.Module, cpu_offload: bool = False):
        super().__init__()
        self.module = module
        self.cpu_offload = cpu_offload
        self.input_args = []
        self.input_kwargs = []

    def forward(self, *input_args, **input_kwargs):
        """
        Assumes that the wrapped module has a single
        input that can reside in inputs or input_kwargs.
        """
        if self.cpu_offload:
            input_args = to(input_args, device="cpu")
            input_kwargs = to(input_kwargs, device="cpu")
        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)
        raise ForwardInterrupt
    
def get_number_of_rows_and_cols(layer):
    return layer.weight.shape[0], np.prod(layer.weight.shape[1:])

def get_mlp_layer(config: AutoConfig, ep_size: int = None, ep_rank: int = None, ep_group = None):
    # Auto-detect ep_size and ep_rank if not provided
    if ep_size is None:
        ep_size = _get_ep_size()
    if ep_rank is None:
        ep_rank = _get_ep_rank()
    
    if config.model_type == "llama":
        return QuantizedLlamaMLP
    elif config.model_type == "qwen3":
        return QuantizedQwen3MLP
    elif config.model_type == "qwen3_moe":
        # Pass expert parallel parameters to the MoE block
        def create_mlp(*args, **kwargs):
            return QuantizedQwen3MoeSparseMoeBlock(
                *args, 
                ep_size=ep_size, 
                ep_rank=ep_rank, 
                ep_group=ep_group,
                **kwargs
            )
        return create_mlp
    else:
        raise ValueError(f"Model type {config.model_type} not supported")

def get_attention_layer(config: AutoConfig):
    if config.model_type == "llama":
        return QuantizedLlamaAttention
    elif config.model_type == "qwen3" or config.model_type == "qwen3_moe":
        return QuantizedQwen3Attention
    else:
        raise ValueError(f"Model type {config.model_type} not supported")
