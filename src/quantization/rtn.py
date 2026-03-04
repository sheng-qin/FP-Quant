import gc
import re
import argparse
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from .qlinear import QLinear
from .quant_ops import pack_fp4_to_uint8, cast_scales_to_eXmY, ScalePrecision

from ..utils.common_utils import clear_device_cache, to, maybe_first_element, get_global_layer_name
from ..utils.model_utils import InputCollector, ForwardInterrupt, get_attention_layer, get_mlp_layer
from ..transforms.transforms import build_transform, get_transform_matrix


def rtn_quantization(
    model: AutoModelForCausalLM, 
    calibration_data: List[torch.Tensor],
    args: argparse.Namespace, 
    device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    print("RTN quantization...")
    orig_dtype = model.config.torch_dtype if args.dtype == "auto" else args.dtype
    act_offload_device = "cpu" if args.cpu_offload_activations else device
    need_calibration = args.a_bits < 16 and args.scale_precision == ScalePrecision.E4M3
    # State dict with quantized weights, scales and hadamards
    quantized_state_dict = {}
    non_quantized_state_dict = {}
    
    # Check for expert parallel
    ep_size = 1
    ep_rank = 0
    ep_group = None
    is_moe = hasattr(model.config, 'num_local_experts') and model.config.num_local_experts > 1
    if is_moe and dist.is_available() and dist.is_initialized():
        ep_size = dist.get_world_size()
        ep_rank = dist.get_rank()
        # Create expert parallel group for MoE models
        ep_group = dist.group.WORLD
        print(f"Expert parallel enabled: ep_size={ep_size}, ep_rank={ep_rank}, ep_group={ep_group}")
    
    # Get transformer blocks
    blocks = model.model.layers
    # Define common transform kwargs
    transform_kwargs = dict(device=device, group_size=args.hadamard_group_size)
    # Init quantizer kwargs
    weight_quantizer_kwargs = None
    if args.w_bits < 16:
        weight_quantizer_kwargs = dict(
            bits=args.w_bits, 
            symmetric= not args.w_asymmetric, 
            format=args.format,
            granularity=args.w_granularity,
            observer=args.w_observer, 
            group_size=args.w_group_size,
            scale_precision=args.scale_precision,
        )
    act_quantizer_kwargs = None
    if args.a_bits < 16:
        act_quantizer_kwargs = dict(
            bits=args.a_bits, 
            symmetric=True, 
            format=args.format,
            granularity=args.a_granularity,
            observer=args.a_observer, 
            group_size=args.a_group_size,
            scale_precision=args.scale_precision,
        )

    if need_calibration:
        blocks = model.model.layers
        blocks[0] = InputCollector(blocks[0], cpu_offload=args.cpu_offload_activations)
        if args.cpu_offload_modules:
            model.get_input_embeddings().to(device)
            blocks[0] = blocks[0].to(device)

        for sample in calibration_data:
            try:
                with torch.no_grad():
                    model(sample.to(device=device))
            except ForwardInterrupt:
                pass
            
        input_args = blocks[0].input_args
        input_kwargs = blocks[0].input_kwargs
        blocks[0] = blocks[0].module

        if args.cpu_offload_modules:
            model.get_input_embeddings().cpu()

    # Iterate over transformer blocks
    for block_idx, block in enumerate(blocks):
        print(f"Processing block {block_idx}...")
        if args.cpu_offload_modules:
            block.to(device)
        # 1. Init transforms
        qkv_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        o_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        gate_up_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        down_in_transform = build_transform(args.transform_class, size=model.config.intermediate_size, **transform_kwargs)     

        # 2. Replace blocks with quantized versions
        quantized_attn = get_attention_layer(model.config)(
            model.config,
            layer_idx=block_idx,
            weight_quantizer_kwargs=weight_quantizer_kwargs,
            act_quantizer_kwargs=act_quantizer_kwargs,
            qkv_in_transform=qkv_in_transform,
            o_in_transform=o_in_transform
        )
        
        # For MoE models, use expert parallel if enabled
        if is_moe:
            # Get the MLP class with ep parameters
            from functools import partial
            mlp_class = get_mlp_layer(model.config, ep_size=ep_size, ep_rank=ep_rank, ep_group=ep_group)
            quantized_mlp = mlp_class(
                model.config,
                weight_quantizer_kwargs=weight_quantizer_kwargs,
                act_quantizer_kwargs=act_quantizer_kwargs,
                gate_up_in_transform=gate_up_in_transform,
                down_in_transform=down_in_transform
            )
        else:
            quantized_mlp = get_mlp_layer(model.config)(
                model.config,
                weight_quantizer_kwargs=weight_quantizer_kwargs,
                act_quantizer_kwargs=act_quantizer_kwargs,
                gate_up_in_transform=gate_up_in_transform,
                down_in_transform=down_in_transform
            )

        quantized_attn.load_state_dict(block.self_attn.state_dict(), strict=False)
        
        # For MoE with expert parallel, only load the local experts
        # Check if model has num_local_experts (set by transformers when EP is enabled)
        use_ep_filter = False
        if hasattr(model.config, 'num_local_experts') and model.config.num_local_experts is not None:
            total_experts = getattr(model.config, 'num_experts', None)
            num_local_experts = model.config.num_local_experts
            
            # Only filter if we have both values and local < total
            if total_experts is not None and num_local_experts < total_experts:
                use_ep_filter = True
                print(f"[Rank {ep_rank}] EP filter: total_experts={total_experts}, num_local_experts={num_local_experts}, ep_size={ep_size}")
                
                # Calculate which experts belong to this rank
                expert_start_idx = ep_rank * num_local_experts
                
                # Get full state dict
                state_dict = block.mlp.state_dict()
                
                # Filter to only include local experts
                local_state_dict = {}
                for key, value in state_dict.items():
                    if 'experts.' in key:
                        parts = key.split('.')
                        exp_idx = int(parts[1])
                        # Only include experts that belong to this rank
                        if exp_idx >= expert_start_idx and exp_idx < expert_start_idx + num_local_experts:
                            local_key = key.replace(f'experts.{exp_idx}', f'experts.{exp_idx - expert_start_idx}')
                            local_state_dict[local_key] = value
                    else:
                        local_state_dict[key] = value
                
                quantized_mlp.load_state_dict(local_state_dict, strict=False)
        
        if not use_ep_filter:
            quantized_mlp.load_state_dict(block.mlp.state_dict(), strict=False)

        block.self_attn = quantized_attn
        block.mlp = quantized_mlp

        # Move to original device and dtype
        block = block.to(device=device, dtype=orig_dtype)  

        # Init anf freeze global scales (for NVFP)
        if args.w_bits < 16 and args.scale_precision == ScalePrecision.E4M3:
            for layer_name, layer in block.named_modules():
                if isinstance(layer, QLinear):
                    with torch.no_grad():
                        ## removed
                        layer.weight_quantizer.get_global_scale(layer.weight)
                        # Stop tracking global scale
                        layer.weight_quantizer._track_global_scale = False

            if args.fuse_global_scale:
                qkv_global_scale = min(
                    quantized_attn.q_proj.weight_quantizer.global_scale,
                    quantized_attn.k_proj.weight_quantizer.global_scale,
                    quantized_attn.v_proj.weight_quantizer.global_scale,
                )
                quantized_attn.q_proj.weight_quantizer.global_scale = qkv_global_scale
                quantized_attn.k_proj.weight_quantizer.global_scale = qkv_global_scale
                quantized_attn.v_proj.weight_quantizer.global_scale = qkv_global_scale
                if not is_moe:
                    # gate_up fusion
                    gate_up_global_scale = min(
                        quantized_mlp.gate_proj.weight_quantizer.global_scale,
                        quantized_mlp.up_proj.weight_quantizer.global_scale
                    )
                    quantized_mlp.gate_proj.weight_quantizer.global_scale = gate_up_global_scale
                    quantized_mlp.up_proj.weight_quantizer.global_scale = gate_up_global_scale

        # Calibrate activations (if needed)
        if need_calibration:
            quantized_mlp.amax_calib = True
            device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                    block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            quantized_mlp.amax_calib = False

        
        if args.export_quantized_model:
            for layer_name, layer in block.named_modules():
                if isinstance(layer, QLinear):
                    with torch.no_grad():
                        ## removed
                        scales, zeros = layer.weight_quantizer.get_quantization_params(layer.weight)
                        qweight = layer.weight_quantizer.quantize(layer.weight, scales, zeros)
                        dqweight = layer.weight_quantizer(layer.weight, scales, zeros)

                    weight_global_scale = layer.weight_quantizer.global_scale.to(scales.device)
                    act_global_scale = layer.act_quantizer.global_scale if layer.act_quantizer else torch.ones_like(weight_global_scale)

                    # Stop tracking global scale
                    layer.weight_quantizer._track_global_scale = False
                    if layer.act_quantizer:
                        layer.act_quantizer._track_global_scale = False

                    transform_matrix = get_transform_matrix(args.transform_class, args.hadamard_group_size, device, orig_dtype).cpu()

                    # Convert layer_name to use global expert index if EP is enabled
                    global_layer_name = get_global_layer_name(layer_name, ep_rank, model.config.num_local_experts) if is_moe and ep_size > 1 else layer_name
                    if args.export_quantized_model == "realquant":
                        quantized_state_dict[f"model.layers.{block_idx}.{global_layer_name}"] = {
                            "weight": pack_fp4_to_uint8(qweight).cpu(),
                            "weight_scale": cast_scales_to_eXmY(scales * weight_global_scale, args.scale_precision).cpu(),
                            "weight_scale_2": 1 / weight_global_scale.clone(),
                            "input_scale": 1 / act_global_scale.clone()
                        }
                    # pseudoquant
                    else:
                        quantized_state_dict[f"model.layers.{block_idx}.{global_layer_name}"] = {
                            "weight": dqweight.cpu(),
                        }  
        

        # 3. Fix model parametrization
        ## removed
        # 4. Fix transforms and remove parametrizations
        ## removed

        if need_calibration:
            device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                    out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                out = maybe_first_element(out).to(act_offload_device)
                # change only first input argument
                if len(inp_args) > 0:
                    inp_args[0].data = out
                elif "hidden_states" in inp_kwargs:
                    inp_kwargs["hidden_states"] = out
                else:
                    raise ValueError("Unsupported block input format.")

        if args.cpu_offload_modules:
            block.cpu()

        clear_device_cache(garbage_collection=True)

    clear_device_cache(garbage_collection=True)

    return quantized_state_dict, non_quantized_state_dict
