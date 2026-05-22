import argparse
from contextlib import contextmanager
import gc
from typing import Dict, Iterable, List

import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .qlinear import QLinear
from ..transforms.transforms import BaseTransform, IdentityTransform
from ..utils.common_utils import clear_device_cache, get_global_layer_name, maybe_first_element, to
from ..utils.model_utils import (
    ForwardInterrupt,
    InputCollector,
    get_attention_layer,
    get_mlp_layer,
)


class SmoothQuantScaleTransform(BaseTransform):
    """Per-channel SmoothQuant transform: activations are divided, weights multiplied."""

    def __init__(self, scales: torch.Tensor):
        super().__init__()
        self.register_buffer("scales", scales.detach().clone())

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        scales = self.scales.to(device=x.device, dtype=x.dtype)
        view_shape = [1] * x.ndim
        view_shape[dim if dim >= 0 else x.ndim + dim] = scales.numel()
        scales = scales.view(view_shape)
        return x * scales if inv_t else x / scales

    def remove_parametrizations(self) -> None:
        pass


def _get_ep_context(model):
    ep_size = 1
    ep_rank = 0
    ep_group = None
    is_moe = hasattr(model.config, "num_local_experts") and model.config.num_local_experts > 1
    if is_moe and dist.is_available() and dist.is_initialized():
        ep_size = dist.get_world_size()
        ep_rank = dist.get_rank()
        ep_group = dist.group.WORLD
        print(f"Expert parallel enabled: ep_size={ep_size}, ep_rank={ep_rank}, ep_group={ep_group}")
    return is_moe, ep_size, ep_rank, ep_group


def _get_quantizer_kwargs(args: argparse.Namespace):
    weight_quantizer_kwargs = None
    if args.w_bits < 16:
        weight_quantizer_kwargs = dict(
            bits=args.w_bits,
            symmetric=not args.w_asymmetric,
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
    return weight_quantizer_kwargs, act_quantizer_kwargs


def _build_quantized_block(model, block, block_idx, weight_quantizer_kwargs, act_quantizer_kwargs, is_moe, ep_size, ep_rank, ep_group):
    identity = IdentityTransform()
    quantized_attn = get_attention_layer(model.config)(
        model.config,
        layer_idx=block_idx,
        weight_quantizer_kwargs=weight_quantizer_kwargs,
        act_quantizer_kwargs=act_quantizer_kwargs,
        qkv_in_transform=identity,
        o_in_transform=identity,
    )

    if is_moe:
        mlp_class = get_mlp_layer(model.config, ep_size=ep_size, ep_rank=ep_rank, ep_group=ep_group)
        quantized_mlp = mlp_class(
            model.config,
            weight_quantizer_kwargs=weight_quantizer_kwargs,
            act_quantizer_kwargs=act_quantizer_kwargs,
            gate_up_in_transform=identity,
            down_in_transform=identity,
        )
    else:
        quantized_mlp = get_mlp_layer(model.config)(
            model.config,
            weight_quantizer_kwargs=weight_quantizer_kwargs,
            act_quantizer_kwargs=act_quantizer_kwargs,
            gate_up_in_transform=identity,
            down_in_transform=identity,
        )

    quantized_attn.load_state_dict(block.self_attn.state_dict(), strict=False)
    quantized_mlp.load_state_dict(block.mlp.state_dict(), strict=False)
    block.self_attn = quantized_attn
    block.mlp = quantized_mlp
    return block


@torch.no_grad()
def _collect_activation_max(
    block: nn.Module,
    input_args: Iterable[tuple],
    input_kwargs: Iterable[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    activation_maxes: Dict[str, torch.Tensor] = {}
    hooks = []

    def update_activation_max(name):
        def _hook(_, inp, _out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).float()
            if name in activation_maxes:
                activation_maxes[name] = torch.maximum(activation_maxes[name].to(x.device), x)
            else:
                activation_maxes[name] = x
        return _hook

    for layer_name, layer in block.named_modules():
        if isinstance(layer, QLinear):
            hooks.append(layer.register_forward_hook(update_activation_max(layer_name)))

    device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    with _temporarily_disable_qlinear_quantizers(block):
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                block(*to(inp_args, device=device), **to(inp_kwargs, device=device))

    for hook in hooks:
        hook.remove()

    return activation_maxes


def _iter_qlinear_layers(block: nn.Module, layer_names: Iterable[str]):
    modules = dict(block.named_modules())
    for layer_name in layer_names:
        layer = modules.get(layer_name)
        if isinstance(layer, QLinear):
            yield layer_name, layer


def _group_smooth_scale(
    block: nn.Module,
    activation_maxes: Dict[str, torch.Tensor],
    layer_names: Iterable[str],
    alpha: float,
    min_scale: float,
) -> torch.Tensor | None:
    act_max = None
    weight_max = None
    for layer_name, layer in _iter_qlinear_layers(block, layer_names):
        if layer_name not in activation_maxes:
            continue
        current_act_max = activation_maxes[layer_name].to(layer.weight.device)
        current_weight_max = layer.weight.detach().abs().amax(dim=0).float()
        act_max = current_act_max if act_max is None else torch.maximum(act_max, current_act_max)
        weight_max = current_weight_max if weight_max is None else torch.maximum(weight_max, current_weight_max)

    if act_max is None or weight_max is None:
        return None

    act_max = act_max.clamp(min=min_scale)
    weight_max = weight_max.clamp(min=min_scale)
    return act_max.pow(alpha).div(weight_max.pow(1 - alpha)).clamp(min=min_scale)


def _set_identity_smoothquant_transforms(block: nn.Module):
    identity = IdentityTransform()
    block.self_attn.qkv_in_transform = identity
    block.self_attn.o_in_transform = identity
    block.mlp.gate_up_in_transform = identity
    block.mlp.down_in_transform = identity
    if hasattr(block.mlp, "experts"):
        for expert in block.mlp.experts:
            expert.gate_up_in_transform = block.mlp.gate_up_in_transform
            expert.down_in_transform = block.mlp.down_in_transform


def _get_moe_expert_idx(layer_name: str) -> int | None:
    parts = layer_name.split(".")
    if len(parts) < 4 or parts[0] != "mlp" or parts[1] != "experts":
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def _set_smoothquant_transforms(
    block: nn.Module,
    activation_maxes: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    alpha: float,
):
    min_scale = args.smoothquant_scale_min

    qkv_scale = _group_smooth_scale(
        block,
        activation_maxes,
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        alpha,
        min_scale,
    )
    if qkv_scale is not None:
        block.self_attn.qkv_in_transform = SmoothQuantScaleTransform(qkv_scale)

    o_scale = _group_smooth_scale(block, activation_maxes, ("self_attn.o_proj",), alpha, min_scale)
    if o_scale is not None:
        block.self_attn.o_in_transform = SmoothQuantScaleTransform(o_scale)

    if hasattr(block.mlp, "experts"):
        for expert_idx, expert in enumerate(block.mlp.experts):
            gate_up_scale = _group_smooth_scale(
                block,
                activation_maxes,
                (f"mlp.experts.{expert_idx}.gate_up_proj",),
                alpha,
                min_scale,
            )
            if gate_up_scale is not None:
                expert.gate_up_in_transform = SmoothQuantScaleTransform(gate_up_scale)

            down_scale = _group_smooth_scale(
                block,
                activation_maxes,
                (f"mlp.experts.{expert_idx}.down_proj",),
                alpha,
                min_scale,
            )
            if down_scale is not None:
                expert.down_in_transform = SmoothQuantScaleTransform(down_scale)
    else:
        gate_up_layer_names = ("mlp.gate_proj", "mlp.up_proj")
        gate_up_scale = _group_smooth_scale(block, activation_maxes, gate_up_layer_names, alpha, min_scale)
        if gate_up_scale is not None:
            block.mlp.gate_up_in_transform = SmoothQuantScaleTransform(gate_up_scale)
            if hasattr(block.mlp, "experts"):
                for expert in block.mlp.experts:
                    expert.gate_up_in_transform = block.mlp.gate_up_in_transform

        down_layer_names = ["mlp.down_proj"]
        down_scale = _group_smooth_scale(block, activation_maxes, down_layer_names, alpha, min_scale)
        if down_scale is not None:
            block.mlp.down_in_transform = SmoothQuantScaleTransform(down_scale)
            if hasattr(block.mlp, "experts"):
                for expert in block.mlp.experts:
                    expert.down_in_transform = block.mlp.down_in_transform


def _get_expert_transform(block: nn.Module, layer_name: str, transform_name: str) -> BaseTransform | None:
    expert_idx = _get_moe_expert_idx(layer_name)
    if expert_idx is None or not hasattr(block.mlp, "experts"):
        return None
    if expert_idx >= len(block.mlp.experts):
        return None
    return getattr(block.mlp.experts[expert_idx], transform_name, None)


def _get_smoothquant_scale(transform: BaseTransform) -> torch.Tensor | None:
    scales = getattr(transform, "scales", None)
    if scales is None:
        return None
    return scales.detach().float().cpu()


def _add_scale_for_layers(
    scale_state_dict: dict[str, torch.Tensor],
    block_idx: int,
    layer_names: Iterable[str],
    scale: torch.Tensor | None,
    model_config,
    is_moe: bool,
    ep_rank: int,
    ep_size: int,
) -> None:
    if scale is None:
        return
    for layer_name in layer_names:
        global_layer_name = (
            get_global_layer_name(layer_name, ep_rank, model_config.num_local_experts)
            if is_moe and ep_size > 1
            else layer_name
        )
        scale_state_dict[f"model.layers.{block_idx}.{global_layer_name}.smoothquant_scale"] = scale


def collect_smoothquant_scale_state(
    block: nn.Module,
    block_idx: int,
    model_config,
    is_moe: bool,
    ep_rank: int,
    ep_size: int,
) -> dict[str, torch.Tensor]:
    scale_state_dict: dict[str, torch.Tensor] = {}

    _add_scale_for_layers(
        scale_state_dict,
        block_idx,
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        _get_smoothquant_scale(block.self_attn.qkv_in_transform),
        model_config,
        is_moe,
        ep_rank,
        ep_size,
    )
    _add_scale_for_layers(
        scale_state_dict,
        block_idx,
        ("self_attn.o_proj",),
        _get_smoothquant_scale(block.self_attn.o_in_transform),
        model_config,
        is_moe,
        ep_rank,
        ep_size,
    )

    if hasattr(block.mlp, "experts"):
        for expert_idx in range(len(block.mlp.experts)):
            _add_scale_for_layers(
                scale_state_dict,
                block_idx,
                (
                    f"mlp.experts.{expert_idx}.gate_proj",
                    f"mlp.experts.{expert_idx}.up_proj",
                ),
                _get_smoothquant_scale(block.mlp.experts[expert_idx].gate_up_in_transform),
                model_config,
                is_moe,
                ep_rank,
                ep_size,
            )
            _add_scale_for_layers(
                scale_state_dict,
                block_idx,
                (f"mlp.experts.{expert_idx}.down_proj",),
                _get_smoothquant_scale(block.mlp.experts[expert_idx].down_in_transform),
                model_config,
                is_moe,
                ep_rank,
                ep_size,
            )
    else:
        gate_up_layers = ("mlp.gate_proj", "mlp.up_proj")
        down_layers = ("mlp.down_proj",)
        _add_scale_for_layers(
            scale_state_dict,
            block_idx,
            gate_up_layers,
            _get_smoothquant_scale(block.mlp.gate_up_in_transform),
            model_config,
            is_moe,
            ep_rank,
            ep_size,
        )
        _add_scale_for_layers(
            scale_state_dict,
            block_idx,
            down_layers,
            _get_smoothquant_scale(block.mlp.down_in_transform),
            model_config,
            is_moe,
            ep_rank,
            ep_size,
        )

    return scale_state_dict


def _get_layer_smoothquant_scale(block: nn.Module, layer_name: str) -> torch.Tensor | None:
    if layer_name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        return _get_smoothquant_scale(block.self_attn.qkv_in_transform)
    if layer_name == "self_attn.o_proj":
        return _get_smoothquant_scale(block.self_attn.o_in_transform)
    if hasattr(block.mlp, "experts"):
        if layer_name.endswith(".gate_up_proj"):
            return _get_smoothquant_scale(_get_expert_transform(block, layer_name, "gate_up_in_transform"))
        if layer_name.endswith(".down_proj"):
            return _get_smoothquant_scale(_get_expert_transform(block, layer_name, "down_in_transform"))
    if layer_name in ("mlp.gate_proj", "mlp.up_proj") or layer_name.endswith(".gate_up_proj"):
        return _get_smoothquant_scale(block.mlp.gate_up_in_transform)
    if layer_name == "mlp.down_proj" or layer_name.endswith(".down_proj"):
        return _get_smoothquant_scale(block.mlp.down_in_transform)
    return None


def _fold_smoothquant_scale_into_weight(block: nn.Module, layer_name: str, weight: torch.Tensor) -> torch.Tensor:
    scale = _get_layer_smoothquant_scale(block, layer_name)
    if scale is None:
        return weight
    return weight / scale.to(device=weight.device, dtype=weight.dtype).view(1, -1)


def fix_smoothquant_block(block: nn.Module):
    if hasattr(block.self_attn, "fix_parametrization"):
        block.self_attn.fix_parametrization()
    if hasattr(block.mlp, "experts"):
        _fix_qwen3_moe_parametrization(block)
    elif hasattr(block.mlp, "fix_parametrization"):
        block.mlp.fix_parametrization()


def _fix_qwen3_moe_parametrization(block: nn.Module):
    if not hasattr(block.mlp, "experts"):
        return
    for expert in block.mlp.experts:
        expert.gate_up_proj.fix_parametrization(expert.gate_up_in_transform)
        expert.down_proj.fix_parametrization(expert.down_in_transform)


@contextmanager
def _temporarily_disable_qlinear_quantizers(block: nn.Module):
    saved_quantizers = []
    for layer in block.modules():
        if isinstance(layer, QLinear):
            saved_quantizers.append((layer, layer.weight_quantizer, layer.act_quantizer))
            layer.weight_quantizer = None
            layer.act_quantizer = None
    try:
        yield
    finally:
        for layer, weight_quantizer, act_quantizer in saved_quantizers:
            layer.weight_quantizer = weight_quantizer
            layer.act_quantizer = act_quantizer


@contextmanager
def _temporary_qlinear_train_mode(block: nn.Module, train_mode: bool):
    saved_modes = []
    for layer in block.modules():
        if isinstance(layer, QLinear):
            saved_modes.append((layer, layer._train_mode))
            layer._train_mode = train_mode
    try:
        yield
    finally:
        for layer, old_mode in saved_modes:
            layer._train_mode = old_mode


def _get_alpha_candidates(args: argparse.Namespace) -> List[float]:
    if not args.smoothquant_search_alpha:
        if args.smoothquant_alpha is None:
            raise ValueError("`smoothquant_alpha` must be provided when SmoothQuant alpha search is disabled.")
        return [float(args.smoothquant_alpha)]

    if args.smoothquant_alpha_grid_size == 1:
        return [float(args.smoothquant_alpha_min)]

    return torch.linspace(
        args.smoothquant_alpha_min,
        args.smoothquant_alpha_max,
        args.smoothquant_alpha_grid_size,
        dtype=torch.float32,
    ).tolist()


@torch.no_grad()
def _smoothquant_alpha_loss(
    block: nn.Module,
    activation_maxes: Dict[str, torch.Tensor],
    alpha: float,
    input_args: Iterable[tuple],
    input_kwargs: Iterable[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    total_loss = 0.0
    num_samples = 0

    with _temporarily_disable_qlinear_quantizers(block), _temporary_qlinear_train_mode(block, True):
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            _set_identity_smoothquant_transforms(block)
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                original_out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            original_out = maybe_first_element(original_out).detach().float()

            _set_smoothquant_transforms(block, activation_maxes, args, alpha)
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                smooth_out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            smooth_out = maybe_first_element(smooth_out).detach().float()

            total_loss += torch.mean((smooth_out - original_out).pow(2)).item()
            num_samples += 1

    _set_identity_smoothquant_transforms(block)
    return total_loss / max(num_samples, 1)


def _select_smoothquant_alpha(
    block: nn.Module,
    activation_maxes: Dict[str, torch.Tensor],
    input_args: Iterable[tuple],
    input_kwargs: Iterable[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    candidates = _get_alpha_candidates(args)
    if len(candidates) == 1:
        return candidates[0]

    best_alpha = candidates[0]
    best_loss = float("inf")
    for alpha in candidates:
        loss = _smoothquant_alpha_loss(block, activation_maxes, alpha, input_args, input_kwargs, args, device)
        print(f"SmoothQuant alpha={alpha:.6g}, reconstruction loss={loss:.6e}")
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha

    print(f"Selected SmoothQuant alpha={best_alpha:.6g}, reconstruction loss={best_loss:.6e}")
    return best_alpha


def collect_smoothquant_activation_maxes(
    block: nn.Module,
    input_args: Iterable[tuple],
    input_kwargs: Iterable[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return _collect_activation_max(block, input_args, input_kwargs, args, device)


def apply_smoothquant_to_block(
    block: nn.Module,
    activation_maxes: Dict[str, torch.Tensor],
    input_args: Iterable[tuple],
    input_kwargs: Iterable[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    alpha = _select_smoothquant_alpha(block, activation_maxes, input_args, input_kwargs, args, device)
    _set_smoothquant_transforms(block, activation_maxes, args, alpha)
    return alpha


def smoothquant_quantization(
    model: AutoModelForCausalLM,
    calibration_data: List[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    print("SmoothQuant + RTN quantization...")
    if args.export_quantized_model == "realquant":
        if not (args.format == "int" and args.w_bits == 8 and args.a_bits == 8):
            raise NotImplementedError(
                "SmoothQuant realquant export is supported only for int W8A8 "
                "compressed-tensors checkpoints."
            )

    orig_dtype = model.config.torch_dtype if args.dtype == "auto" else args.dtype
    quantized_state_dict = {}
    non_quantized_state_dict = {}
    smoothquant_scale_state_dict = {}
    skip_linear_layer_name = []
    is_moe, ep_size, ep_rank, ep_group = _get_ep_context(model)
    weight_quantizer_kwargs, act_quantizer_kwargs = _get_quantizer_kwargs(args)

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

    for block_idx, block in enumerate(blocks):
        if ep_rank == 0:
            available = psutil.virtual_memory().available / 1024 / 1024 / 1024
            print(f"Processing block {block_idx}... available: {available:.2f} GB")
        if args.cpu_offload_modules:
            block.to(device)

        block = _build_quantized_block(
            model,
            block,
            block_idx,
            weight_quantizer_kwargs,
            act_quantizer_kwargs,
            is_moe,
            ep_size,
            ep_rank,
            ep_group,
        )
        block = block.to(device=device, dtype=orig_dtype)
        block.requires_grad_(False)

        activation_maxes = collect_smoothquant_activation_maxes(block, input_args, input_kwargs, args, device)
        apply_smoothquant_to_block(block, activation_maxes, input_args, input_kwargs, args, device)

        fix_smoothquant_block(block)

        if args.export_quantized_model:
            smoothquant_scale_state_dict.update(
                collect_smoothquant_scale_state(
                    block,
                    block_idx,
                    model.config,
                    is_moe,
                    ep_rank,
                    ep_size,
                )
            )
            for layer_name, layer in block.named_modules():
                if "experts" not in layer_name and ep_rank > 0:
                    continue
                if isinstance(layer, QLinear):
                    global_layer_name = (
                        get_global_layer_name(layer_name, ep_rank, model.config.num_local_experts)
                        if is_moe and ep_size > 1
                        else layer_name
                    )
                    weight = _fold_smoothquant_scale_into_weight(
                        block,
                        layer_name,
                        layer.weight.detach(),
                    )
                    if args.export_quantized_model == "realquant":
                        with torch.no_grad():
                            scales, zeros = layer.weight_quantizer.get_quantization_params(weight)
                            qweight = layer.weight_quantizer.quantize(weight, scales, zeros).to(torch.int8)
                        quantized_state_dict[f"model.layers.{block_idx}.{global_layer_name}"] = {
                            "weight": qweight.cpu(),
                            "weight_scale": scales.float().cpu(),
                        }
                    else:
                        quantized_state_dict[f"model.layers.{block_idx}.{global_layer_name}"] = {
                            "weight": weight.cpu(),
                        }
                elif ep_rank == 0 and hasattr(layer, "weight"):
                    non_quantized_state_dict[f"model.layers.{block_idx}.{layer_name}.weight"] = layer.weight.detach().cpu()
                    if isinstance(layer, torch.nn.Linear):
                        skip_linear_layer_name.append(f"model.layers.{block_idx}.{layer_name}")

        device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            out = maybe_first_element(out)
            if len(inp_args) > 0:
                inp_args[0].data.copy_(out)
            elif "hidden_states" in inp_kwargs:
                inp_kwargs["hidden_states"].copy_(out)
            else:
                raise ValueError("Unsupported block input format.")

        if args.cpu_offload_modules:
            block.self_attn = None
            block.mlp = None
            block = block.cpu()

        del activation_maxes
        gc.collect()
        clear_device_cache(garbage_collection=True)

    clear_device_cache(garbage_collection=True)
    args.smoothquant_scale_state_dict = smoothquant_scale_state_dict
    return quantized_state_dict, non_quantized_state_dict, skip_linear_layer_name
