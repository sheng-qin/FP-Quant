from typing import Any


def _get_ignore_list(skip_linear_layer_name: list = None) -> list[str]:
    ignore_list = ["lm_head"]
    if skip_linear_layer_name:
        ignore_list.extend(skip_linear_layer_name)
    return ignore_list


def prepare_compressed_tensors_w8a8_config(
    weight_strategy: str,
    skip_linear_layer_name: list = None,
) -> dict[str, Any]:
    if weight_strategy not in {"tensor", "channel"}:
        raise ValueError(
            "`compressed-tensors` W8A8 export supports tensor/channel weight "
            f"strategies for vLLM, got: {weight_strategy}"
        )

    return {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "memoryless",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": weight_strategy,
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "int-quantized",
        "ignore": _get_ignore_list(skip_linear_layer_name),
        "quant_method": "compressed-tensors",
        "quantization_status": "frozen",
    }


def prepare_quantization_config(
    hadamard_group_size: int, 
    format: str,
    skip_linear_layer_name: list = None,
    pseudoquantization: bool = False
) -> dict[str, Any]:
    # Default ignore list
    ignore_list = _get_ignore_list(skip_linear_layer_name)
    if format in ["nvfp"]:
        return {
            "config_groups": {
                "group_0": {
                    "input_activations": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16
                    },
                    "weights": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16
                    },
                    "targets": ["Linear"]
                }
            },
            "ignore": ignore_list,
            "quant_algo": "NVFP4",
            "kv_cache_scheme": {
                "dynamic": False,
                "num_bits": 8,
                "type": "float"
            },
            "producer": {
                "name": "modelopt",
                "version": "0.35.0"
            },
            "quant_method": "modelopt"
        }
    else:
        raise ValueError(f"Invalid format: {format}")
