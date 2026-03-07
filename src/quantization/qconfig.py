from typing import Any

def prepare_quantization_config(
    hadamard_group_size: int, 
    format: str,
    skip_linear_layer_name: list = None,
    pseudoquantization: bool = False
) -> dict[str, Any]:
    # Default ignore list
    ignore_list = ["lm_head"]
    
    # Add skip_linear_layer_name to ignore list if provided
    if skip_linear_layer_name:
        ignore_list.extend(skip_linear_layer_name)
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
