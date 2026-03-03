from typing import Any

def prepare_quantization_config(
    hadamard_group_size: int, 
    format: str,
    pseudoquantization: bool = False
) -> dict[str, Any]:
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
            "ignore": ["lm_head"],
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
