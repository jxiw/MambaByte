import os
import random
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int = 4740):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Setting `torch.backends.cudnn.benchmark = False` slows down training.
    # Reference: https://pytorch.org/docs/stable/notes/randomness.html.
    torch.backends.cudnn.benchmark = True


def set_pytorch_backends():
    # TF32: https://docs.monai.io/en/stable/precision_accelerating.html.
    # Set TF32 for speedup: https://x.com/GuggerSylvain/status/1599190137367166977?s=20.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_device(device_type: str = "auto") -> torch.device:
    if device_type == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_type == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_type == "mps":
        raise ValueError(f"{device_type=} not supported; must be one of ['cpu', 'cuda']")
    return torch.device("cpu")


def normalize_state_dict_prefix(model_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    old_to_new_prefix_mapping = {
        "module.": "",
        "backbone": "mambabyte",
        "embedding": "token_embedding",
        "norm_f": "final_norm",
    }
    for module_name, value in list(model_state_dict.items()):
        new_module_name = module_name
        for old_prefix, new_prefix in old_to_new_prefix_mapping.items():
            new_module_name = new_module_name.replace(old_prefix, new_prefix)
        model_state_dict[new_module_name] = model_state_dict.pop(module_name)
    return model_state_dict


if __name__ == "__main__":
    test_state_dict = {
        "module.backbone.embedding.weight": 0,
        "module.backbone.layers.0.mixer.A_log": 1,
        "module.backbone.norm_f.bias": 2,
    }
    updated_test_state_dict_keys = list(normalize_state_dict_prefix(test_state_dict).keys())
    assert updated_test_state_dict_keys[0] == "mambabyte.token_embedding.weight"
    assert updated_test_state_dict_keys[1] == "mambabyte.layers.0.mixer.A_log"
    assert updated_test_state_dict_keys[2] == "mambabyte.final_norm.bias"
