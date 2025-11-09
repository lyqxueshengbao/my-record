from .args import update_config_dict, parse_configs, parse_transforms
from .carrada_functions import get_transformations, get_metrics, get_class_weights
from .models_utils import get_models
import torch.nn as nn
from functools import partial

def get_norm_layer(norm: str):
    """
    Return the normalization layer constructor
    @param norm: (str) normalization type (batchnorm, layernorm, groupnorm)
    @return: normalization layer constructor
    """
    if norm == 'batchnorm':
        return nn.BatchNorm2d
    elif norm == 'layernorm':
        # Use GroupNorm with 1 group as a LayerNorm drop-in for CNNs
        return partial(nn.GroupNorm, 1)
    elif norm == 'groupnorm':
        # Default to 32 groups for GroupNorm, a common setting
        return partial(nn.GroupNorm, 32)
    else:
        raise ValueError(f"Unknown normalization layer: {norm}")