from functools import partial
from typing import Optional, Union, Callable

import torch
import torch.nn.functional as F

from src.nn.utils.utils import set_jit_flags

set_jit_flags()

_supported_activations = [
    None,
    "linear",
    "relu",
    "leaky_relu",
    "lrelu",
    "elu",
    "gelu",
    "swish",
    "silu",
    "tanh",
    "softmax",
    "softplus",
]


@torch.jit.script
def linear(
    y: torch.Tensor, bias: Optional[torch.Tensor] = None, dropout_proba: float = 0.0, training: bool = True
) -> torch.Tensor:
    y = y + bias if bias is not None else y
    return F.dropout(y, p=dropout_proba, training=training) if dropout_proba > 0.0 else y


@torch.jit.script
def relu(
    y: torch.Tensor, bias: Optional[torch.Tensor] = None, dropout_proba: float = 0.0, training: bool = True
) -> torch.Tensor:
    y = y + bias if bias is not None else y
    y = F.relu(y)
    return F.dropout(y, p=dropout_proba, training=training) if dropout_proba > 0.0 else y


@torch.jit.script
def leaky_relu(
    y: torch.Tensor, bias: Optional[torch.Tensor] = None, dropout_proba: float = 0.0, training: bool = True
) -> torch.Tensor:
    y = y + bias if bias is not None else y
    y = F.leaky_relu(y)
    return F.dropout(y, p=dropout_proba, training=training) if dropout_proba > 0.0 else y


@torch.jit.script
def elu(
    y: torch.Tensor, bias: Optional[torch.Tensor] = None, dropout_proba: float = 0.0, training: bool = True
) -> torch.Tensor:
    y = y + bias if bias is not None else y
    y = F.elu(y)
    return F.dropout(y, p=dropout_proba, training=training) if dropout_proba > 0.0 else y


@torch.jit.script
def gelu(
    y: torch.Tensor, bias: Optional[torch.Tensor] = None, dropout_proba: float = 0.0, training: bool = True
) -> torch.Tensor:
    y = y + bias if bias is not None else y
    y = F.gelu(y)
    return F.dropout(y, p=dropout_proba, training=training) if dropout_proba > 0.0 else y


@torch.jit.script
def swish(
    y: torch.Tensor, bias: Optional[torch.Tensor] = None, dropout_proba: float = 0.0, training: bool = True
) -> torch.Tensor:
    y = y + bias if bias is not None else y
    y = F.silu(y)
    return F.dropout(y, p=dropout_proba, training=training) if dropout_proba > 0.0 else y


@torch.jit.script
def tanh(
    y: torch.Tensor, bias: Optional[torch.Tensor] = None, dropout_proba: float = 0.0, training: bool = True
) -> torch.Tensor:
    y = y + bias if bias is not None else y
    y = F.tanh(y)
    return F.dropout(y, p=dropout_proba, training=training) if dropout_proba > 0.0 else y


@torch.jit.script
def softplus(
    y: torch.Tensor, bias: Optional[torch.Tensor] = None, dropout_proba: float = 0.0, training: bool = True
) -> torch.Tensor:
    y = y + bias if bias is not None else y
    y = F.softplus(y)
    return F.dropout(y, p=dropout_proba, training=training) if dropout_proba > 0.0 else y


@torch.jit.script
def softmax(
    y: torch.Tensor, dim: int, bias: Optional[torch.Tensor] = None, dropout_proba: float = 0.0, training: bool = True
) -> torch.Tensor:
    y = y + bias if bias is not None else y
    y = F.softmax(y, dim=dim)
    return F.dropout(y, p=dropout_proba, training=training) if dropout_proba > 0.0 else y


def get_activation_fn(
    activation: Optional[str] = None, **kwargs: int
) -> Union[partial[Callable], Callable[[torch.Tensor, Optional[torch.Tensor], float, bool], torch.Tensor]]:
    if activation is None or activation == "linear":
        return linear
    elif activation == "elu":
        return elu
    elif activation == "gelu":
        return gelu
    elif activation in ["swish", "silu"]:
        return swish
    elif activation == "tanh":
        return tanh
    elif activation == "relu":
        return relu
    elif activation in ["lrelu", "leaky_relu"]:
        return leaky_relu
    elif activation == "softplus":
        return softplus
    elif activation == "softmax":
        return partial(softmax, dim=kwargs["dim"])
    else:
        raise ValueError(f"{activation} not supported; must be one of {_supported_activations}")


if __name__ == "__main__":
    for activation_name, pytorch_activation_fn in zip(
        ["relu", "leaky_relu", "elu", "gelu", "swish", "tanh", "softplus"],
        [F.relu, F.leaky_relu, F.elu, F.gelu, F.silu, F.tanh, F.softplus],
    ):
        test_input, test_bias = torch.randn((500,)), torch.randn((500,))

        activation_fn = get_activation_fn(activation_name)
        assert torch.allclose(activation_fn(test_input, test_bias), pytorch_activation_fn(test_input + test_bias))
        assert torch.allclose(activation_fn(test_input, None, 1.0), F.dropout(pytorch_activation_fn(test_input), 1.0))
        assert torch.allclose(
            activation_fn(test_input, test_bias, 1.0), F.dropout(pytorch_activation_fn(test_input + test_bias), 1.0)
        )
        assert torch.allclose(activation_fn(test_input, None), pytorch_activation_fn(test_input))

    test_input = torch.randn((500, 1000, 300))
    assert torch.allclose(softmax(test_input, dim=-1), F.softmax(test_input, dim=-1))
