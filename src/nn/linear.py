import math
from typing import Optional, Any

import torch
import torch.nn.functional as F
from torch import nn

from src.nn.module import Module
from src.nn.utils.activations import get_activation_fn


class Linear(Module):
    # See: https://discuss.pytorch.org/t/why-do-we-use-constants-or-final/70331/4.
    __constants__ = ["in_features", "out_features", "bias", "activation", "dropout_proba"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[str] = None,
        dropout_proba: float = 0.0,
        dtype: Optional[torch.dtype] = torch.float32,
        device: torch.device = torch.device("cuda"),
        **kwargs: Any,
    ):
        """Kernel fusion: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-pointwise-operations."""
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = get_activation_fn(activation, **kwargs)

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=dtype, device=device), requires_grad=True
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, dtype=dtype, device=device), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        self.dropout_proba = dropout_proba

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(
            y=F.linear(input=input, weight=self.weight, bias=None),
            bias=self.bias,
            dropout_proba=self.dropout_proba,
            training=self.training,
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        batch_size = 200
        max_length = 100
        embedding_dim = 300

        activation = ["gelu", None, "softmax"]
        dim = -1
        bias = True
        dropout_proba = 0.0

        output_dim = 30

    test_config = TestConfig()
    test_linear = Linear(
        in_features=test_config.embedding_dim,
        out_features=test_config.output_dim,
        activation=test_config.activation[0],
        bias=test_config.bias,
        dropout_proba=test_config.dropout_proba,
        dim=test_config.dim,
    )
    test_linear.print_params()
    test_input = torch.randn((test_config.batch_size, test_config.max_length, test_config.embedding_dim))
    assert test_linear(test_input).shape == (test_config.batch_size, test_config.max_length, test_config.output_dim)
    assert torch.allclose(
        test_linear(test_input), F.gelu(F.linear(input=test_input, weight=test_linear.weight, bias=test_linear.bias))
    )
    assert torch.sum(test_linear(test_input)) > 0

    test_linear = Linear(
        in_features=test_config.embedding_dim,
        out_features=test_config.output_dim,
        activation=test_config.activation[1],
        bias=test_config.bias,
        dropout_proba=test_config.dropout_proba,
        dim=test_config.dim,
    )
    test_input = torch.randn((test_config.batch_size, test_config.max_length, test_config.embedding_dim))
    assert torch.allclose(
        test_linear(test_input), F.linear(input=test_input, weight=test_linear.weight, bias=test_linear.bias)
    )

    test_linear = Linear(
        in_features=test_config.embedding_dim,
        out_features=test_config.output_dim,
        activation=test_config.activation[2],
        bias=test_config.bias,
        dropout_proba=test_config.dropout_proba,
        dim=test_config.dim,
    )
    test_input = torch.randn((test_config.batch_size, test_config.max_length, test_config.embedding_dim))
    assert torch.allclose(
        torch.sum(test_linear(test_input), dim=-1), torch.ones((test_config.batch_size, test_config.max_length))
    )
