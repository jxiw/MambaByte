from typing import Optional

import torch
from mamba_ssm.ops.triton.layernorm import rms_norm_fn
from torch import nn

from src.nn.module import Module


class RMSNorm(Module):
    def __init__(
        self,
        hidden_size: int,
        bias: bool = True,
        eps: float = 1e-5,
        dtype: Optional[torch.dtype] = torch.float32,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device), requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        residual_in_fp32: bool = True,
    ):
        return rms_norm_fn(
            x,
            weight=self.weight,
            bias=self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


if __name__ == "__main__":
    pass
