from functools import partial
from typing import Optional, Dict, Tuple

import torch
from mamba_ssm.ops.triton.layernorm import layer_norm_fn
from mamba_ssm.utils.generation import InferenceParams
from torch import nn

from src.mambabyte.mamba import Block, Mamba
from src.nn.module import Module


class MambaByte(Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 1792,
        num_layers: int = 48,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_fast_path: bool = True,
        disable_optimizations: bool = False,
        norm_eps: float = 1e-5,
        dtype: Optional[torch.dtype] = torch.float32,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()

        self._d_model = d_model
        self._d_state = d_state
        self._d_conv = d_conv
        self._expand = expand
        self._use_fast_path = use_fast_path
        self._disable_optimizations = disable_optimizations
        self._norm_eps = norm_eps
        self._dtype = dtype
        self._device = device

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList([self._create_mamba_block(layer_idx=layer_idx) for layer_idx in range(num_layers)])
        self.final_norm = nn.LayerNorm(normalized_shape=d_model, bias=True, eps=norm_eps, device=device, dtype=dtype)

    def _create_mamba_block(self, layer_idx: int) -> Block:
        mamba_mixer = partial(
            Mamba,
            d_state=self._d_state,
            d_conv=self._d_conv,
            expand=self._expand,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=self._use_fast_path,
            disable_optimizations=self._disable_optimizations,
            device=self._device,
            dtype=self._dtype,
            layer_idx=layer_idx,
        )
        norm = partial(nn.LayerNorm, bias=True, eps=self._norm_eps, device=self._device, dtype=self._dtype)
        mamba_block = Block(
            dim=self._d_model, mixer_cls=mamba_mixer, norm_cls=norm, fused_add_norm=True, residual_in_fp32=True
        )
        mamba_block.layer_idx = layer_idx
        return mamba_block

    def allocate_inference_cache(
        self, batch_size: int, max_seq_length: int, dtype: Optional[torch.device] = torch.float32, **kwargs
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        return {
            layer_idx: layer.allocate_inference_cache(
                batch_size=batch_size, max_seqlen=max_seq_length, dtype=dtype, **kwargs
            )
            for layer_idx, layer in enumerate(self.layers)
        }

    def forward(self, input_ids: torch.tensor, inference_params: Optional[InferenceParams] = None) -> torch.Tensor:
        hidden_states = self.token_embedding(input_ids)  # (b, L, e)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states=hidden_states, residual=residual, inference_params=inference_params
            )

        # Set `prenorm` to False since we don't need the residual.
        hidden_states = layer_norm_fn(
            hidden_states,
            weight=self.final_norm.weight,
            bias=self.final_norm.bias,
            eps=self._norm_eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )
        return hidden_states


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        batch_size = 100
        vocab_size = 256
        d_model = 1792

        num_layers = 48
        d_state = 16
        d_conv = 4
        expand = 2
        use_fast_path = True

    test_config = TestConfig()
    test_mambabyte = MambaByte(
        vocab_size=test_config.vocab_size,
        d_model=test_config.d_model,
        num_layers=test_config.num_layers,
        d_state=test_config.d_state,
        d_conv=test_config.d_conv,
        expand=test_config.expand,
        use_fast_path=test_config.use_fast_path,
    )
    test_mambabyte.print_params()
