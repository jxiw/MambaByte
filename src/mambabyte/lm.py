from collections import namedtuple
from typing import Optional, Dict, Tuple, NamedTuple

import torch
from mamba_ssm.utils.generation import GenerationMixin, InferenceParams

from src.mambabyte.model import MambaByte
from src.nn.linear import Linear
from src.nn.module import Module


class MambaByteLM(Module, GenerationMixin):
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
        is_generating: bool = False,
        dtype: Optional[torch.dtype] = torch.float32,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()

        self._is_generating = is_generating

        self.mambabyte = MambaByte(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=use_fast_path,
            disable_optimizations=disable_optimizations,
            norm_eps=norm_eps,
            dtype=dtype,
            device=device,
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            bias=False,
            activation=None,
            dropout_proba=0.0,
            dtype=dtype,
            device=device,
        )

    def allocate_inference_cache(
        self, batch_size: int, max_seq_length: int, dtype: Optional[torch.device] = torch.float32, **kwargs
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        return self.mambabyte.allocate_inference_cache(
            batch_size=batch_size, max_seq_length=max_seq_length, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        num_last_tokens: int = 0,
    ) -> NamedTuple:
        # Unused param: `position_ids` (for Mamba generation modules).
        hidden_states = self.mambabyte(input_ids=input_ids, inference_params=inference_params)
        if self._is_generating and num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        logits = self.lm_head(hidden_states)

        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=logits)


if __name__ == "__main__":
    pass
