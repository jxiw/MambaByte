from collections import namedtuple
from functools import partial
from typing import Any, Union, List, Optional, NamedTuple, Literal, Callable

import torch
import torch.nn.functional as F
from einops import einsum
from torch import nn

from src.nn.module import Module
from src.nn.utils.activations import get_activation_fn
from src.nn.utils.activations import softmax


def _get_layer_idx_from_module_name(module_name: str) -> Optional[int]:
    try:
        layer_idx = int(module_name.split(".")[2])
    except IndexError:
        # Modules such as `token_embedding`, `final_norm` don't belong a specific layer.
        layer_idx = None
    return layer_idx


def _get_reducer(
    reduction: Optional[Literal["mean", "min", "max", "norm", "fro", "nuc"]] = None, reduction_dim: Optional[int] = None
) -> Optional[partial[Callable]]:
    if reduction == "mean":
        return partial(torch.mean, dim=reduction_dim)
    elif reduction == "min":
        return partial(torch.min, dim=reduction_dim)
    elif reduction == "max":
        return partial(torch.max, dim=reduction_dim)
    elif reduction == "norm":
        return partial(torch.norm, p="fro", dim=reduction_dim)
    elif reduction == "fro":
        return partial(torch.norm, p="fro", dim=(1, 2))  # one norm value per timestep
    elif reduction == "nuc":
        return partial(torch.norm, p="nuc", dim=(1, 2))  # one norm value per timestep
    return None


def _attach_hooks(
    model: Module,
    module_id: str,
    layer_idxs: List[int],
    nonlinearity: Optional[str] = None,
) -> None:
    if nonlinearity is None:
        nonlinearity = "linear"
    nonlinearity_fn = partial(get_activation_fn(nonlinearity), training=False)

    def get_module_fwd_hook(
        _module: Union[nn.Module, Module], _inputs: torch.Tensor, _outputs: Any, _layer_idx: int
    ) -> None:
        # Debug: default arguments have to be to the end; see https://stackoverflow.com/a/26182275.
        # _outputs.squeeze(0): (L, 256=vocab_size)
        model.hooks["outputs"][module_id][_layer_idx] = nonlinearity_fn(_outputs.detach().cpu().squeeze(0))

    model.hooks["outputs"][module_id] = {}
    for name, module in model.named_modules():
        layer_idx = _get_layer_idx_from_module_name(name)
        if layer_idx is not None and module_id in name and layer_idx in layer_idxs:
            model.hooks["outputs"][module_id][layer_idx] = []
            model.attach_hook(
                module=module,
                hook=partial(get_module_fwd_hook, _layer_idx=layer_idx),
                hook_type="forward",
            )


def _get_preds(
    model: Module,
    prompt: str,
    return_log_probs: bool = True,
    return_ranks: bool = True,
) -> NamedTuple:
    input_ids = torch.tensor(bytearray(prompt.encode("utf-8"))).long()[None, :].to(model.device)
    logits = model(input_ids=input_ids).logits.squeeze(0)  # (L, 256=vocab_size)

    teacher_forced_input_probs, ranks = None, None
    if return_log_probs or return_ranks:
        probs = softmax(logits, dim=-1, training=False)  # (L, 256=vocab_size)
        teacher_forced_input_probs = probs[:-1, :].gather(dim=1, index=input_ids[:, 1:].T)  # autoregressive: (L - 1,)
        ranks = (probs[:-1, :] > teacher_forced_input_probs).long().sum(dim=1)

    PredictionArtefacts = namedtuple("PredictionArtefacts", ["log_probs", "ranks"])
    return PredictionArtefacts(
        log_probs=teacher_forced_input_probs.squeeze(1).detach().cpu() if return_log_probs else None,
        ranks=ranks.detach().cpu() if return_ranks else None,
    )


def get_module_outputs(
    model: Module,
    module_id: str,
    layer_idxs: List[int],
    prompt: str,
    nonlinearity: Optional[str] = None,
    return_log_probs: bool = True,
    return_ranks: bool = True,
) -> NamedTuple:
    _attach_hooks(model=model, module_id=module_id, layer_idxs=layer_idxs, nonlinearity=nonlinearity)
    return _get_preds(model=model, prompt=prompt, return_log_probs=return_log_probs, return_ranks=return_ranks)


def get_discrete_A(
    model: Module,
    layer_idxs: List[int],
    prompt: str,
    return_log_probs: bool = True,
    return_ranks: bool = True,
    reduction: Optional[Literal["mean", "min", "max", "norm", "fro", "nuc"]] = None,
    reduction_dim: Optional[int] = None,
) -> NamedTuple:
    prediction_artefacts = get_module_outputs(
        model=model,
        module_id="dt_proj",
        layer_idxs=layer_idxs,
        prompt=prompt,
        nonlinearity="softplus",
        return_log_probs=return_log_probs,
        return_ranks=return_ranks,
    )

    reducer = _get_reducer(reduction=reduction, reduction_dim=reduction_dim)
    model.hooks["outputs"]["discrete_A"] = {}
    for layer_idx in layer_idxs:
        A = -torch.exp(model.mambabyte.layers[layer_idx].mixer.A_log.detach().cpu().float())
        delta = model.hooks["outputs"]["dt_proj"].pop(layer_idx)
        delta_A = torch.exp(einsum(delta, A, "l d, d n -> l d n"))
        if reducer is not None:
            delta_A = reducer(delta_A)
        model.hooks["outputs"]["discrete_A"][layer_idx] = delta_A

    assert model.hooks["outputs"].pop("dt_proj") == {}
    return prediction_artefacts


def get_input_sim_with_discrete_A(
    model: Module,
    layer_idxs: List[int],
    prompt: str,
    return_log_probs: bool = True,
    return_ranks: bool = True,
) -> NamedTuple:
    _attach_hooks(model=model, module_id="dt_proj", layer_idxs=layer_idxs, nonlinearity="softplus")
    _attach_hooks(model=model, module_id="in_proj", layer_idxs=layer_idxs, nonlinearity=None)
    prediction_artefacts = _get_preds(
        model=model, prompt=prompt, return_log_probs=return_log_probs, return_ranks=return_ranks
    )

    model.hooks["outputs"]["input_sim_with_discrete_A"] = {}
    for layer_idx in layer_idxs:
        A = -torch.exp(model.mambabyte.layers[layer_idx].mixer.A_log.detach().cpu().float())
        delta = model.hooks["outputs"]["dt_proj"].pop(layer_idx)
        delta_A = torch.exp(einsum(delta, A, "l d, d n -> l d n"))
        expanded_x, _ = model.hooks["outputs"]["in_proj"].pop(layer_idx).chunk(2, dim=1)  # expanded_x: (L, d_inner)
        model.hooks["outputs"]["input_sim_with_discrete_A"][layer_idx] = F.cosine_similarity(
            delta_A, expanded_x.unsqueeze(2), dim=1
        )

    return prediction_artefacts


if __name__ == "__main__":
    pass
