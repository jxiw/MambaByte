from typing import Union, Sequence, Tuple, List

import torch

from src.nn.module import Module


def id2token(token_id: int) -> str:
    return str(chr(token_id))


def decode(input_ids: Sequence[Union[int, torch.Tensor]]) -> str:
    return "".join(list(map(id2token, input_ids)))


@torch.inference_mode()
def generate(
    model: Module,
    prompt: str = None,
    max_new_tokens: int = 20,
    num_samples: int = 1,
    temperature: float = 1.0,
    top_k: int = 256,
    top_p: float = 0.0,
    return_dict_in_generate: bool = True,
    output_scores: bool = False,
) -> Union[List[str], Tuple[List[str], Sequence[torch.Tensor]]]:
    assert top_k <= 256 and top_p <= 1.0, f"{top_k=} must be <= 256 and {top_p=} must be <= 1.0"

    model._is_generating = True
    model.eval()

    if prompt is None:
        prompt = ""  # unconditional generation
    input_ids = torch.tensor(bytearray(prompt.encode("utf-8"))).long().expand(num_samples, -1).to(model.device)
    prompt_len = input_ids.shape[-1]

    # `cg` = cache_graph.
    samples = model.generate(
        input_ids=input_ids,
        max_length=(prompt_len + max_new_tokens),
        cg=True,
        return_dict_in_generate=return_dict_in_generate,
        output_scores=output_scores,
        enable_timing=False,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    generations = [decode(sequence[prompt_len:]) for sequence in samples.sequences]
    return generations if not output_scores else (generations, samples.scores)


if __name__ == "__main__":
    test_prompt = "hello there, alien!"
    test_input_ids = torch.tensor(bytearray(test_prompt.encode("utf-8"))).long()
    assert decode(test_input_ids) == test_prompt
