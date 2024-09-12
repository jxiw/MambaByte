from argparse import ArgumentParser
from typing import Optional, Literal

import torch
import yaml

from src.mambabyte.lm import MambaByteLM
from src.utils.lm_utils import generate
from src.utils.torch_utils import get_device, set_pytorch_backends, set_seed

set_pytorch_backends()


@torch.no_grad()
def main(
    config_path: str,
    prompt: str = None,
    max_new_tokens: int = 20,
    num_samples: int = 1,
    temperature: float = 1.0,
    top_k: int = 256,
    top_p: float = 0.0,
    pretrained_model_filepath: Optional[str] = None,
    model_id: Literal["353M", "972M"] = "972M",
    seed: int = 4740,
) -> None:
    set_seed(seed)

    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    dtype = torch.float32 if config["general"]["dtype"] == "fp32" else torch.bfloat16
    device = get_device(config["general"]["device"])
    for key, value in config.items():
        if isinstance(value, dict):
            if model_id in value:
                config[key] = config[key]["common"]
                config[key].update(value[model_id])

    model = MambaByteLM(dtype=dtype, device=device, **config["model"], **config["generation"])
    if pretrained_model_filepath is not None:
        model.from_pretrained(pretrained_model_filepath)

    generations = generate(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=False,
    )

    print(f"PROMPT\n{'=' * 6}\n{prompt}\n\nGENERATIONS\n{'=' * 11}")
    for generation in generations:
        print(f"{generation}\n{'-' * 80}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate (text) bytes using the pretrained MambaByte model.")

    parser.add_argument("--config_path", type=str, help="Path to the config file.", required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        help="The input prompt to autocomplete; if not specified, with default to an empty string.",
        required=False,
        default=None,
    )
    parser.add_argument("--max_new_tokens", type=int, help="Maximum number of new tokens to generate.", default=100)
    parser.add_argument("--num_samples", type=int, help="Number of samples to generate.", default=1)
    parser.add_argument("--temperature", type=float, help="The generation temperature (higher = diverse).", default=1.0)
    parser.add_argument(
        "--top_k",
        type=int,
        help="If specified, only the top-k candidates will be used in generating; top-k is applied before top-p.",
        default=256,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="If specified, only the top-p candidates will be used in generating; top-k is applied before top-p.",
        default=0.98,
    )
    parser.add_argument(
        "--pretrained_model_filepath",
        type=str,
        help="Path to pretrained model file; if not provided, a random initialization will be used.",
        default=None,
    )
    parser.add_argument("--model_id", choices=["353M", "972M"], help="The MambaByte model identifier.", default="972M")
    parser.add_argument("--seed", type=int, help="Random seed (for reproducibility).", default="4740")

    args = parser.parse_args()

    main(
        config_path=args.config_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        pretrained_model_filepath=args.pretrained_model_filepath,
        model_id=args.model_id,
        seed=args.seed,
    )
