from argparse import ArgumentParser
from typing import Optional, Literal, List

import torch
import yaml

from src.mambabyte.lm import MambaByteLM
from src.utils.torch_utils import get_device, set_pytorch_backends, set_seed
from src.utils.utils import dict_to_pickle
from src.utils.visualization_utils import get_discrete_A

set_pytorch_backends()


@torch.no_grad()
def main(
    config_path: str,
    filepath_to_store_outputs: str,
    prompt: str,
    layer_idxs: Optional[List[int]] = None,
    pretrained_model_filepath: Optional[str] = None,
    model_id: Literal["353M", "972M"] = "972M",
    return_log_probs: bool = True,
    return_ranks: bool = True,
    reduction: Optional[Literal["mean", "min", "max", "norm", "fro", "nuc"]] = None,
    reduction_dim: Optional[int] = None,
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

    model = MambaByteLM(dtype=dtype, device=device, **config["model"], **config["visualization"])
    if pretrained_model_filepath is not None:
        model.from_pretrained(pretrained_model_filepath)

    if layer_idxs is None:
        layer_idxs = list(range(config["model"]["num_layers"]))
    prediction_artefacts = get_discrete_A(
        model=model,
        layer_idxs=layer_idxs,
        prompt=prompt,
        return_log_probs=return_log_probs,
        return_ranks=return_ranks,
        reduction=reduction,
        reduction_dim=reduction_dim,
    )

    return_dict = {
        "prompt": prompt,
        "discrete_A": model.hooks["outputs"]["discrete_A"],
        "log_probs": prediction_artefacts.log_probs,
        "ranks": prediction_artefacts.ranks,
        "model_id": model_id,
        "layer_idxs": layer_idxs,
        "reduction": reduction,
        "pretrained_model_filepath": pretrained_model_filepath,
    }
    dict_to_pickle(filepath_to_store_outputs=filepath_to_store_outputs, dict_to_write=return_dict)
    print(f"stored the outputs to {filepath_to_store_outputs}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract discrete-A (hidden -> hidden transition) matrix from the pretrained MambaByte model."
    )

    parser.add_argument("--config_path", type=str, help="Path to the config file.", required=True)
    parser.add_argument(
        "--filepath_to_store_outputs",
        type=str,
        help="Basepath to store the outputs (e.g., `/tmp/tg352.pkl`).",
        required=True,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="The input prompt to autocomplete; if not specified, with default to an empty string.",
        required=True,
    )
    parser.add_argument(
        "--layer_idxs",
        nargs="+",
        type=int,
        help="Layer indices to extract delta from; if unspecified, module outputs from all layers are extracted.",
    )
    parser.add_argument(
        "--pretrained_model_filepath",
        type=str,
        help="Path to pretrained model file; if not provided, a random initialization will be used.",
        default=None,
    )
    parser.add_argument("--model_id", choices=["353M", "972M"], help="The MambaByte model identifier.", default="972M")
    parser.add_argument("--return_log_probs", action="store_true", help="Indicator to return log probabilities.")
    parser.add_argument(
        "--return_ranks",
        action="store_true",
        help="Indicator to return the rank of the true token using the predicted probabilities.",
    )
    parser.add_argument(
        "--reduction",
        choices=["mean", "min", "max", "norm", "fro", "nuc"],
        help="Indicates how to aggregate the discrete-A matrix (fro: frobenius norm, nuc: nuclear norm).",
        default=None,
    )
    parser.add_argument("--reduction_dim", type=int, help="The dimension to reduce discrete-A on.", default=None)
    parser.add_argument("--seed", type=int, help="Random seed (for reproducibility).", default="4740")

    args = parser.parse_args()

    main(
        config_path=args.config_path,
        filepath_to_store_outputs=args.filepath_to_store_outputs,
        prompt=args.prompt,
        layer_idxs=args.layer_idxs,
        pretrained_model_filepath=args.pretrained_model_filepath,
        model_id=args.model_id,
        return_log_probs=args.return_log_probs,
        return_ranks=args.return_ranks,
        reduction=args.reduction,
        reduction_dim=args.reduction_dim,
        seed=args.seed,
    )
