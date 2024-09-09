# MambaByte: Token-free Selective State Space Model

## Installation

```shell
conda create -n mambabyte-env python=3.9
conda activate mambabyte-env

# CUDA>=11.6 needed for `mamba-ssm` and `causal-conv1d`.
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

git clone https://github.com/TushaarGVS/mambabyte.git
cd mambabyte/

# Install PyTorch (with CUDA 11.8) before everything else.
pip install torch --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -e .
```

Make sure the following two commands produce the same CUDA version:

```shell
# See: https://github.com/state-spaces/mamba/issues/55#issuecomment-1858638484.
python3 -c "import torch; print(torch.version.cuda)"
nvcc --version
```

## Loading model

To load a pretrained `$HOME/mambabyte/pretrained_models/mambabyte_972M.pt`, run:

```python
from src.mambabyte.lm import MambaByteLM

# The following loads the 972M model; change accordingly for other models.
config = {
    "vocab_size": 256,
    "d_model": 1792,
    "num_layers": 48,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "use_fast_path": True,
    "norm_eps": 1e-5,
}
mambabyte_lm = MambaByteLM(**config)
mambabyte_lm.from_pretrained("$HOME/mambabyte/pretrained_models/mambabyte_972M.pt")
```

## Generating bytes

```shell
cd $HOME/mambabyte

prompt="High up on the hillside in the midst of a rugged group of jack pines the
Union Jack shook out its folds gallantly in the breeze that swept down
the Kicking Horse Pass. "
generate.py \
    --config_path "$HOME/mambabyte/scripts/configs/mambabyte_lm.yml" \
    --prompt "$prompt" \
    --max_new_tokens 300 \
    --num_samples 2 \
    --temperature 1.0 \
    --top_k 256 \
    --top_p 0.98 \
    --pretrained_model_filepath "$HOME/mambabyte/pretrained_models/mambabyte_972M.pt" \
    --model_id "972M" \
    --seed 4740
```

## Extracting module outputs

```shell
cd $HOME/mambabyte

prompt="High up on the hillside in the midst of a rugged group of jack pines the
Union Jack shook out its folds gallantly in the breeze that swept down
the Kicking Horse Pass. "
extract_module_outputs.py \
    --config_path "$HOME/mambabyte/scripts/configs/mambabyte_lm.yml" \
    --filepath_to_store_outputs "$HOME/mambabyte/artefacts/test_run_outputs.pkl" \
    --prompt "$prompt" \
    --layer_idxs 0 1 2 \
    --pretrained_model_filepath "$HOME/mambabyte/pretrained_models/mambabyte_972M.pt" \
    --model_id "972M" \
    --module_ids "dt_proj" \
    --nonlinearity "softplus" \
    --return_log_probs \
    --return_ranks \
    --seed 4740
```

To extract $\overline{\mathrm{A}}$ (discrete-A), run:

```shell
cd $HOME/mambabyte

prompt="High up on the hillside in the midst of a rugged group of jack pines the
Union Jack shook out its folds gallantly in the breeze that swept down
the Kicking Horse Pass. "
# discrete_A: (l, d, n)
extract_discrete_A.py \
    --config_path "$HOME/mambabyte/scripts/configs/mambabyte_lm.yml" \
    --filepath_to_store_outputs "$HOME/mambabyte/artefacts/test_discrete_A.pkl" \
    --prompt "$prompt" \
    --layer_idxs 0 1 \
    --pretrained_model_filepath "$HOME/mambabyte/pretrained_models/mambabyte_972M.pt" \
    --model_id "972M" \
    --return_log_probs \
    --return_ranks \
    --reduction "norm" \
    --reduction_dim -2 \
    --seed 4740
```

To extract the $\overline{\mathrm{A}}$ (or, delta) for a long PG19 sample, use the samples included in the
`pg19_samples` folder (these are the same samples used in Appendix F of MambaByte paper).

```shell
cd $HOME/mambabyte

prompt=$(<$HOME/mambabyte/pg19_samples/860.txt)
# Extract the frobenius norm of discrete-A.
extract_discrete_A.py \
    --config_path "$HOME/mambabyte/scripts/configs/mambabyte_lm.yml" \
    --filepath_to_store_outputs "$HOME/mambabyte/artefacts/discrete_A_860_fro.pkl" \
    --prompt "$prompt" \
    --pretrained_model_filepath "$HOME/mambabyte/pretrained_models/mambabyte_972M.pt" \
    --model_id "972M" \
    --return_log_probs \
    --return_ranks \
    --reduction "fro" \
    --reduction_dim -2 \
    --seed 4740
```

To extract input (cosine) similarity with $\overline{\mathrm{A}}$, run:

```shell
prompt=$(<$HOME/mambabyte/pg19_samples/860.txt)
extract_input_sim_with_discrete_A.py \
    --config_path "$HOME/mambabyte/scripts/configs/mambabyte_lm.yml" \
    --filepath_to_store_outputs "$HOME/mambabyte/artefacts/input_sim_with_discrete_A_860.pkl" \
    --prompt "$prompt" \
    --pretrained_model_filepath "$HOME/mambabyte/pretrained_models/mambabyte_972M.pt" \
    --model_id "972M" \
    --return_log_probs \
    --return_ranks \
    --seed 4740
```