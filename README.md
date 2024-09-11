# MambaByte: Token-free Selective State Space Model

## Installation

```shell
conda create -n mambabyte-env python=3.9
conda activate mambabyte-env

# CUDA>=11.6 needed for `mamba-ssm` and `causal-conv1d`.
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

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

```

## Generating bytes

```shell

```

## Speculative decoding

To enable resuming during the parallel scan, we extended the fast CUDA kernel, allowing verification to restart from the mismatched position instead of beginning from the start. 

Check the kernel to support resume from an exisit state in [here](csrc/README.md)

## Extracting module outputs

