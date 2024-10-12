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

## Generating bytes

Code: [MambaByte_Code](https://huggingface.co/JunxiongWang/MambaByte_Code)
```python
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import numpy as np

model=MambaLMHeadModel.from_pretrained("JunxiongWang/MambaByte_Code", device='cuda', dtype=torch.float32)

text = "import torch"
text_byte = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
input_ids = torch.from_numpy(text_byte[None, :].copy()).long().cuda()

sample = model.generate(
    input_ids=input_ids,
    max_length=2048,
    cg=True,
    return_dict_in_generate=True,
    output_scores=True,
    enable_timing=True,
    temperature=1,
    top_k=256,
    top_p=0.9,
)

print(bytes(sample.sequences[0].tolist()).decode('utf-8'))
```

Stories: [MambaByte_Arxiv](https://huggingface.co/JunxiongWang/MambaByte_Stories)

Arxiv: [MambaByte_Arxiv](https://huggingface.co/JunxiongWang/MambaByte_Arxiv)

Books: [MambaByte_Books](https://huggingface.co/JunxiongWang/MambaByte_Books)

PG19 972M: [MambaByte_Stories](https://huggingface.co/JunxiongWang/MambaByte_PG19_972M)

PG19 353M: [MambaByte_PG19_353M](https://huggingface.co/JunxiongWang/MambaByte_PG19_353M)

# Compute bit-per-byte and PPL

```python
# This script defines functions to convert between two commonly used metrics in language model evaluation: 
# Byte-level Cross-Entropy (byte_ce), Bits Per Byte (bpb), and Perplexity (ppl), considering different length ratios (LT_by_LB).
# LT_by_LB is the ratio of total tokens (L_T) to total bytes (L_B), which varies depending on whether 
# word-level or subword-level tokenization is used.

import math

# Converts Byte-level Cross-Entropy (byte_ce) to word-level perplexity (ppl), using the ratio LT_by_LB (L_T/L_B).
def ppl_from_byte_ce(byte_ce, LT_by_LB):
    # Formula: ppl = exp(byte_ce / (L_T / L_B))
    return math.exp(byte_ce / LT_by_LB)

# Converts Bits Per Byte (bpb) to word-level perplexity (ppl), using the ratio LT_by_LB.
def ppl_from_bpb(bpb, LT_by_LB):
    # Formula: ppl = exp(bpb * log(2) / (L_T / L_B))
    return f"word_ppl={math.exp(bpb * math.log(2) / LT_by_LB)}"

def bpb_from_ppl(ppl, LT_by_LB):
    # Formulas:
    # bpb = (L_T / L_B) * log(ppl) / log(2)
    # byte_ce = (L_T / L_B) * log(ppl)
    return f"\tbpb={LT_by_LB * math.log(ppl) / math.log(2)}\n\tbyte_ce={LT_by_LB * math.log(ppl)}"

# We use the `wc -w` command to count the number of words (this is roughly consistent with the report in Rae et al. (2020)).
# pg19 train: 1973048393, val: 3007061, test: 6965511,
# We count the number of byte
# pg19 train: 11677824216, val: 17733002, test: 41289101
# We count the number of subwords using a SentencePiece tokenizer (32k vocab size). 

# LT_by_LB values for the PG-19 dataset (word-level) are precomputed for training, validation, and test sets.
pg19_train_LT_by_LB = 1973048393 / 11677824216
pg19_val_LT_by_LB = 3007061 / 17733002
pg19_test_LT_by_LB = 6965511 / 41289101

# LT_by_LB values for the PG-19 dataset (subword-level) are precomputed for training, validation, and test sets.
# We count the number of tokens using a subword tokenizer (see following section for that),
# pg19 train: 2914600562, val: 4357506, test: 10282006
pg19_train_LT_by_LB_subword = 1973048393 / 2914600562
pg19_val_LT_by_LB_subword = 3007061/ 4357506
pg19_test_LT_by_LB_subword = 6965511 / 10282006

# convert ppl using the subword tokenizer to ppl 
# assume subword tokenizer cross entropy loss is 2.399497291852516
ppl_from_byte_ce(byte_ce=2.399497291852516, LT_by_LB=pg19_test_LT_by_LB_subword)

# convert ppl using the byte level tokenizer to ppl 
# assume byte cross entropy loss is 0.5901387288495581
ppl_from_byte_ce(byte_ce=0.5901387288495581, LT_by_LB=pg19_test_LT_by_LB)
```

### Subword tokenizer

We follow Rae et al. (2020) to create a tokenizer using the PG-19 training set.

```python
import tensorflow_datasets as tfds

pg_train_path = 'pg-19_train.txt'

with open(pg_train_path, 'r') as file:
    # Read the entire content of the file into a single string
    pg_content = file.read()

# Train a SubwordTextEncoder on your dataset
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    [pg_content], target_vocab_size=32000)

# Save the tokenizer for later use
tokenizer.save_to_file('pg_subword_tokenizer')
```

Get subword tokens,

```python
import tensorflow_datasets as tfds
import pickle

vocab_fname = 'pg_subword_tokenizer'
encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(vocab_fname)

pg_train_path = 'pg-19_train.txt'
pg_validation_path = 'pg-19_validation.txt'
pg_test_path = 'pg-19_test.txt'

with open(pg_train_path, 'r') as file:
    pg_train_text = file.read()

with open(pg_validation_path, 'r') as file:
    pg_validation_text = file.read()
    
with open(pg_test_path, 'r') as file:
    pg_test_text = file.read()

pg_train_tokens = encoder.encode(pg_train_text)
with open('pg-19_train.pickle', 'wb') as file:
    pickle.dump(pg_train_tokens, file)

pg_validation_tokens = encoder.encode(pg_validation_text)
with open('pg-19_validation.pickle', 'wb') as file:
    pickle.dump(pg_validation_tokens, file)

pg_test_tokens = encoder.encode(pg_test_text)
with open('pg-19_test.pickle', 'wb') as file:
    pickle.dump(pg_test_tokens, file)
```

## Speculative decoding

To enable resuming during the parallel scan, we extended the fast CUDA kernel, allowing verification to restart from the mismatched position instead of beginning from the start. 

Check the kernel to support resume from an exist state in [here](csrc/)

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

## Citation

If you use this codebase, or otherwise found our work valuable, please cite:

```
@article{wang2024mambabyte,
  title={Mambabyte: Token-free selective state space model},
  author={Wang, Junxiong and Gangavarapu, Tushaar and Yan, Jing Nathan and Rush, Alexander M},
  journal={arXiv preprint arXiv:2401.13660},
  year={2024}
}
```



