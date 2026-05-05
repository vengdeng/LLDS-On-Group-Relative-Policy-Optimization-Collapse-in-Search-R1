
<h1 align="center">[ICML2026] On GRPO Collapse in Search-R1: The Lazy Likelihood-Displacement Death Spiral</h1>


<p align="center">
📃 <a href="https://arxiv.org/abs/2512.04220" target="_blank">Paper</a> </a> ｜🤗 <a href="https://huggingface.co/SEGAgentRL" target="_blank">LLDS-Huggingface</a> ｜🐙 <a href="https://github.com/vengdeng/LLDS-On-Group-Relative-Policy-Optimization-Collapse-in-Search-R1" target="_blank">GitHub</a>
</p>

## ⚡ Introduction

**LLDS** (Lazy Likelihood-Displacement Stabilization) is a lightweight likelihood-preserving regularization for tool-integrated RL training (GRPO / Search-R1 style).

In tool-use RL, we identify **Lazy Likelihood Displacement (LLD)** as a key collapse mechanism: the policy can drift by reducing likelihood on useful actions.  
LLDS stabilizes this by activating regularization **selectively**:

- only when likelihood decreases,
- only on preserving sets (e.g., non-negative-advantage actions),
- and only on responsible tokens/spans.

We study two gates:

- **A-gate**: action-level gate (best performance in our experiments)
- **R-gate**: response-level gate

## 🔍 Tool-Integrated Search Inference (Search-R1 style)

This repo follows the Search-R1 workflow, where the model interacts with a local retriever server during multi-step reasoning.

Pipeline:
1. Launch a local retriever server.
2. Run model evaluation/inference with retrieval calls enabled.

Implementation reference: Search-R1 README  
https://github.com/PeterGriffinJin/Search-R1/blob/main/README.md

## Repo Layout

- `train_grpo_nqhot.sh`: LLDS/GRPO training script.
- `evaluate.sh`: evaluation / inference-style validation with tool calls.
- `retrieval_launch.sh`: launch local retrieval server.
- `search_r1/search/retrieval_server.py`: retriever API backend (`/retrieve`).

## Installation

Use the same environments/dependencies as Search-R1 (training env + retriever env).  

### Search-r1 environment
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### Retriever environment

```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```



## Quick Start

### 1) Prepare data

Download preprocessed train/test parquet:

```bash
huggingface-cli download \
  --repo-type dataset \
  PeterJinGo/nq_hotpotqa_train \
  --local-dir /path/to/nq_hotpotqa_train
```


### 2) Prepare retriever index + corpus

```bash
python scripts/download.py --save_path /path/to/search_assets
cat /path/to/search_assets/part_* > /path/to/search_assets/e5_Flat.index
gzip -d /path/to/search_assets/wiki-18.jsonl.gz
```

Then edit `retrieval_launch.sh`:

- `file_path=/path/to/search_assets`
- ensure `index_file` and `corpus_file` point to valid files.

### 3) Launch retriever server

```bash
conda activate retriever
bash retrieval_launch.sh
```


### 4) Train LLDS-GRPO

Edit `train_grpo_nqhot.sh` first:

- `DATA_DIR` to your dataset path (`train.parquet`, `test.parquet`)
- `CUDA_VISIBLE_DEVICES` to your available GPUs
- `trainer.default_local_dir` for checkpoint output

Then run:

```bash
conda activate searchr1
bash train_grpo_nqhot.sh
```

### 5) Run evaluation / inference-style validation

Edit `evaluate.sh`:

- `BASE_MODEL` to your checkpoint path (e.g., `.../actor/global_step_xxx`)
- `DATA_DIR` to your validation dataset path

Then run:

```bash
conda activate searchr1
bash evaluate.sh
```


## LLDS Hyperparameters (in `train_grpo_nqhot.sh`)

- `NO_REDUCE_LAMBDA`: LLDS regularization strength.
- `REDUCE_THRES`: activation threshold for reduction penalty.
- `ISCHUNK`: gate granularity (`true` enables chunk/action-level style gate).
- `USE_GSPO`: toggles GSPO objective branch.
- `MASK_ANS`: enable answer mask usage when present in data.
- `MASK_WEIGHT`: answer-region LLDS coefficient.
- `MASK_ADAPTIVE`: adaptive answer mask weighting.
- `NAN_CLIP`: gradient NaN clipping safeguard.

## Usage in MoE

- You should activate router replay to ensure the inference and training route the same expert and LLDS can correctly fix on them.
- LLDS is an algorithm-level solution that addresses instability sources such as OOD feedback and training–inference mismatch introduced by inference optimizations (e.g., quantization and kernel optimization), excluding misaligned expert routing where gradients cannot be propagated to the target expert.

## Notes

- This repo is a focused LLDS/Search-R1-style training workspace, so path values in scripts are examples and should be updated to your environment.
- Tool calls require the retriever server to be running before training/evaluation.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{deng2026group,
  title={On Group Relative Policy Optimization Collapse in Agent Search: The Lazy Likelihood-Displacement},
  author={Deng, Wenlong and Li, Yushu and Gong, Boying and Ren, Yi and Thrampoulidis, Christos and Li, Xiaoxiao},
  booktitle={ICML 2026}
}
```
