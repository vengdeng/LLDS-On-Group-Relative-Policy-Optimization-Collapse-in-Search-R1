

<h1 align="center">On GRPO Collapse in Search-R1: The Lazy Likelihood-Displacement Death Spiral</h1>


<p align="center">
📃 <a href="https://arxiv.org/abs/2512.04220" target="_blank">Paper</a> </a> ｜🤗 <a href="https://huggingface.co/SEGAgentRL" target="_blank">LLDS-Huggingface</a> ｜🐙 <a href="https://github.com/vengdeng/LLDS-On-Group-Relative-Policy-Optimization-Collapse-in-Search-R1" target="_blank">GitHub</a>
</p>


## ⚡ Introduction

**LLDS** is a lightweight likelihood-preserving regularization designed to stabilize **tool-integrated reinforcement learning** (e.g., GRPO / Search-R1 style training).
It prevents training collapse by regularizing **only when** the likelihood of (good) action decreases, and **only on** the tokens responsible for the decrease.

- We identify **Lazy Likelihood Displacement (LLD)** as a key mechanism behind collapse in tool-integrated GRPO training.
- LLDS activates **selectively**: it penalizes likelihood reduction on a *preserving set* (e.g., non-negative-advantage actions).
- We release our **LLDS-tuned Qwen2.5-3B-BASE** checkpoint for searchs-integrated reasoning and QA.
- **A refer to action-level gate**, R refer to response-level gate, **action (A) level gate achieve the best performance**.

## 🔍 LLDS Tool-Integrated Search Training (Code Coming soon) 

## 🔍 Tool-Integrated Search Inference (Search-R1 style)

We support tool-integrated inference using the same workflow as **[Search-R1](https://github.com/PeterGriffinJin/Search-R1)**, where the LLM interacts with a local retrieval server for multi-step reasoning.

The pipeline consists of two parts:

1. Launch a local retriever server
2. Run inference with the LLDS model

---

### 1️⃣ Launch the local retrieval server

Search-R1 recommends running the retriever in a separate environment.

```bash
conda activate retriever
bash retrieval_launch.sh
```
### 2️⃣ Run inference with LLDS-A-Qwen2.5-3B-BASE-MA


```bash
conda activate searchr1
python infer.py

MODEL_NAME = "<YOUR_ORG>/<YOUR_MODEL_NAME>" # e.g. my-org/LLDS-A-Qwen2.5-3B-BASE-MA

question = "Your question here"
```

## 📖 Citation
```
@article{deng2025grpo,
  title={On GRPO Collapse in Search-R1: The Lazy Likelihood-Displacement Death Spiral},
  author={Deng, Wenlong and Li, Yushu and Gong, Boying and Ren, Yi and Thrampoulidis, Christos and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2512.04220},
  year={2025}
}
```
