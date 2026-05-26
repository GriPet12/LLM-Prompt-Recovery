# LLM Prompt Recovery Production Pipeline

This repository contains a robust baseline notebook pipeline designed for the Kaggle **LLM - Prompt Recovery** competition. The solution uses a 4-bit quantized **Gemma 2B IT** model optimized for execution on free-tier T4 GPUs.

## Technical Architecture Overview
The system follows a sequential data engineering and inference layout:

### Key Engineering Features:
* **Quantization Matrix:** Leverages `bitsandbytes` to parse model weights in NormalFloat 4 (`nf4`) layout and compute activations with `torch.float16`. This workflow ensures strict operational stability under the 16GB VRAM ceiling.
* **Hybrid Prompt Construction:** Concatenates a highly reliable baseline instructions pattern (`MEAN_PROMPT`) with a dynamic context variance vector evaluated via greedy text generation (`temperature=0.1`).

## Deployment & Setup Guide

### 1. Credentials Setup
* **Kaggle API:** Generate your unique `kaggle.json` credential layer via account settings and upload it into your execution directory.
* **Hugging Face Token:** Use an active User Access Token with Read access privileges to authenticate your download request for gated repository configurations (`google/gemma-2b-it`).

### 2. Operational Pipeline Stages
The implementation workspace in the notebook consists of the following progressive operational layers:
1. **Block 1:** Installs external runtime dependencies and initializes Kaggle workspace structures.
2. **Block 2:** Compiles the quantized parameters and maps model parameters across the CUDA architecture.
3. **Block 3:** Automatically reads input data assets without relying on standard mock matrices.
4. **Block 4:** Executes iterative inference configurations using rigid formatting system instructions.
5. **Block 5:** Saves execution outputs onto a clean, evaluation-compliant `submission.csv` tracking matrix.
