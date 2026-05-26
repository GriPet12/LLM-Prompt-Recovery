# LLM Prompt Recovery Production Pipeline

This repository contains a robust baseline notebook pipeline designed for the Kaggle **LLM - Prompt Recovery** competition. The solution uses a 4-bit quantized **Gemma 2B IT** model optimized for execution on free-tier T4 GPUs.

## Technical Architecture Overview
The system follows a sequential data engineering and inference layout:

### Key Engineering Features:
* **Quantization Matrix:** Leverages `bitsandbytes` to parse model weights in NormalFloat 4 (`nf4`) layout and compute activations with `torch.float16`. This workflow ensures strict operational stability under the 16GB VRAM ceiling.
* **Hybrid Prompt Construction:** Concatenates a highly reliable baseline instructions pattern (`MEAN_PROMPT`) with a dynamic context variance vector evaluated via greedy text generation (`temperature=0.1`).
