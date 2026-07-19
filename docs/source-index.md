# Reviewed source index

**Review date:** 2026-07-19

This file records the external sources used to lock the starter bundle. Update
it and the machine-readable locks together. A new upstream revision is an
experiment boundary, not a silent dependency refresh.

## Gemma 4 31B and Transformers

- Checkpoint config, pinned:
  <https://huggingface.co/google/gemma-4-31B/blob/2d418d1b7ed8c04d732c3359e19a11fbc85b6842/config.json>
- Transformers commit, pinned:
  <https://github.com/huggingface/transformers/tree/7ea2320c76117e6742364808a666ef6f2fb40a67>
- Gemma 4 configuration:
  <https://github.com/huggingface/transformers/blob/7ea2320c76117e6742364808a666ef6f2fb40a67/src/transformers/models/gemma4/configuration_gemma4.py>
- Gemma 4 implementation source:
  <https://github.com/huggingface/transformers/blob/7ea2320c76117e6742364808a666ef6f2fb40a67/src/transformers/models/gemma4/modular_gemma4.py>
- Generated implementation used at runtime:
  <https://github.com/huggingface/transformers/blob/7ea2320c76117e6742364808a666ef6f2fb40a67/src/transformers/models/gemma4/modeling_gemma4.py>
- Mask primitives and backend adapters:
  <https://github.com/huggingface/transformers/blob/7ea2320c76117e6742364808a666ef6f2fb40a67/src/transformers/masking_utils.py>

The conclusions derived from these files are documented line-by-line in
`docs/hf-implementation-audit.md` and enforced by the optional oracle tests.

## FlashAttention-4 and CuTe DSL

- FlashAttention commit, pinned:
  <https://github.com/Dao-AILab/flash-attention/tree/77aacb68d194ba9af1010eda5eac3e7c0df8e6f6>
- FA4 package metadata and exact DSL pin:
  <https://github.com/Dao-AILab/flash-attention/blob/77aacb68d194ba9af1010eda5eac3e7c0df8e6f6/flash_attn/cute/pyproject.toml>
- FA4 CuTe package README:
  <https://github.com/Dao-AILab/flash-attention/blob/77aacb68d194ba9af1010eda5eac3e7c0df8e6f6/flash_attn/cute/README.md>
- CUTLASS functionality matrix:
  <https://github.com/NVIDIA/cutlass#current-functionality>

## Environment selection

- CUDA Toolkit archive:
  <https://developer.nvidia.com/cuda-toolkit-archive>
- CUDA 13.3 release notes:
  <https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html>
- CUDA 13.3 Linux installation guide:
  <https://docs.nvidia.com/cuda/archive/13.3.0/cuda-installation-guide-linux/index.html>
- PyTorch 2.13.0 release notes:
  <https://github.com/pytorch/pytorch/releases/tag/v2.13.0>
- Official PyTorch CUDA 13.2 wheel index:
  <https://download.pytorch.org/whl/cu132/torch/>

The bundle selects CUDA 13.3 GA rather than CUDA 13.4 Developer Preview and
selects the official PyTorch 2.13.0 cu132 wheel. This compatibility set still
requires real H100 and B300 compile/execution validation before it becomes a
measured kernel environment.
