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

The H100 profile applies the focused combined patch
`patches/flash-attention/0002-sm90-gemma4-d512-forward-backward.patch` to that
exact base revision. Its SHA256 is
`3c5a40718f8c08bf2e0b95c38f3a967a09b29a450b321732ef410966a6ac546b`.
The patch retains the SM90 asymmetric d512-QK/d256-V forward specialization
and adds the EXP-0006 split global backward path. EXP-0035 retains one dKV
launch per V256 slab while replacing the four slab-specific dQ launches with
two exact full-D D256-streaming variants, with FP32 accumulation.
EXP-0012 adds resource preflight and validates the unchanged runtime scheduler
through fixed and exactly composed per-segment S/K2048. EXP-0013 adds the
native packed THD/cu-seqlens ABI for nonempty segments through K2048. Its
runtime totals and cumulative values reuse three bounded scheduler classes;
fixed application-key suffixes and fixed main-object contents remain unchanged.
EXP-0014 changes only native packed admission and the matching upstream
assertion through K262144 under signed-INT32 and guarded-HBM preflight. Fixed
BSHD and the exact composer remain capped at S/K2048; K>2048 budget rejection
propagates before forward. Long runtime lengths reuse the same three scheduler
classes and generated main-object contents. EXP-0015 changes only mixed packed
admission and the upstream host assertion: per-segment
`0 <= Sq <= Sk <= 262144` is accepted when aggregate Q/K totals and exact
maxima remain positive. Empty-query segments schedule no main work, including
query-empty/key-nonempty K/V slices; plateau replays retain the existing
scheduler/application classes and main-object bytes/resources. Each changed
source fingerprint intentionally creates a fresh cold-cache namespace. The
exact composer is retained only as the guarded native-HBM-budget fallback when
every active-query K segment is at most 2048. The base revision, patch path,
and hash are machine-locked in `upstream.lock.json`; the BSD-3-Clause license
is retained under `third_party/flash-attention/LICENSE`. The project adapter
remains the semantic guard that admits only the locked global-causal text
contract, and rejects all-empty physical workloads before backend launch.

The pinned FA4 package declares `quack-kernels>=0.5.3` and imports its SM90
layout/copy helpers at runtime. The successful H100 environment resolved
0.5.3, so both target policies pin that exact version rather than allowing a
fresh bootstrap to float.

## Environment selection

- CUDA Toolkit archive:
  <https://developer.nvidia.com/cuda-toolkit-archive>
- CUDA 12.8 release notes and driver compatibility (H100):
  <https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/>
- Official PyTorch 2.8.0 cu128 installation command (H100):
  <https://pytorch.org/get-started/previous-versions/>
- CUDA 13.3 release notes:
  <https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html>
- CUDA 13.3 Linux installation guide:
  <https://docs.nvidia.com/cuda/archive/13.3.0/cuda-installation-guide-linux/index.html>
- PyTorch 2.13.0 release notes:
  <https://github.com/pytorch/pytorch/releases/tag/v2.13.0>
- Official PyTorch CUDA 13.2 wheel index:
  <https://download.pytorch.org/whl/cu132/torch/>

The H100 policy follows the pinned FA4 CUDA-12 `[dev]` path with CUDA 12.8 and
the official PyTorch 2.8.0 cu128 wheel. The B300 policy retains CUDA 13.3 and
the official PyTorch 2.13.0 cu132 wheel with the `cu13` extra. Each target must
pass its own compile/execution validation before producing measured results.
