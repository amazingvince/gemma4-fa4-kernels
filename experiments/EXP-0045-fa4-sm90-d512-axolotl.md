# EXP-0045: native FA4 d512 Axolotl integration

- Date / author: 2026-07-21 / Codex
- Kernel family: integration
- Architecture: sm_90
- Upstream FA4 revision: `2409214a03797b168f648ea30df1adbc09ce658a`
- Candidate FA4 revision: `4a207fc3e94ab36c2dde12e9cb5822bd048037a8`
- CuTe DSL / CUDA / PyTorch: CuTe DSL `4.6.0.dev0`, CUDA `12.8`,
  PyTorch `2.11.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar: `google/gemma-4-12B-it` at
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`

## Invariant changed

The Axolotl boundary no longer pads Gemma 4 12B prepared tensors into the
31B-only `32Q/4KV` geometry. Text-only local layers call public FA4 at native
`16Q/8KV/d256`; global layers call public FA4 at native `16Q/1KV/d512`.
Packed batches derive public varlen cumulative arrays from position resets and
never repeat or pad KV heads.

## Hypothesis

Generic public SM90 d512 support permits all 48 Gemma 4 12B layers to train at
native geometry, with prepared-QKV error no worse than a BF16 SDPA baseline,
while remaining within 3% of Axolotl's hybrid median step time and without a
peak-memory increase.

## Single change

Register a separate model-side native-shape attention backend that validates
the exact prepared Q/K/V geometry, calls `flash_attn_func` or
`flash_attn_varlen_func`, and records per-layer route, storage-alias, physical
length, and packed-segment evidence. The FlashAttention fork remains
model-agnostic.

## Correctness evidence

- [x] Six eight-step 12B LoRA jobs complete at exact physical S512/S1024/S2048.
- [x] All runs cover exactly 40 local and 8 global layers.
- [x] Global tensors are native `16Q/1KV/d512`; no 32Q/4KV adapter is active.
- [x] K and V have distinct storage in every layer.
- [x] Packed jobs use only public varlen routes and record 4, 8, or 16 isolated
  document segments per prepared tensor.
- [x] Losses and independent LoRA gradients are finite.
- [x] Exact prepared local/global Q/K/V pass an FP32 attention oracle using BF16
  SDPA as the upstream-style low-precision error baseline.
- [ ] The older whole-model gradient sketch is not stable across the hybrid and
  native BF16 routes (`cosine=0.18163`, relative L2 `1.00531`); prepared-QKV
  checks remain the primary oracle.

The S1024 candidate maximum loss delta from hybrid is `0.0573778`, within the
existing hybrid-versus-SDPA maximum BF16 loss envelope of `0.0734282` on the
same replay.

## Synchronization and generated code

The underlying fork matrix passed memcheck, synccheck, and racecheck with zero
device findings. Generated d512 objects report `REG:168 STACK:0 LOCAL:0`, and
show the intended TMA/WGMMA and barrier instructions. Dynamic shared memory is
compile-gated against the SM90 227 KiB per-CTA limit. Nsight Compute counters
remain unavailable on the RunPod host with `ERR_NVGPUCTRPERM`.

The final committed fork test rerun reports `96 passed, 1 skipped`; the opt-in
Q1/K262144 smoke reports `1 passed, 96 deselected`.

## Measurement

- Clock/power state: RunPod H100 80 GB; no new clock lock claimed.
- Hot/cold L2: whole-model fresh-process steps; kernel microbenchmarks record
  hot/cold modes separately.
- Warmup/repetitions/statistic: three warmups, five timed steps, median.
- Semantically equivalent baseline: Axolotl hybrid attention at native 12B
  model geometry.

| 12B case | candidate median ms | peak allocated bytes | route |
|---|---:|---:|---|
| S512 unpacked | 273.543 | 24,114,589,696 | fixed |
| S512 packed | 310.360 | 24,114,589,696 | varlen, 4 documents |
| S1024 unpacked | 268.708 | 24,114,589,696 | fixed |
| S1024 packed | 292.780 | 24,114,589,696 | varlen, 8 documents |
| S2048 unpacked | 387.352 | 24,114,589,696 | fixed |
| S2048 packed | 381.259 | 24,114,589,696 | varlen, 16 documents |

At S1024 unpacked, the fresh hybrid baseline is `253.072 ms` and the candidate
is `268.708 ms`, a 6.18% regression. Peak allocated and reserved memory are
identical (`24,114,589,696` and `40,045,117,440` bytes).

## Decision

**REFINE.** Native-shape correctness, packed isolation, and memory gates pass,
but the whole-model 3% performance gate fails. Keep the fork PR as a draft and
profile the integration overhead before claiming an Axolotl speedup.

Gemma 4 31B QLoRA remains unrun: the pod has no cached
`google/gemma-4-31B` checkpoint and no Hugging Face authentication for that
gated repository. Synthetic/native `32Q/4KV/d512` kernel tests and benchmarks
pass, but they are not a whole-model claim.

## Reproduction outline

All GPU commands ran through the exclusive `scripts/remote/gpu-run.sh h100`
lease. Each candidate run installed the fork's `flash-attn-4`, selected backend
`native`, set physical `GEMMA4_FA4_SEQUENCE_LEN`, generated an exact-length
dataset, and invoked:

```bash
axolotl train configs/axolotl/gemma4-12b-smoke.yaml --launcher python
```

Packed runs additionally set `GEMMA4_FA4_SAMPLE_PACKING=true` and use 128-token
records so the physical S512/S1024/S2048 batches contain 4/8/16 documents.
