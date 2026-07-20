# EXP-0034: H100 semantically equivalent naive global baselines

- Date / author: 2026-07-20 / Codex
- Kernel family: benchmark/integration
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`

## Invariant changed

No kernel invariant. Add explicit baseline admission and distinguish PyTorch
SDPA's automatic GQA route from a composite that explicitly repeats K/V heads
before non-GQA SDPA. Both retain causal masking, scale 1.0, BF16, original
32Q/4KV inputs, distinct K/V, and gradients reduced to separate dQ/dK/dV.

## Hypothesis

The accepted project FA4 composition is faster than both PyTorch automatic-GQA
SDPA and the stronger explicit-KV-expansion SDPA composite at global S8K
forward, backward, and combined timing, after each passes the frozen S128
reference policy.

## Single change

Benchmark/probe support only. No FA4 kernel, adapter routing, tolerance, mask,
or compile key changes.

## Evidence

- [x] strict H100 environment and exact accepted patch
- [x] S128 O and separate dQ/dK/dV reference admission for all implementations
- [x] automatic-GQA SDPA S8K profile identifies separate GEMM/softmax kernels
- [x] hot/cold S8K distributions, 10 warmups and 30 CUDA-event repetitions
- [x] S64K reduced feasibility screen, 2 warmups and 5 repetitions

The admission probe used the frozen project reference policy at S128. All
three implementations passed O and separate upstream-relative dQ/dK/dV
checks. Maximum absolute O errors were 0.03125 for FA4, 0 for automatic-GQA
SDPA, and 0.015625 for expanded SDPA. The raw admission rows retain the
separate gradient errors.

The automatic-GQA SDPA forward was not fused in this environment: Nsight
Systems attributed 65.9% of captured GPU time to two cuBLAS GEMMs and 11.9%
to a separate softmax kernel, with the remainder dominated by elementwise
mask/copy work. The explicit expansion baseline was therefore added as the
stronger practical comparator; its repeat-interleave work remains inside the
timed operation.

| case | FA4 median/IQR | expanded SDPA median/IQR | FA4 speedup |
|---|---:|---:|---:|
| S8K fwd hot | 6.256 / 0.123 ms | 20.018 / 0.092 ms | 3.20x |
| S8K bwd hot | 96.928 / 1.308 ms | 145.203 / 0.770 ms | 1.50x |
| S8K fwd_bwd hot | 103.098 / 1.224 ms | 164.986 / 0.767 ms | 1.60x |
| S8K fwd cold | 6.155 / 0.117 ms | 20.105 / 0.136 ms | 3.27x |
| S8K bwd cold | 96.839 / 1.101 ms | 145.436 / 0.508 ms | 1.50x |
| S8K fwd_bwd cold | 103.333 / 1.209 ms | 165.618 / 0.733 ms | 1.60x |
| S64K fwd hot screen | 357.017 / 13.804 ms | 2018.190 / 0.667 ms | 5.65x |
| S64K bwd hot screen | 6039.759 / 23.567 ms | 8442.541 / 46.829 ms | 1.40x |
| S64K fwd_bwd hot screen | 6400.255 / 2.418 ms | 10442.245 / 29.685 ms | 1.63x |

Automatic-GQA SDPA at S8K hot measured 130.594 ms forward, 189.684 ms
backward, and 320.300 ms combined. It is retained as a decomposition baseline,
not presented as PyTorch's fastest possible attention route.

Raw CUDA-event rows and the Nsight Systems kernel summary are under
`agent_space/remote-h100-exp0034/`. Clocks were not locked, so every table and
claim is labeled unlocked-clock. The S64K rows are directional screens, not
full 30-repetition confirmation runs.

## Decision

BASELINE

The hypothesis passes for the measured exact global causal workloads. The
accepted project FA4 composition is faster than both automatic-GQA SDPA and
the stronger explicitly expanded fused-SDPA composite in every measured mode.
This does not establish a local multimodal comparison, a B300 result, or a
speedup over every attention implementation.
