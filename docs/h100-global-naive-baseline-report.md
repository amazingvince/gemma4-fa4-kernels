# H100 global causal baseline report

## Summary

- Kernel/revision: accepted Gemma 4 FA4 patch
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- Date: 2026-07-20
- Correctness: all compared paths passed the frozen S128 O and separate
  dQ/dK/dV policy
- Main result: FA4 is 1.60x faster than explicitly expanded fused SDPA for
  global S8K forward+backward and 1.63x faster in the reduced S64K screen
- Scope: B1, BF16, 32 Q heads, 4 distinct K/V heads, d512, causal, scale 1.0,
  H100 only

## Comparators

`sdpa` calls PyTorch SDPA with `enable_gqa=True`. On the pinned H100 stack its
S8K forward decomposed into two cuBLAS GEMMs, a separate softmax, and
elementwise kernels. It is a useful naive/decomposition baseline, but not the
strongest comparator.

`sdpa_expanded` explicitly repeats K and V from 4 to 32 heads inside the timed
operation, then calls non-GQA PyTorch SDPA. Autograd reduces repeated-head
gradients back into the original distinct four-head K and V operands. This
route admitted the fused SDPA backend and is the primary performance baseline.

Neither comparator substitutes full-causal attention for Gemma's local
multimodal predicate. The benchmark rejects `sdpa_expanded` for local cases.

## Environment and method

| Item | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, compute capability 9.0 |
| Driver / CUDA toolkit | 580.126.09 / 12.8.93 |
| PyTorch | 2.8.0+cu128 |
| CuTe DSL / quack | 4.6.0.dev0 / 0.5.3 |
| Clocks | unlocked and labeled |
| Timing | CUDA events, median and p25/p75 IQR |
| S8K | 10 warmups, 30 timed repetitions, hot and cold L2 |
| S64K | 2 warmups, 5 timed repetitions, hot L2 directional screen |

The expanded tensors and, for backward, their anticipated gradients are
included in the benchmark memory preflight. Cold mode thrashes 200 MiB between
timed operations. Compilation and first-call overhead are excluded.

## Results

| Workload | FA4 median/IQR | expanded SDPA median/IQR | FA4 speedup |
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
backward, and 320.300 ms combined. The profile attributed 33.6% and 32.3% of
GPU time to its two GEMMs and 11.9% to softmax.

## Interpretation

The accepted FA4 composition is already faster than the two semantically
equivalent PyTorch baselines tested here. Backward remains the dominant project
cost and has the smallest margin, so dQ is still the correct tuning target.
The claim is deliberately limited to these shapes, modes, software versions,
and this H100. The S64K screen needs a full 10/30 confirmation only if a future
candidate reaches that gate.

## Reproduction

```bash
bash scripts/remote/check.sh h100
bash scripts/remote/run.sh h100 python scripts/probe_h100_global_baselines.py \
  --json agent_space/remote-h100-exp0034/s128-admission.jsonl
bash scripts/remote/run.sh h100 python benchmarks/bench_attention.py \
  --ladder smoke --only global_s8k --impl sdpa_expanded --mode fwd_bwd \
  --l2 hot --warmup 10 --reps 30
```

Raw evidence is retained under `agent_space/remote-h100-exp0034/`.
