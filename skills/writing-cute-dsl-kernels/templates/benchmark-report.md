# CuTe DSL Kernel Benchmark Report

## Summary

- Kernel/revision:
- Date:
- Correctness status:
- Sanitizer status:
- Main result:
- Scope of claim:

## Environment

| Item | Value |
|---|---|
| GPU / compute capability | |
| Driver | |
| CUDA toolkit | |
| CUTLASS / `nvidia-cutlass-dsl` | |
| Python / framework | |
| Clock/power policy | |
| OS / CPU | |

## Kernel Configuration

- Architecture path:
- Tile / MMA atom:
- Stages:
- Cluster / CTA group:
- Threads / warps:
- SMEM:
- Registers:
- TMEM:
- Copy mechanisms:
- Epilogue:
- Specialization/cache key:

## Workloads

| ID | Shape(s) | Dtypes | Layout/strides | Alignment | Mask/epilogue | Weight in target distribution |
|---|---|---|---|---|---|---|
| | | | | | | |

## Method

- Compilation included:
- Warmup iterations:
- Timed iterations:
- Timing primitive:
- Synchronization boundaries:
- Candidate ordering:
- Statistic:
- Outlier handling:
- Correctness check cadence:

## Results

| Workload | Candidate | Median | P10/P90 or min/max | Baseline | Speedup | GB/s or TFLOP/s | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Profiling Evidence

- Expected instruction present:
- GMEM transaction efficiency:
- SMEM bank-conflict evidence:
- Occupancy/residency:
- Register spills/local memory:
- Dominant wait/stall:
- Pipeline overlap:
- Epilogue share:
- Trace/IR/SASS artifacts retained at:

## Autotuning

- Search space:
- Invalid configurations filtered:
- Number compiled:
- Cache hit rate:
- Winner selection rule:
- Holdout shapes:
- Tuning-result cache key:

## Interpretation

- Bottleneck before:
- Bottleneck after:
- Why the change helps:
- Where it loses:
- Generalization limits:
- Follow-up experiment:

## Reproducibility

- Command:
- Commit/tag:
- Configuration file:
- Required environment variables:
- Raw result artifact:
