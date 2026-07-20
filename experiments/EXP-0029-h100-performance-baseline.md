# EXP-0029: establish the H100 performance ruler

- Date / author: 2026-07-20 / Codex
- Kernel family: integration
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar path and revision: pinned
  `flash_attn/cute/flash_fwd_sm90.py`,
  `flash_attn/cute/flash_bwd_sm90.py`, and project adapters at product source
  `86444608855a2baf9951d9c1bde92167618f3a61`

## Invariant changed

No kernel invariant changes in this experiment. It establishes the first
reproducible H100 performance ruler after EXP-0028 closed the declared
correctness and sanitizer gates. All measurements retain exact Gemma 4
geometry, scale, masks, prepared distinct K/V, output/LSE, and separate
gradient semantics.

## Hypothesis

At the locked smoke workloads, the exact composed global d512 path contributes
more model-weighted attention time than the local d256 path and exposes the
largest optimization opportunity through its duplicated two-slab forward and
six-main-launch backward structure.

## Single change

Capture the ruler only: environment and clock state, BF16 GEMM/HBM
denominators, warm-cache and cold-L2 fwd/bwd/fwd_bwd timing distributions, and
launch/profile evidence. Do not alter a kernel, tolerance, mask, dispatch rule,
or compile key.

## Correctness evidence

- [x] locked contract and optional HF oracle (inherited from final H100 bundle)
- [x] targeted reference matrix including boundary/tail/adversarial case
  (inherited from EXP-0004 through EXP-0028)
- [x] O, LSE, dQ, dK, dV as applicable (inherited)
- [x] invalid/fallback cases (inherited)
- [x] repeated-run/determinism check (inherited within each accepted scope)

Benchmark admission still requires a pre-timing correctness check of every
measured implementation and workload. Inherited correctness is not permission
to substitute a semantically different baseline.

## Synchronization and generated code

- [x] memcheck (inherited for current binaries; no code change)
- [x] synccheck (inherited for current binaries; no code change)
- [x] racecheck (inherited for current binaries; no code change)
- [x] IR/PTX/SASS observation (current retained artifacts)
- [x] registers/spills/SMEM recorded (current retained artifacts)

Any later kernel change starts a new experiment and reruns the applicable
safety and generated-code gates.

## Measurement

- Device: NVIDIA H100 80GB HBM3, 132 SMs, compute capability 9.0
- Software: driver 580.126.09; CUDA toolkit/runtime 12.8; PyTorch
  2.8.0+cu128; CuTe DSL 4.6.0.dev0; QuACK 0.5.3
- Clock/power state: unlocked. The warm roofline sample reported 1920 MHz SM,
  163.49 W, and 46 C. The idle preflight reported P0, 345 MHz SM,
  2619 MHz memory, 72.45 W, and 37 C. No competing compute process was
  present.
- Measured denominators: 780.646 BF16 TFLOP/s and 3.02456 TB/s aggregate
  HBM read+write, for a 258.103 FLOP/byte ridge point
- Hot/cold L2: both, reported separately
- Warmup/repetitions/statistic: 10 warmups, 30 timed CUDA-event repetitions;
  median, p25, p75, and IQR
- Timed implementation: the project FA4 adapters, retaining the exact layer
  geometry, scale, mask, dtype, LSE, and gradient contract. Raw upstream FA4
  rejects `(head_dim, head_dim_v)=(512, 512)` on SM90, so it is not reported
  as a false equivalent baseline.

| hot-L2 case | median ms | IQR ms | TFLOP/s |
|---|---:|---:|---:|
| local S4K fwd | 0.399 | 0.005 | 301.1 |
| local S32K fwd | 2.270 | 0.006 | 476.9 |
| global S8K fwd | 6.231 | 0.157 | 352.9 |
| global S64K fwd | 364.299 | 3.539 | 386.3 |
| local S4K bwd | 1.378 | 0.054 | 218.2 |
| local S32K bwd | 11.637 | 0.012 | 232.5 |
| global S8K bwd | 98.239 | 0.379 | 56.0 |
| global S64K bwd | 6043.744 | 3.456 | 58.2 |
| local S4K fwd+bwd | 1.916 | 0.045 | 219.7 |
| local S32K fwd+bwd | 13.911 | 0.057 | 272.3 |
| global S8K fwd+bwd | 104.533 | 0.647 | 73.6 |
| global S64K fwd+bwd | 6396.478 | 2.788 | 77.0 |

Cold-L2 results are retained beside the hot-L2 records. They differ by less
than 4% and are sometimes faster, consistent with the explicitly unlocked
clock state rather than a cache-residency conclusion.

For a 50-local/10-global model weighting, global attention contributes 75.7%
of forward, 93.4% of backward, and 91.6% of combined attention time for the
S4K/S8K pair. At the S32K/S64K pair, those shares rise to 97.0%, 99.0%, and
98.9% respectively.

The one-repetition Nsight Systems trace is diagnostic only, not a timing
result. Six backward mainloop launches consumed 94.512 ms (91.9% of traced
GPU-kernel time); the two dKV launches total approximately 22.231 ms and the
four dQ launches approximately 72.281 ms. The dQ kernels therefore account for
76.5% of the backward mainloop and roughly 70% of all traced kernel time. Each
mainloop launch used a 384-thread block, 168 registers/thread, and about
218--222 KB dynamic shared memory. NCU hardware counters remain unavailable on
this pod because of `ERR_NVGPUCTRPERM`.

## Decision

ACCEPT

The ruler and bottleneck evidence are retained under
`agent_space/remote-h100-exp0029/`. The measured priority is global d512
backward dQ. EXP-0030 will test one already-supported SM90 tuning knob before
any new TMA/WGMMA synchronization protocol is introduced.

## Record

```bash
python scripts/record_result.py EXP-0029 \
  --kernel h100-performance-ruler --arch sm_90 --decision <accepted|rejected> \
  --hypothesis '<measured bottleneck sentence>' --bench <jsonl>
```
