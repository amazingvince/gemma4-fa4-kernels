# Experimental plan

## North star

Model-weighted attention time for all 50 local and 10 global layers, forward
plus backward, at 32K and 128K BF16 on one H100 and one B300, compared with the
best semantically equivalent composite baseline.

A two-times speedup at 128K is a research target, not a promised outcome.

## M0 — establish the ruler

On each host:

1. capture `scripts/check_env.py --strict` output;
2. lock sustainable clocks or flag the run as unlocked;
3. record GEMM and HBM-copy denominators with `benchmarks/roofline.py`;
4. compile the current upstream FA4 matrix with persistent cache enabled;
5. capture separate fwd, bwd, and fwd+bwd timings;
6. profile first-call JIT separately from warm calls;
7. record exact unsupported cases rather than substituting different masks.

## Benchmark ladder

Smoke:

```text
local:  B=1, 32Q/16KV, d256, W1024, S={4K,32K}
global: B=1, 32Q/4KV,  d512, causal, S={8K,64K}
```

Run each in `fwd`, `bwd`, and `fwd_bwd`. Add TP-local GQA ratios 1/2/4/8 to the
full ladder. The full ladder spans 1K–256K, varlen document mixes, text and
vision masks, deterministic/nondeterministic modes, and BF16/FP16 coverage.

Memory preflight may skip infeasible entries but records estimates and free
memory. A skipped full shape is not silently replaced by a smaller head count.

## Semantic baseline rule

A result is comparable only if these match:

- scale 1.0;
- local/global mask including vision behavior;
- Q/K/V and GQA geometry;
- dtype and accumulation contract;
- fwd versus bwd versus combined timing region;
- LSE/output/gradient contract.

For example, plain full-causal SDPA is not a local-window baseline.

## Measurement hygiene

- fixed clocks, temperature, power, and a canary kernel;
- CUDA events for device time; CUDA graphs only as a separate stacked result;
- hot- and cold-L2 modes reported explicitly;
- direct GPU input generation and dynamically sized cache-thrash buffers;
- median, p25, p75, and IQR after warmup;
- cold compile and warm cache reported separately;
- append-only accepted and rejected experiment records;
- no chasing changes inside the measured noise floor.

Default acceptance requires at least 1.5% improvement beyond noise, no
correctness or sanitizer regression, no new spills, no cache-key churn, and no
more than 3% loss on another important regime.

## Milestones

### H100-M1: H100 correctness path

Status 2026-07-19: the pinned environment, local d256 forward/autograd
backward, and exact two-launch global d512 forward composition passed their
declared gates. EXP-0003's fixed elementwise dQ/dK envelope remains rejected;
EXP-0004 diagnosed the BF16 oracle mismatch and accepted the unchanged local
backward under a predeclared upstream-relative policy across the boundary
matrix, streams, repeats, sanitizers, and generated-code inspection. Global
d512 backward is the next ordered gate; multimodal masking and benchmarks have
not run. See `docs/status.md` and EXP-0001 through EXP-0004.

- pinned FA4 CuTe SM90 build on CUDA 12.x;
- local d256 forward and a scoped local d256 backward configuration
  (**complete for the declared M1 text envelope**);
- global d512 slabbed forward (**complete**) and backward (**next gate**);
- exact scale, O/LSE, separate dQ/dK/dV, GQA, boundaries, and multimodal mask;
- no performance tuning until every H100 correctness and sanitizer gate passes.

### B300-M1: B300 local correctness (deferred in the H100 session)

- prepared Q/K/V, BF16, fixed length, text mask;
- exact scale/window/O/LSE;
- one-CTA and two-CTA variants compile and pass sanitizer;
- add GQA-2 and boundary masks.

### M2: Global d=512 forward

- H100 D-split schedule;
- B300 one-CTA/two-CTA schedule comparison;
- GQA 8 plus TP-local ratios;
- O and LSE green through adversarial boundaries.

### M3: Global backward

- preprocess;
- owner-computes dQ;
- owner-computes dK and dV;
- no required whole-layer FP32 accumulation buffer in the target path;
- deterministic ownership mode.

### M4: Local multimodal, varlen, and integration

- vision-block tile classification;
- packed document boundaries;
- explicit verification that cross-layer KV reuse is disabled;
- per-layer HF dispatch and context-parallel offsets.

### M5: model-specific fusion and precision exploration

- shared-source K/V preparation fusion;
- exact preparation backward to dZ;
- FP8/FP4 only with numerical and training-convergence evidence.

## Profiler decision tree

1. Nsight Systems: launches, JIT, host gaps, graphs.
2. Nsight Compute: SOL, tensor pipeline, SMEM/L2/HBM, barriers, occupancy.
3. PTX/SASS and ptxas: target instructions, register counts, spills.
4. Compute Sanitizer: memcheck, synccheck, racecheck.
5. Error heatmaps: position, tile boundaries, vision/window edges, max drift.

Use the full CuTe skill's design brief, review checklist, and benchmark report
templates for every structural kernel change.
