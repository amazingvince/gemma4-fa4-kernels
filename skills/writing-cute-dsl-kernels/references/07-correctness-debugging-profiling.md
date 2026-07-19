# Correctness, Debugging, Profiling, and Tuning

## Correctness Before Performance

Establish a trusted reference before optimizing. Compare complete outputs and any auxiliary results. A benchmark candidate is invalid until it passes the declared correctness matrix.

## Correctness Matrix

Include dimensions from each relevant category:

| Category | Cases |
|---|---|
| Degenerate | zero extents where supported, one element, one row/column, K=0 or K=1 |
| Tile boundaries | `tile-1`, `tile`, `tile+1`, multiple tiles, final partial tile |
| Shape character | tall-skinny, short-wide, square, tiny, large, batch/group variation |
| Layout | contiguous, supported transpose/major modes, supported noncontiguous strides |
| Alignment | fully aligned hot path; minimally aligned supported path; misaligned rejection/fallback |
| Values | zeros, ones, alternating signs, unique coordinate code, random, large/small magnitude, NaN/Inf policy |
| Dtype | every advertised input/accumulator/output combination |
| Masking | no mask, partial, full row, causal/local/ragged where relevant |
| Concurrency | repeated launches, non-default stream, multiple CTAs/clusters, persistent multiple work items |
| Architecture | every advertised SM or explicit rejection/fallback |

For grouped/persistent kernels, vary each work item independently.

## Numeric Comparison

Define tolerance from the mathematical and accumulation policy, not convenience.

Record:

- reference dtype and accumulation;
- absolute and relative tolerance;
- ULP policy if used;
- expected nonassociativity;
- NaN equivalence policy;
- saturation/rounding for low precision;
- distribution of error, not just maximum.

For reductions and attention, compare invariant quantities such as row sums, running max/sum behavior, or normalization.

## Test Data That Exposes Layout Bugs

Use values encoding coordinates, for example a combination of row, column, batch, and K index. Repeated random values can miss transposes, duplicates, and aliasing.

For scale factors, make each block’s scale unique. For masked operations, give masked elements extreme values to prove they do not leak.

## Progressive Correctness Ladder

1. tiny scalar/synchronous kernel;
2. one CTA;
3. multiple CTAs;
4. full tails/predication;
5. vectorization;
6. SMEM swizzle;
7. asynchronous copy;
8. multistage pipeline;
9. architecture MMA;
10. fused epilogue;
11. persistent/grouped scheduling;
12. autotuned variants.

When a step fails, revert only that step and inspect its invariants.

## Compile-Time and Meta-Stage Debugging

Use Python `print()` to inspect:

- layouts and hierarchical shapes;
- static/dynamic values;
- selected atom/tile/stage policy;
- JIT arguments and cache key inputs;
- launch grid/block/cluster and SMEM size.

Enable the documented debug environment option for full stack traces and line information. Treat compile warnings as actionable.

Common compile failures:

| Symptom | Likely cause |
|---|---|
| cannot lower Python operation | unsupported Python subset or object mutation |
| branch/loop type mismatch | loop-carried value changes type/layout |
| static assertion in copy | partition width/alignment incompatible |
| MMA construction failure | dtype/major/tile/atom mismatch |
| layout/type explosion | too much static hierarchy or unrolling |
| launch/argument validation | wrong layout, device, pointer, stream, or shared-memory contract |

Reduce the kernel to the smallest layout/copy/atom construction that reproduces the error.

## Device Runtime Debugging

Use bounded `cute.printf()` for a single CTA/thread to observe runtime coordinates, state indices, values, and predicates. Gate prints with static or runtime conditions to avoid changing timing and producing unusable output.

Run NVIDIA compute-sanitizer tools appropriate to:

- out-of-bounds/misaligned memory;
- data races;
- synchronization hazards;
- uninitialized memory.

A sanitizer-clean run on one friendly shape is insufficient; include partial tiles and multiple CTAs.

## Artifact Inspection

Preserve or print compilation artifacts through documented JIT options/environment variables:

- MLIR/DSL IR;
- LLVM IR;
- PTX;
- CUBIN/SASS.

Inspect artifacts when:

- expected MMA/TMA/cp.async instruction is absent;
- scalar loads appear instead of vectors;
- spills/local memory appear;
- barriers/fences are unexpectedly frequent;
- loop unrolling/code size is excessive;
- architecture target is wrong;
- performance changes after a package upgrade.

Do not infer machine instructions solely from high-level API calls.

## Naming and Regions

Assign name prefixes to major functions/regions where supported. Use phase names that match the design:

- `prologue`;
- `mainloop_load`;
- `consumer_wait`;
- `mma`;
- `epilogue`;
- `tma_store`;
- `drain`.

This makes IR and profiler traces easier to connect to source.

## IKET and Event Tracing

Where supported on Hopper/newer targets, IKET can record in-kernel events for visualization. It is experimental and must not become a correctness dependency.

Instrument natural waits and phases, not every instruction. Check:

- whether producer/consumer overlap exists;
- which wait dominates;
- whether pipeline slots are idle;
- whether epilogue begins only after a long drain;
- whether work is imbalanced across roles/CTAs.

Remove or disable instrumentation for final timing unless measuring its overhead explicitly.

## Performance Measurement Contract

A benchmark report must state:

- GPU model and SM;
- driver/CUDA/package/framework versions;
- clock/power state or whether clocks were controlled;
- problem shapes, dtypes, layouts, and batch/group distribution;
- compilation excluded or included;
- warmup count;
- timing primitive (CUDA events preferred for device time);
- synchronization boundaries;
- repetitions and reported statistic;
- baseline and its configuration;
- correctness status;
- variability/outliers;
- achieved bandwidth or FLOP/s with formula.

Use medians or robust statistics and show a distribution/range when variance matters. Never use Python wall-clock time around asynchronous launches without synchronization.

## Roofline-Oriented Diagnosis

Estimate:

- bytes moved from/to GMEM;
- arithmetic operations;
- arithmetic intensity;
- expected bottleneck;
- transaction efficiency;
- occupancy/resource limits.

Then use profiling counters to decide whether to tune:

- memory coalescing/vectorization;
- SMEM bank behavior;
- pipeline latency;
- instruction issue;
- occupancy;
- scheduler/load balance;
- epilogue.

Do not tune tile dimensions randomly before identifying the limiting resource.

## Autotuning

### Search-space design

Use a bounded, valid search space:

- CTA/MMA tile;
- warp/warpgroup arrangement;
- stage count;
- cluster shape;
- copy vector width;
- epilogue/store strategy;
- one-CTA/two-CTA mode where supported;
- persistent scheduling policy.

Filter configurations by architecture, dtype, divisibility/alignment, SMEM/TMEM, threads, and other resource constraints before compile.

### Compile and benchmark

Compile each unique static configuration once using the executor cache. Warm up. Use repeated CUDA-event timing with synchronization at the measurement boundary. Randomize or interleave candidate order when thermal/clock drift matters.

### Cache tuning results

Key results by:

- GPU/SM;
- package/revision;
- dtype/layout/alignment class;
- operation/epilogue;
- shape bucket;
- relevant environment.

Validate a cached winner against correctness and resource constraints before reuse.

### Avoid overfitting

Use a representative workload distribution. A winner for one exact shape may lose badly on tails or small sizes. Prefer a small policy table over per-shape compilation explosion.

## Production-Exemplar Diagnostics: Copy the Reason, Not the Trick

Production repositories contain valuable low-level patterns, but each must remain explicitly scoped:

- **Named barrier numbering:** FlashAttention-4 starts its named-barrier enums at 1 because barrier 0 is reserved for `sync_threads()` in that implementation. Verify the target CuTe/architecture path and barrier users before adopting the same numbering.
- **Power-of-two pipeline stages:** its `PipelineStateSimple` notes that power-of-two stage counts can lower index/phase division to bit operations. This is a micro-optimization after correctness and resource tuning, not a universal stage-count rule.
- **Frozen dataclass reclassification:** its pipeline wrappers use `object.__setattr__(obj, "__class__", ...)` to extend frozen factory-created objects. Treat this as a version-sensitive compatibility technique; prefer supported public extension points when available.
- **Artifact hooks:** repository variables for custom `ptxas`, retained PTX, CUBIN/SASS dumps, or source-line mapping are excellent debugging aids but are not interchangeable with official CuTe DSL options.
- **Sanitizer interpretation:** asynchronous TMA paths can produce tool-specific diagnostics. Investigate against a minimal reproducer and generated code; do not dismiss a report merely because a large project labels a similar pattern a false positive.

Record the source file/revision beside every borrowed low-level pattern and add a regression test that would fail if its hidden assumption changes.

## Debugging Decision Tree

### Wrong answer, deterministic

1. compare coordinate-coded data;
2. inspect tiling and partitions;
3. disable vectorization;
4. disable swizzle;
5. replace async with synchronous;
6. test one K tile;
7. inspect predication/neutral fill;
8. verify epilogue mapping and conversion.

### Wrong answer, nondeterministic

1. run race/sync sanitizer;
2. force waits and block barriers;
3. reduce to one stage;
4. check stage release and group drain;
5. check persistent state reset;
6. check output writer uniqueness.

### Hang

1. verify launch and CTA-group/cluster shape;
2. inspect barrier participant counts;
3. test zero/one iteration;
4. trace pipeline states;
5. verify all roles reach shutdown;
6. reduce stages/roles.

### Correct but slow

1. separate compilation/launch/device time;
2. confirm target instructions in SASS;
3. inspect resource usage/spills;
4. profile memory transactions and bank conflicts;
5. inspect pipeline overlap/waits;
6. compare epilogue cost;
7. tune a constrained search space.

## Release Gate

- [ ] All correctness matrix cases pass.
- [ ] Sanitizer checks pass on representative tail and concurrency cases.
- [ ] Invalid contracts reject or fall back cleanly.
- [ ] Generated code targets the intended architecture.
- [ ] No unbounded JIT specialization.
- [ ] Benchmark methodology is reproducible.
- [ ] Performance is reported only for measured cases.
- [ ] Version and source revision are pinned.
