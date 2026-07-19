# Kernel-Family Patterns

## How to Use This Reference

Choose the nearest family, copy its **invariants and development order**, then locate an official example for the target architecture. Do not copy a kernel optimized for a different dtype, shape regime, or SM and merely rename it.

For every family, answer:

- What data is reused, and by whom?
- What is the reduction domain?
- Which memory space holds each live value?
- Which dimensions are parallel versus sequential?
- What are the partial-tile and degenerate cases?
- Is the kernel bandwidth-, latency-, instruction-, or occupancy-bound?

## Pattern A: Elementwise and Fused Pointwise

### Suitable operations

Unary/binary transforms, activation, bias, conversion, clamping, simple broadcasting, and short fusion chains without cross-element dependence.

### Baseline dataflow

`GMEM → per-thread registers → scalar/vector compute → GMEM`

Use a flat logical index only when it preserves coalescing for the actual layout. For strided/broadcast inputs, derive coordinates from a compact output layout and project them into each operand.

### Development order

1. scalar, predicated implementation;
2. validate arbitrary shapes and strides;
3. choose a thread/value layout;
4. add vector loads/stores for an alignment class;
5. specialize only a small number of useful alignment/vector-width classes;
6. fuse adjacent operations while monitoring register pressure.

### Invariants

- each output element has exactly one writer;
- broadcast modes never participate in a write layout;
- vector width respects all input/output alignments and contiguous extents;
- tail lanes are individually masked or handled by a scalar remainder;
- in-place operation is safe under the declared aliasing policy.

### Common failure

A vectorized path checks base-pointer alignment but ignores that a row stride or tail offset breaks alignment. Validate every vector start or dispatch based on stronger layout guarantees.

## Pattern B: Copy, Transpose, and Layout Transform

### Suitable operations

Transpose, permutation, pack/unpack, dtype conversion with reordering, interleave/deinterleave.

### Baseline dataflow

For naturally coalesced source and destination, use a direct TiledCopy. When one side is strided, use an SMEM tile:

`GMEM source → SMEM layout → synchronization → GMEM destination`

The SMEM layout changes the access order and may be padded/swizzled to avoid bank conflicts.

### Development order

1. define logical source and destination coordinate mapping;
2. build source/destination TiledCopy partitions;
3. predicate source and destination independently;
4. add SMEM only when direct access is inefficient or impossible;
5. measure bank conflicts and transaction efficiency;
6. tune tile shape and vector width.

### Invariants

- the logical permutation is bijective over valid output coordinates;
- no SMEM address is written by multiple threads;
- all SMEM writes complete before reads;
- vectorized transactions remain within valid rows/segments.

## Pattern C: Reductions, Norms, and Softmax

### Suitable operations

Sum/max/min, layer norm, RMS norm, softmax, row/column statistics.

### Hierarchical reduction

`per-thread values → warp reduction → CTA reduction (if needed) → result/epilogue`

Use register accumulation first. Use warp shuffle or architecture utility for intra-warp exchange. Use SMEM and a CTA barrier only when multiple warps contribute to one logical reduction.

### Numeric policy

State:

- accumulator dtype;
- stable algorithm (for example, online max/sum for softmax);
- treatment of NaN and Inf;
- behavior for empty or fully masked reductions;
- conversion and epsilon placement.

For norm, avoid computing variance as `E[x²] - E[x]²` when cancellation is unacceptable. For softmax, preserve the max/subtraction invariant and define mask semantics.

### Shape regimes

A single row may be:

- smaller than a warp;
- one or several warps;
- one CTA;
- larger than a CTA and require a multi-pass or persistent strategy.

Do not force one tile onto all regimes. Dispatch a small family if benchmarks justify it.

### Pipeline opportunity

Overlap input loads with reduction only after the non-pipelined version is correct. The dependency chain of max/sum may limit useful staging.

### Common failures

- divergent threads skip a CTA barrier;
- reduction identity is wrong for masked lanes;
- multiple warps race on a shared partial;
- output statistic and normalized output use inconsistent precision.

## Pattern D: GEMV and Small-K Matmul

### Characteristics

These can be memory-bound or latency-bound and often underuse a full GEMM mainloop. Compare:

- SIMT dot-product reduction;
- warp-level MMA;
- batched/grouped processing;
- fusing the following pointwise operation.

### Design choices

- Map the long output dimension across CTAs.
- Parallelize the reduction axis enough to hide latency without excessive final reduction.
- Reuse the vector operand in registers/SMEM where beneficial.
- For very small K, launch overhead and framework integration can dominate; benchmark end to end.
- Keep a non-MMA fallback for unsupported dtype/alignment.

## Pattern E: Dense GEMM-Like Kernels

### Logical contract

For a conventional multiply:

`C[m,n] = epilogue(sum_k convert(A[m,k]) * convert(B[n,k]), prior_C_or_bias, ...)`

Write the actual operand major modes; do not rely on “transposed” labels alone.

### Dataflow

1. CTA selects M/N tile and iterates over K tiles.
2. Producer loads A/B K tiles into staged SMEM.
3. Consumer obtains architecture-specific fragments/descriptors.
4. Tiled MMA accumulates C.
5. Mainloop drains.
6. Epilogue converts, fuses, predicates, and stores.

### Development order

1. validate tiled MMA construction and accumulator layout;
2. implement one K tile, synchronous movement;
3. loop over K synchronously;
4. implement predicated M/N/K tails or declare strict divisibility;
5. add architecture pipeline;
6. add epilogue fusion;
7. tune tile, warpgroup count, stage count, cluster shape, and store method.

### Invariants

- A/B partitions match the atom’s expected logical modes;
- invalid K values are zero-filled;
- accumulators are initialized exactly once;
- stage reuse occurs only after consumer release;
- epilogue maps every valid accumulator coordinate to one output coordinate;
- split-K or multi-CTA reduction has an explicit combining protocol.

### Tile selection

A larger CTA tile increases reuse but also SMEM, registers/TMEM, synchronization cost, and tail waste. Choose candidates from official examples for the dtype/architecture, then autotune over the actual shape distribution.

## Pattern F: Fused Epilogue

### Suitable fusion

Bias, scale, activation, residual, output conversion, auxiliary statistics, or quantization metadata when data is already resident near the accumulator.

### Rules

- preserve accumulation precision until the mathematically correct conversion point;
- separate logical coordinate mapping from physical store layout;
- predicate stores using the output coordinate tensor;
- account for additional live registers and SMEM;
- specify aliasing for residual or prior-output reads;
- compare fused and unfused numerics, not only speed.

For Hopper/Blackwell, epilogue staging can be a separate producer/consumer pipeline. Model it explicitly rather than reusing mainloop barriers by accident.

## Pattern G: Attention / FMHA

### Phases

A tiled forward attention kernel typically combines:

1. load Q and a K/V block;
2. QK MMA;
3. row-wise max/sum update;
4. probability conversion or scaling;
5. PV MMA;
6. normalization and store.

The exact schedule depends on architecture and dtype.

### Online-softmax invariants

For each row and block, maintain a numerically stable running maximum and normalization sum. When the maximum changes, rescale the existing accumulator consistently. Masked keys must contribute zero probability. Define the all-masked result.

### Design pressure

Attention keeps more intermediates live than GEMM. Monitor:

- accumulator registers/TMEM;
- softmax state;
- Q/K/V staging;
- causal/local mask coordinates;
- pipeline overlap;
- epilogue state.

Start from the closest official FMHA example and remove features before adding new ones.

### Testing

Include causal and noncausal; ragged sequence lengths; all-masked rows; extreme logits; head dimensions around tile boundaries; dropout or auxiliary outputs if present.

## Pattern H: Grouped, Persistent, and MoE Kernels

### Scheduler-first design

A persistent kernel is a distributed scheduler plus a compute body. Define:

- work descriptor format;
- atomic/queue ownership;
- problem-to-CTA mapping;
- cluster implications;
- termination condition;
- per-work-item descriptor and barrier reinitialization;
- fairness/load-balancing strategy.

Prove the scheduler independently with a lightweight kernel before combining it with expensive MMA.

### State hygiene

When a CTA processes multiple work items:

- reset predicates, descriptors, pipeline phases, and accumulator state;
- ensure prior asynchronous stores are complete before storage reuse;
- validate varying shapes/strides/dtypes allowed by the interface;
- avoid caching pointers or layouts whose lifetime belongs to a previous item.

### Grouped GEMM

Bucket compatible problems when specialization or descriptor setup cost matters. Document whether each problem may have independent M/N/K, strides, alpha/beta, and epilogue.

## Pattern I: Mixed-Input and Block-Scaled Kernels

Use only an official atom and example supporting the exact dtype combination and architecture.

### Additional contracts

- scale-factor tensor logical shape and granularity;
- scale-factor physical layout required by the MMA;
- zero-point semantics, if any;
- conversion/rounding/saturation;
- block alignment and tail handling;
- where scale factors live: GMEM, SMEM, RMEM, or TMEM;
- whether scale factors are shared across rows/columns/K blocks.

Test with nonuniform scale factors that expose indexing errors. Constant scale factors can hide a wrong layout.

## Choosing Between Fusion and Composition

Fuse when:

- an intermediate would otherwise round-trip through GMEM;
- producer/consumer tiles align;
- added live state does not collapse occupancy or spill;
- the combined synchronization remains understandable;
- correctness can be tested phase by phase.

Keep separate kernels when:

- shape regimes or optimal tiles differ sharply;
- fusion creates a global synchronization need;
- live ranges explode;
- an intermediate is reused by multiple consumers;
- framework/compiler launch overhead is not material.

Benchmark both end-to-end, including compilation/cache and launch overhead when relevant.
