# CuTe DSL Kernel Review Checklist

## Version and Scope

- [ ] Package/repository revision, CUDA, GPU, and target SM are recorded.
- [ ] Imported symbols and volatile helper signatures were verified against that revision.
- [ ] The implementation states its supported shapes, strides, alignment, dtypes, and devices.
- [ ] Unsupported cases reject clearly or dispatch to a valid fallback.
- [ ] JIT specialization count is bounded and measured.

## Logical Correctness

- [ ] Mathematical formula and accumulation/conversion order match the reference.
- [ ] NaN/Inf, masks, zero extents, and all-masked/empty cases are defined.
- [ ] Aliasing and in-place behavior are safe or rejected.
- [ ] Every valid output coordinate has exactly one writer or explicit reduction/atomic semantics.

## Layout and Partitioning

- [ ] Each important tensor has documented logical modes, shape/stride, memspace, and owner.
- [ ] CTA, warp/warpgroup, thread, copy, and MMA partitions are congruent.
- [ ] Source and destination value counts match for each copy.
- [ ] Vector width and alignment are proven for every lane/start.
- [ ] SMEM swizzle/storage span and alignment are validated.
- [ ] MMA fragments/descriptors come from the selected TiledMMA.
- [ ] Coordinate tensors undergo the same tiling/partitioning as data.
- [ ] Tail loads use correct neutral values and tail stores are masked.

## Architecture

- [ ] Atom supports exact SM, dtype, majors, instruction shape, and accumulation.
- [ ] Operand and accumulator memory spaces match the architecture playbook.
- [ ] CTA-group/cluster launch matches instruction and barrier assumptions.
- [ ] Ampere, WGMMA, and tcgen05 synchronization idioms are not mixed.
- [ ] Generated code contains the expected target instruction when performance depends on it.

## Synchronization and Pipelines

- [ ] Every asynchronous operation has acquire/issue/commit/wait/consume/release ownership.
- [ ] Barrier participants and arrival counts match role topology.
- [ ] Stage index and phase advance consistently for producer and consumer.
- [ ] Prologue, zero/one-iteration, steady state, and drain are correct.
- [ ] SMEM/TMEM is not reused before all consumers and async engines finish.
- [ ] WGMMA/tcgen05 groups are fully drained before accumulator use.
- [ ] TMEM allocation, visibility, use, and deallocation are balanced.
- [ ] Persistent iterations reset all state.
- [ ] No participating barrier is reachable through divergent control flow.

## Integration

- [ ] Public wrapper validates device, dtype, rank, stride, alignment, and integer limits.
- [ ] Correct framework/current stream is used.
- [ ] DLPack/allocation lifetime covers asynchronous execution.
- [ ] Compiled executor is cached with an intentional key.
- [ ] Compilation and first-call overhead are separated from steady-state timing.

## Tests and Debug Evidence

- [ ] Reference comparisons cover boundary, tail, layout, value, dtype, and concurrency cases.
- [ ] Coordinate-coded inputs and nonuniform scale factors are used where relevant.
- [ ] Numerical tolerance is justified.
- [ ] Sanitizer/race/synchronization checks pass on representative cases.
- [ ] Invalid input tests verify rejection/fallback.
- [ ] Nondeterminism/repeated-run tests pass.

## Performance Evidence

- [ ] Device/software versions and clocks/environment are reported.
- [ ] CUDA-event or equivalent synchronized device timing is used.
- [ ] Warmup, repetitions, statistic, and variability are stated.
- [ ] Baseline uses comparable semantics and dtype.
- [ ] Resource usage/spills and expected instructions were inspected.
- [ ] Search space was validity-filtered and not overfit to one shape.
- [ ] Claims are limited to the measured workload.
