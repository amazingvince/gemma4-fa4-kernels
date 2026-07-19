# Layouts, Tensors, Tiling, Partitioning, and Predication

## Core Model

A CuTe `Layout` is a function from a coordinate space to an index space. It is represented by a hierarchical **shape** and **stride**. A `Tensor` combines an engine or pointer with a layout.

This is more general than “row-major versus column-major.” Hierarchical modes encode tiles, subtiles, lane/value ownership, instruction fragments, swizzles, and composed address transformations.

Reason in this order:

1. logical coordinates;
2. layout mapping;
3. tiling;
4. ownership partition;
5. memory instruction;
6. compute instruction.

Do not start with pointer arithmetic.

## Notation and Invariants

A printed layout is commonly shown as:

`Shape : Stride`

Static integers are distinguishable from runtime integers in printed/debug forms. Shape and stride must be congruent: their tuple profiles align mode by mode.

For each layout, record:

- rank and hierarchical depth;
- logical mode names (`M`, `N`, `K`, batch, head, row, column, etc.);
- shape and stride;
- static/dynamic status;
- memory space;
- alignment;
- size and storage span;
- whether the mapping is injective for writes;
- expected contiguous/vector mode.

A write layout with duplicated indices is a race unless the operation is intentionally atomic or otherwise coordinated.

## Essential Layout Algebra

### Coalesce

Coalescing simplifies adjacent modes without changing the one-dimensional mapping or total size. Use it to expose vectorizable contiguous regions or reduce irrelevant hierarchy. Do not coalesce away semantic mode boundaries that later logic needs; use by-mode operations when preserving rank matters.

### Composition

Composition applies one layout through another. It is the foundation for expressing a sublayout, instruction mapping, swizzle, and thread/value ownership. Think:

`result(coord) = outer(inner(coord))`

When a partition is confusing, expand the composition on a tiny shape and print the resulting mapping.

### Divide/tiling

Logical, zipped, tiled, and flat divide operations split a tensor/layout into a **tile** and **rest**. The variants differ in how tile and rest modes are grouped.

Use a division operation to express “which local block” and a slice to select a specific rest coordinate. Prefer `local_tile` for the common pattern of applying a CTA tiler and selecting a CTA coordinate.

### Product

Products extend layouts with repeated or tiled structure. Use them to build larger thread/value arrangements from a proven atom. Check that the resulting hierarchy matches the API that consumes it.

### Swizzle and composed layouts

A swizzle is an address transformation used mainly for shared-memory bank behavior and instruction-compatible layouts. Treat it as part of the SMEM layout, not as a decoration. Prove:

- the allocated storage span covers the transformed indices;
- required alignment is satisfied;
- the copy and MMA views use compatible composed layouts;
- the epilogue reverses or interprets the mapping correctly.

## Tiling Hierarchy

A practical hierarchy for a GEMM-like kernel is:

1. problem tensor `(M,K)`, `(N,K)`, `(M,N)`;
2. CTA tile `(BLK_M, BLK_N, BLK_K)`;
3. warp or warpgroup tile;
4. instruction atom tile;
5. per-thread/per-lane value fragment.

For other kernels, replace modes but keep the hierarchy explicit.

Every level should answer:

- Which logical region does this agent own?
- Which values does it load?
- Where are those values stored?
- Which instruction consumes them?
- How are boundary values masked?

## `local_tile` and CTA Ownership

Use a CTA coordinate and a tiler to select the CTA’s problem region. Preserve a rest mode when the CTA iterates over tiles, such as K tiles in GEMM.

Write expected shapes next to each tiling expression. Example conceptual annotation:

- `gA`: `(CTA_M, CTA_K, K_TILE_INDEX)`
- `gB`: `(CTA_N, CTA_K, K_TILE_INDEX)`
- `gC`: `(CTA_M, CTA_N)`

If an actual shape differs, stop and inspect the tiler/step projection.

## Local Partitioning

A thread layout partitions a tile across agents. The partition is not just a workload split; it determines memory coalescing, vector width, fragment arrangement, and instruction legality.

Use `local_partition` for simple SIMT ownership. For copies and MMA, prefer the dedicated partition methods of TiledCopy/TiledMMA because source and destination mappings may differ.

## TiledCopy Pattern

A TiledCopy combines:

- a copy atom/instruction;
- a layout of participating threads;
- a value layout per thread.

The standard pattern is:

1. construct or select the TiledCopy;
2. obtain the thread slice for the current thread;
3. partition the source with `partition_S`;
4. partition the destination with `partition_D`;
5. verify the instruction/value mode has the required width;
6. execute the copy;
7. synchronize according to the memory primitive.

Proof obligations:

- source and destination partitions are congruent in element count;
- pointer alignment supports the copy width;
- vector lanes do not cross invalid boundaries;
- the destination layout matches the consumer;
- predicates correspond to the partitioned values, not merely the CTA tile.

Do not assume source and destination partitions are identical. Some instructions intentionally rearrange values.

## TiledMMA Pattern

A TiledMMA combines an MMA atom with a tiling over participating agents. Use its partitioning methods to derive operand and accumulator views.

Conceptual flow:

1. choose an architecture/dtype-compatible MMA atom;
2. build or use a helper to create the tiled MMA;
3. obtain the current agent’s slice;
4. partition A, B, and C;
5. create fragments in the required memory spaces;
6. retile copy results when the MMA fragment ordering requires it;
7. execute MMA groups with architecture-specific ordering;
8. transfer accumulators through the epilogue.

Do not manually map lanes into MMA fragments unless implementing a missing primitive and validating against the instruction specification.

## Coordinate-Tensor Predication

The most general tail strategy is to create a logical identity/coordinate tensor with the same problem shape, apply exactly the same tiling and partitioning as the data tensor, and compare resulting logical coordinates against bounds.

This has three benefits:

- it does not depend on physical input strides;
- it survives arbitrary CTA/thread/MMA partitions;
- it generalizes to any rank.

For each operand:

1. create a coordinate tensor for its logical shape;
2. apply the same CTA tiling;
3. apply the same copy/MMA partition;
4. derive a predicate over the logical modes relevant to that access;
5. use a neutral fill for invalid loads;
6. mask invalid stores.

### Neutral values

Choose by operation:

- sum/GEMM input: zero;
- maximum: negative infinity or the minimum representable value as mathematically appropriate;
- minimum: positive infinity;
- product: one;
- softmax mask: a value that produces zero probability under the defined numeric policy.

State behavior for all-masked rows; do not let it emerge accidentally as `NaN`.

### Predicate reuse

If a bound is invariant across loop iterations, compute the predicate once in the prologue and broadcast it through a stride-zero or equivalent view. Do not recompute expensive coordinate checks in the inner loop without evidence it is necessary.

## Shared-Memory Layouts

SMEM layouts serve two goals:

1. producer efficiency: coalesced/vectorized writes from GMEM;
2. consumer efficiency: bank-conflict-safe, instruction-compatible reads for MMA or other compute.

Those goals may require a swizzled or transformed layout. Select a layout from an official helper/example for the architecture and atom before designing a custom swizzle.

Validate custom SMEM layouts with:

- tiny mapping printout;
- storage `cosize` or equivalent span;
- alignment;
- bank-conflict metrics or profiler evidence;
- source/destination partition visualization;
- correctness with repeated values that reveal aliasing.

## Memory Spaces and Naming

Use official-style prefixes so dataflow is visible:

| Prefix | Meaning |
|---|---|
| `g` | global-memory tensor |
| `s` | shared-memory tensor |
| `r` | register-memory tensor/fragment |
| `t` | tensor-memory (TMEM) tensor where supported |
| `tA`, `tB`, `tC` | thread/tiled-MMA slices or partitions associated with operands |
| `t...g...`, `t...s...`, `t...r...` | partitioned tensor showing owner and memory space |

Names should encode both role and memory location. Avoid generic `tmp1`, `frag`, or `buf` when multiple pipeline stages exist.

## Layout-Proof Worksheet

For every important tensor, fill a row:

| Tensor | Logical modes | Shape/stride | Memspace | Owner | Vector/instruction mode | Bounds strategy | Consumer |
|---|---|---|---|---|---|---|---|
| `<name>` | `<modes>` | `<layout>` | `<G/S/R/T>` | `<CTA/warp/thread>` | `<width/atom>` | `<predicate/padded>` | `<copy/MMA/store>` |

Then check:

- [ ] All participating threads own the expected number of values.
- [ ] Source and destination partitions contain equal logical values.
- [ ] Every output coordinate has one writer, or reduction/atomic semantics are explicit.
- [ ] Every read coordinate is initialized before consumption.
- [ ] Storage spans and alignments cover composed/swizzled addresses.
- [ ] Tail predicates are congruent with the final partition.
- [ ] MMA fragments are created by the selected TiledMMA.
- [ ] Layout assertions encode assumptions rather than comments alone.

## Debugging a Layout Mismatch

1. Replace real dimensions with a tiny, non-square, non-divisible example.
2. Print logical tensor layouts.
3. Print CTA tiles.
4. Print per-thread source and destination partitions.
5. Enumerate `(logical coordinate → physical offset)` pairs.
6. Check duplicates and missing coordinates.
7. Temporarily use scalar copies and synchronous compute.
8. Reintroduce vectorization, swizzle, and async movement one at a time.

A wrong layout often still produces plausible values for square or repeated inputs. Use unique coordinate-coded test data.
