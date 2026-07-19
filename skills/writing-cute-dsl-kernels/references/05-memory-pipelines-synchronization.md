# Memory Movement, Pipelines, and Synchronization

## Start from Ownership

For each buffer and stage, write:

- producer agent;
- consumer agent;
- memory space;
- stage count;
- ready signal;
- free/reuse signal;
- stage index and phase;
- initialization and drain behavior.

A pipeline is correct only when every buffer has one unambiguous ownership state.

## Movement Options

| Mechanism | Use when | Required proof |
|---|---|---|
| Scalar/direct load-store | baseline, irregular or tiny access | bounds and coalescing |
| Vectorized TiledCopy | aligned, regular per-thread values | partition width, pointer alignment, row/tail boundaries |
| `cp.async` family | supported architecture, GMEM→SMEM overlap benefits | commit/wait groups, zero-fill/predication, stage reuse |
| TMA load | supported multidimensional transfer, descriptor-friendly tensor, Hopper/Blackwell playbook | descriptor shape/stride/alignment, mbarrier bytes/arrivals, cluster/multicast rules |
| TMA store | large regular SMEM→GMEM epilogue | store fence/commit/wait and SMEM lifetime |
| SMEM→RMEM tiled copy | MMA or vector consumer requires fragments | copy partition matches consumer layout |
| TMEM→RMEM copy | tcgen05 epilogue | MMA completion and TMEM fragment mapping |

Choose a more complex mechanism only when the simple path is insufficient or the architecture instruction requires it.

## Generic Circular Pipeline

A stage state includes:

- `index`: current circular slot;
- `phase`: distinguishes reuse cycles;
- producer/consumer role state.

Conceptual lifecycle:

```text
initialize barriers and states
producer prologue:
    acquire(stage)
    issue transfer into stage
    commit/arrive(stage)
steady state:
    consumer wait(stage)
    consume stage
    consumer release(stage)
    producer acquire(future_stage)
    producer issue/commit(future_stage)
drain:
    wait for all compute groups
    release final consumed stages
    wait for outstanding stores if storage will be reused/freed
```

The exact APIs differ by pipeline class and revision. The lifecycle does not.

## State-Machine Table

Complete this for every pipeline:

| Transition | Agent | Barrier/group | Index/phase before | Action | Index/phase after |
|---|---|---|---|---|---|
| acquire | producer | `<id>` | `<state>` | wait until free | `<state>` |
| issue | producer | `<id>` | | copy/MMA/store | |
| commit/arrive | producer | `<id>` | | publish work/bytes | advance producer |
| wait | consumer | `<id>` | `<state>` | wait ready/completed | |
| consume | consumer | | | use data | |
| release | consumer | `<id>` | | publish free | advance consumer |

If two roles use different advancement rules, write both explicitly.

## Barrier Agent Sets

Pipelines distinguish agents such as thread, warp, threadblock, or cluster. A barrier’s participant count and execution scope must match the actual role topology.

Check:

- every expected participant arrives;
- no unexpected participant arrives twice;
- inactive/tail CTAs still obey cluster-level protocols where required;
- warp-specialized producer and consumer branches do not cross each other’s barriers;
- named barriers are not reused for unrelated events before the prior generation completes.

## `cp.async` Pipeline Pattern

A conventional staged loop has:

1. prologue issues up to `stages - 1` future copies;
2. commit groups after issuing a logical stage;
3. wait until enough older groups have completed;
4. CTA synchronization if consumers across the CTA require it;
5. consume current stage;
6. synchronize before overwriting shared storage;
7. issue the next copy;
8. drain remaining groups.

### Off-by-one audit

Write a table for a tiny K-tile count (0, 1, `stages-1`, `stages`, `stages+1`). Track:

- which logical K tile is in each stage;
- number of committed groups;
- wait depth;
- current read/write index;
- whether stage data is initialized.

Most pipeline errors appear in prologue or drain, not steady state.

## TMA Load Pattern

TMA describes a multidimensional GMEM tensor and asynchronously transfers a tile into SMEM, with completion tracked by an mbarrier/pipeline.

Proof obligations:

- tensor rank/shape/stride are supported and correctly encoded;
- pointer and stride alignment satisfy the descriptor;
- transfer bytes match the expected barrier transaction;
- the elected issuer is unique where required;
- consumers wait on the correct barrier phase;
- multicast masks and cluster shape match;
- out-of-bounds behavior is explicitly supported by descriptor semantics or handled separately;
- source tensor lifetime extends through the transfer.

Do not assume a tensor accepted by DLPack automatically satisfies TMA alignment/stride constraints.

## TMA Store Pattern

A TMA store pipeline must distinguish:

- SMEM writes completed by producer threads;
- any fence making those writes visible to TMA;
- store issue/commit;
- completion before SMEM stage reuse or kernel-dependent consumption.

A kernel can return before asynchronous device work is host-synchronized, but it cannot destroy/reuse the source SMEM before the device operation has safely consumed it.

## MMA Group Ordering

### WGMMA

Track separately:

- operand readiness in SMEM;
- required warpgroup fence;
- issue of one or more MMA operations;
- commit of the group;
- wait depth;
- final wait before reading C.

A consumer may permit several groups in flight. Tune this only after proving the exact dependency and stage-release order.

### tcgen05

Follow current atom/API documentation for fence, commit, wait, and TMEM visibility. Include CTA-group participants and TMEM lifetime in the same state machine. Never import WGMMA ordering mechanically.

## Shared-Memory Reuse

Before overwriting a stage, all of the following must be true:

- asynchronous load into it completed previously;
- every consumer has finished all reads;
- MMA instruction groups no longer reference its descriptor/data;
- any epilogue/store using it has completed enough to permit reuse;
- the free/release barrier generation matches the producer’s next acquire.

A block-wide `sync_threads` is not a substitute for waiting on an asynchronous engine or MMA group.

## Stage Count

More stages can hide latency but increase:

- SMEM allocation;
- barrier storage;
- register state and live ranges;
- prologue/drain overhead;
- occupancy loss;
- compile complexity.

Candidate stage count is a tuning parameter constrained by resource feasibility. Eliminate candidates that exceed SMEM/TMEM/register/cluster limits before benchmarking.

## Role Specialization

Warp-specialized kernels assign separate producer and consumer roles. Benefits include reduced issue contention and better overlap; costs include fewer compute warps, more barriers, and role-dependent control flow.

Required design notes:

- role selection predicate;
- number of producer and consumer warps/warpgroups;
- barrier participants for each event;
- which role performs descriptor setup and elected TMA issue;
- termination behavior for every role;
- epilogue ownership.

Test with the smallest number of work tiles because role shutdown bugs often hide in long runs.

## Deadlock and Race Checklist

When a kernel hangs:

1. Reduce to one CTA; then one cluster if clustering is required.
2. Disable persistent scheduling.
3. Reduce stages to the minimum.
4. Log role, stage index, phase, and transition with bounded device prints or IKET markers.
5. Verify barrier initialization before any wait/arrive.
6. Compare actual participant counts to barrier configuration.
7. Check every branch reaches matching transitions.
8. Check prologue/drain for zero and one iteration.
9. Confirm CTA-group/cluster launch matches the atom.
10. Run compute-sanitizer synchronization/race tools where applicable.

When results are nondeterministic:

1. replace async movement with synchronous copies;
2. insert conservative waits/barriers;
3. disable stage overlap;
4. use unique coordinate-coded data;
5. re-enable one asynchronous dependency at a time.

## Resource Accounting

Before launch, estimate:

- static and dynamic SMEM per CTA;
- barrier and descriptor storage;
- registers per thread/warpgroup;
- TMEM columns/allocation;
- threads/warps per CTA;
- CTAs per cluster;
- expected resident CTAs/warpgroups.

Measure actual compiler-reported resources. A pipeline that is theoretically overlapped but allows only one resident CTA may lose to a simpler design.

## Instrumentation Points

For a staged kernel, mark:

- descriptor/setup;
- prologue;
- producer acquire/issue/commit;
- consumer wait;
- MMA issue/commit/wait;
- consumer release;
- epilogue load/transform/store;
- tail/drain.

Use lightweight instrumentation and remove or gate it for final benchmarks.
