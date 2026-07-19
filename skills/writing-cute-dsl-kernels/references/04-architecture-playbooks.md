# Architecture Playbooks: Warp MMA, WGMMA, tcgen05, SM120, and CLC

## Rule Zero

Architecture families have different operand locations, execution agents, barrier protocols, and accumulator storage. Similar-looking helper names do not make their synchronization interchangeable.

Before using a playbook, confirm:

- device compute capability;
- exact CUTLASS/CuTe DSL revision;
- supported operand/accumulator dtypes;
- operand major modes and alignment;
- instruction shape and participating agent set;
- official example that compiles for the same target.

## Comparison

| Property | Ampere/Ada warp MMA | Hopper SM90 WGMMA | Blackwell SM100/103/110 tcgen05 | Blackwell SM120/121 warp MMA |
|---|---|---|---|---|
| Compute agent | Warp | Warpgroup | CTA group/instruction-defined agents | Warp |
| A/B consumption | Register fragments populated from SMEM | Shared-memory descriptors for supported forms | Atom-specific descriptors/operands, commonly staged through SMEM | Register fragments and, for block-scaled ops, register scale factors populated from SMEM |
| Accumulator | RMEM | RMEM | TMEM | RMEM |
| Typical GMEM staging | `cp.async` or tiled copy | TMA | TMA | Tiled copy or supported async copy; verify exact example |
| Key compute ordering | Warp convergence plus copy/barrier ordering | warpgroup fence → MMA issue → commit group → wait group | tcgen05/TMEM allocation and MMA group/fence protocol | Warp convergence plus copy/barrier ordering; no tcgen05 group protocol |
| Epilogue source | RMEM accumulator | RMEM accumulator | TMEM → RMEM, then store path | RMEM accumulator |

## Playbook A: Ampere-Class Warp MMA

### Dataflow

`GMEM A/B → staged SMEM A/B → per-warp RMEM fragments → warp MMA → RMEM C → epilogue`

A/B are consumed from register fragments. A tiled copy compatible with the atom, often using an `ldmatrix`-style operation, moves SMEM data into the expected fragment arrangement.

### Construction sequence

1. Choose a warp MMA atom for target SM, dtypes, operand majors, and accumulator.
2. Build a TiledMMA by tiling the atom across one or more warps.
3. Define CTA tile as a multiple of the tiled MMA’s M/N/K coverage.
4. Choose SMEM layouts compatible with the load-to-fragment copy.
5. Construct GMEM→SMEM TiledCopy, optionally backed by `cp.async`.
6. Obtain per-thread MMA slices and A/B/C partitions.
7. Create RMEM A/B fragments from the MMA partitions.
8. Retile copy destinations into the MMA fragment view where required.
9. Loop over K:
   - ensure stage is ready;
   - load SMEM→RMEM fragments;
   - execute warp MMA;
   - release/reuse the stage only after all consumers finish.
10. Store accumulator through a predicated epilogue.

### Synchronization

- Commit and wait on asynchronous copy groups according to the chosen staging distance.
- Use CTA synchronization when all warps share SMEM and a producer/consumer handoff requires it.
- Do not overwrite a stage while another warp still reads it.
- Ensure all threads participating in a CTA barrier follow the same path.

### Common defects

- fragment copy partition does not match the MMA atom;
- missing retile between copy and MMA views;
- `cp.async` wait count off by one during prologue or drain;
- K-tail not zero-filled;
- vector copy alignment assumed from dtype rather than pointer/layout;
- too many stages reduce occupancy without hiding additional latency.

## Playbook B: Hopper SM90 WGMMA

### Dataflow

`GMEM A/B → staged SMEM A/B → WGMMA consumes SMEM descriptors → RMEM C → epilogue`

WGMMA changes the central design: supported A/B operands are described from shared memory rather than first being loaded into ordinary register fragments through `ldmatrix`.

### Construction sequence

1. Select a WGMMA atom matching dtype, major modes, accumulation, and instruction shape.
2. Build a tiled MMA over the warpgroup arrangement.
3. Select an official SM90 SMEM layout helper compatible with the descriptor.
4. Create TMA tensors/descriptors for GMEM→SMEM movement.
5. Allocate staged SMEM and mbarrier storage with required alignment.
6. Partition A/B for the TMA producer and create WGMMA fragments/descriptors.
7. Initialize producer/consumer pipeline states and agent roles.
8. Prologue-fill the first stages.
9. For each K tile:
   - consumer waits for the stage;
   - establish required warpgroup fence before MMA sees producer writes/register state;
   - issue one or more `cute.gemm`/WGMMA operations;
   - commit the MMA group;
   - wait according to the allowed in-flight group depth before dependent use or stage release;
   - release the consumed stage;
   - producer acquires and refills a future stage.
10. Drain WGMMA with a final wait for all groups before reading accumulators.
11. Run a predicated epilogue, optionally with a separate TMA-store pipeline.

### Synchronization contract

Write the exact ordering in comments adjacent to code. At minimum distinguish:

- TMA completion into SMEM;
- consumer visibility;
- warpgroup MMA issue ordering;
- MMA group completion;
- SMEM stage release;
- final accumulator visibility.

A TMA barrier becoming ready does not by itself mean every WGMMA group is complete.

### Warpgroup role partitioning

Some kernels dedicate warps/warpgroups to producer or consumer roles. For every named barrier, specify the arrival count and participating agents. A thread count copied from another tile/role configuration is a deadlock risk.

### Common defects

- WGMMA descriptor uses an incompatible SMEM swizzle/major mode;
- missing `warpgroup.fence` before issue;
- missing `commit_group` or final `wait_group(0)`;
- releasing SMEM before WGMMA has finished consuming it;
- producer and consumer advance pipeline phase/index differently;
- all warps allocate the same logical role but barriers assume specialized roles.

## Playbook C: Blackwell SM100-Family tcgen05

### Dataflow

A common dense pattern is:

`GMEM A/B → staged SMEM A/B → tcgen05 MMA → TMEM accumulator → TMEM→RMEM epilogue copy → GMEM`

TMEM is a first-class resource with explicit allocation/lifetime. Do not treat it as a register array or ordinary shared memory.

### Construction sequence

1. Select a tcgen05 atom for exact target, dtype, scale mode, and CTA-group form.
2. Build a tiled MMA with supported one-CTA or two-CTA tiling.
3. Choose TMA and SMEM layouts from a matching official example.
4. Determine TMEM columns/shape and allocation owner.
5. Allocate TMEM and synchronize so all required consumers see the allocation.
6. Build operand descriptors/fragments and the TMEM C fragment from the tiled MMA.
7. For block-scaled variants, construct the required scale-factor layout and move factors into their required SMEM/TMEM representation.
8. Establish the TMA producer/consumer pipelines.
9. Issue tcgen05 MMA groups according to the atom’s fence/commit/wait protocol.
10. Drain all MMA work before reading TMEM.
11. Copy TMEM accumulator tiles to RMEM in an epilogue-friendly partition.
12. Apply conversion/fusion and predicated stores.
13. Complete outstanding stores before reusing storage.
14. Deallocate TMEM with the required participating agents.

### SM100 versus SM103/SM110

SM103 and SM110 belong to the tcgen05-capable family in current APIs, but they are not permission to reuse every SM100 atom, helper, schedule, or tuning result unchanged.

For each exact target:

- select an op whose supported-architecture list includes the target suffix (`a`/`f` distinctions included where relevant);
- recheck dtype, instruction shape, major-mode, scale-factor, CTA-group, and CUDA-toolkit requirements;
- compile with an explicit architecture target and confirm the expected instruction in PTX/SASS;
- retune stage count, tile, cluster, epilogue, and scheduler policy;
- keep dispatch keys distinct when generated code or performance policy differs.

Treat target-family predicates as capability guards, not as proof that two targets share optimal configuration.

### One-CTA versus two-CTA

Treat CTA-group size as a fundamental algorithm choice, not a launch tweak. It changes:

- tiled MMA construction;
- cluster/CTA coordination;
- barrier participants;
- operand sharing;
- TMEM ownership;
- launch shape and occupancy.

A two-CTA atom cannot safely run under a one-CTA synchronization plan.

### TMEM proof obligations

- allocation size and column layout match the accumulator fragment;
- exactly the required agent allocates/deallocates;
- all users wait for allocation visibility;
- no read occurs before MMA completion;
- epilogue copies cover every accumulator coordinate once;
- deallocation occurs after the final user;
- persistent-loop iterations do not leak or reuse stale TMEM state.

### Block-scaled and mixed-input variants

Scale-factor layouts are instruction contracts. Record granularity and logical axes, then verify the physical layout from current docs/source. Test with per-block unique scales.

### Common defects

- assuming RMEM accumulator APIs on a TMEM atom;
- wrong CTA-group launch or barrier count;
- TMEM epilogue partition does not match C coordinates;
- deallocating before an asynchronous epilogue read completes;
- scale-factor tensor broadcasted along the wrong mode;
- importing an SM100 helper into a different Blackwell target without verification.

## Playbook D: Blackwell SM120/SM121 Warp MMA

### Dataflow

Current CuTe DSL warp-level programming guidance places SM120 block-scaled Tensor Core operations on the warp-synchronous `mma.sync` path:

`GMEM A/B/scales → staged SMEM → per-warp RMEM fragments/scales → warp MMA → RMEM C → epilogue`

This is not the SM100 tcgen05/TMEM model. Do not allocate TMEM, construct tcgen05 atoms, use one-/two-CTA tcgen05 groups, or copy an SM100 epilogue merely because both targets are named Blackwell.

### Construction sequence

1. Confirm the exact SM120/SM121 target, CUDA toolkit, wheel revision, dtype, and scale format.
2. Start from the matching official `blackwell_geforce` or SM120 warp-MMA example in that revision.
3. Select a supported `cutlass.cute.nvgpu.warp` op and create the TiledMMA.
4. Derive A/B/C and scale-factor fragments from the tiled MMA; keep all warp participants converged for `cute.gemm`.
5. Stage values and scale factors through layouts compatible with the atom. For block-scaled ops, preserve the scale-vector granularity and filter/predicate padded scale entries exactly as the example requires.
6. Use the ordinary warp-MMA copy/barrier lifecycle; no WGMMA or tcgen05 commit/wait protocol applies to the MMA itself.
7. Run the RMEM epilogue with independent M/N predicates and validate K/scale tails.
8. Confirm the target instruction and scale operands in generated PTX/SASS.

### Review obligations

- no import or helper implies tcgen05/TMEM ownership;
- every selected op lists the exact architecture as supported;
- block-scale tensor coordinates match the K-vector granularity;
- all 32 lanes issue MMA in convergence;
- SM100 and SM120 binaries/cache keys/dispatch branches remain separate;
- tuning evidence is collected on the actual SM120/SM121 device.

### Common defects

- routing SM120 to an SM100 `tcgen05` kernel;
- describing SM120 merely as “Ampere-like” and missing its target-specific block-scaled ops;
- reusing SM100 scale-factor layouts or TMEM epilogues;
- forgetting that scale factors are MMA operands and need their own partition/predication proof;
- compiling a generic `sm_120` target when an architecture-specific `a`/`f` feature target is required.

## Blackwell Cluster Launch Control Scheduling

Cluster Launch Control (CLC) is a **persistent scheduler contract**, not a replacement for the selected MMA/mainloop playbook. Use it only when the exact target, launch mode, and installed CuTe DSL revision support the CLC operations and scheduler utilities.

### Intended dataflow

```text
initial cluster tile from block coordinates
    → scheduler producer in CTA 0 issues asynchronous CLC query
    → hardware writes a 16-byte response to per-CTA SMEM
    → consumer agents wait, decode the tile coordinate, and process it
    → consumers release the response stage
    → repeat until the response declines/terminates
```

The current high-level route is `cutlass.utils.ClcDynamicPersistentTileScheduler`; the current pipeline route is `cutlass.pipeline.PipelineClcFetchAsync`. Verify both names and constructors in the installed revision.

### Construction obligations

1. Start from the closest official CLC-enabled kernel in the matching revision.
2. Define the cluster shape, static/persistent grid policy, initial tile coordinates, and mapping from returned CLC IDs to logical work.
3. Reserve SMEM for one response per pipeline stage and account for the documented 16-byte response transaction.
4. Elect the scheduler producer exactly as required—normally one issuing thread from the scheduler warp in CTA 0 of the cluster.
5. Set producer/consumer cooperative groups and arrival counts from the actual participant set; consumers include every thread/warp/CTA that waits on and releases a response.
6. Preserve the generic-async proxy ordering required between asynchronous CLC writes and generic SMEM loads. Use the scheduler's default fence behavior unless an equivalent fence is deliberately placed in the pipeline lifecycle.
7. Prove termination: every cluster agrees when no valid ID remains, no consumer waits for an unissued response, and all used response stages are released before role exit.
8. Keep a non-CLC static or conventional persistent scheduler as a correctness fallback when the target or launch environment does not support CLC.

### Common defects

- treating a CLC response as ready before the full-stage wait;
- wrong transaction bytes or arrival count;
- allowing every CTA or every lane to issue a query;
- omitting the cross-proxy fence and racing the next query with the current SMEM read;
- using the returned physical ID without the cluster-shape coordinate transform;
- scheduler shutdown while consumers remain in mainloop or epilogue pipelines;
- assuming CLC makes an imbalanced tile shape efficient without measurement.

Run synchronization checking and an irregular workload matrix that stresses partial waves, occupied SMs, nontrivial cluster shapes, and the final declined response. Inspect generated code to verify the CLC instruction path before attributing performance to it.

## Epilogue Choice by Architecture

| Architecture | Simple path | More advanced path |
|---|---|---|
| Warp MMA, including SM120/121 | RMEM accumulator → predicated vector GMEM store | RMEM → SMEM reorder → vector GMEM store |
| WGMMA | RMEM accumulator → tiled store | RMEM → staged SMEM/TMA store pipeline |
| tcgen05 | TMEM → RMEM → predicated GMEM store | TMEM → RMEM/SMEM transform → TMA or cooperative store |

Select the simplest correct path first. An advanced epilogue can dominate debugging time and resource use.

## Architecture Review Questions

- [ ] Is the exact SM encoded in build/compile/dispatch, with SM100/103/110 separated from SM120/121?
- [ ] Is the instruction atom supported for all requested dtypes and majors?
- [ ] Are operand memory spaces correct for that atom?
- [ ] Is accumulator memory space correct?
- [ ] Are SMEM layout and descriptor construction from a matching example?
- [ ] Does the launch agent set match barriers and CTA-group requirements?
- [ ] Are all instruction groups drained before accumulator use?
- [ ] Are all memory stages released only after final consumption?
- [ ] Does a fallback cover unsupported device/input combinations?
- [ ] Was generated PTX/SASS inspected when an expected instruction did not appear?
