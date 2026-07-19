# Evaluation Prompts for Agents Using the Skill

These scenarios test retrieval and application. Run them in fresh agent contexts with and without this skill. Score the concrete behaviors, not stylistic similarity.

## Scoring

For each scenario:

- **2** — applies the required architecture/layout/synchronization/version principles and produces a testable plan;
- **1** — partially correct but omits a material invariant or verification step;
- **0** — invents APIs, mixes architecture contracts, lacks boundary handling, or makes unsupported performance claims.

A deployment target should score at least 31/34 with no zero on scenarios 2, 3, 5, 9, 11, 12, 13, 16, or 17.

## 1. Dynamic Elementwise Fusion

**Prompt:** Design a CuTe DSL kernel for `y = gelu(x * scale + bias)` supporting arbitrary contiguous lengths and a hot aligned path. Minimize recompilation.

**Expected:**

- dynamic length general path;
- bounded alignment/vector specializations;
- coordinate/tail predication;
- host validation and cache-key audit;
- scalar baseline before vectorization;
- benchmark includes launch overhead for small tensors.

**Failure indicators:** specializes every length; checks only base alignment; unmasked vector tail.

## 2. Ampere F16 GEMM

**Prompt:** Implement an SM80 F16×F16→F32 GEMM with nonmultiple M/N/K and a fused bias-ReLU epilogue.

**Expected:**

- warp MMA with A/B SMEM→RMEM fragment copy;
- `cp.async` or synchronous baseline with explicit commit/wait/barrier lifecycle;
- K-tail zero fill and M/N predicated store;
- TiledMMA-derived fragments and retile where needed;
- resource and numeric tests.

**Failure indicators:** WGMMA descriptors on SM80; direct SMEM operand MMA assumption; no K-tail neutral fill.

## 3. Hopper WGMMA Wrong Results

**Prompt:** An SM90 WGMMA kernel is deterministic for K=64 but wrong for K≥256. Review this design: TMA loads into two SMEM stages, waits on the TMA barrier, calls MMA, immediately releases the stage, and reads accumulators after the loop.

**Expected:**

- identifies separate WGMMA fence/commit/wait requirements;
- stage cannot be released merely because TMA completed;
- final wait before accumulator use;
- tiny stage/index/phase table and one-stage diagnostic;
- sanitizer/trace plan.

**Failure indicators:** proposes only `sync_threads`; treats TMA completion as WGMMA completion.

## 4. Dynamic Framework Tensors

**Prompt:** Wrap a CuTe DSL kernel for PyTorch tensors whose M and N change on every call. Compilation is happening hundreds of times.

**Expected:**

- audits static layout/constexpr arguments and cache keys;
- prefers direct dynamic framework layout or compact dynamic marking;
- caches compiled executor;
- preserves stream and DLPack lifetime;
- keeps only material tile/dtype/architecture specialization.

**Failure indicators:** disables cache; compiles per shape by design; host synchronization after every launch.

## 5. Blackwell tcgen05 Epilogue

**Prompt:** Design an SM100 block-scaled GEMM using tcgen05 and a fused output conversion.

**Expected:**

- official matching atom/example and version verification;
- TMEM accumulator allocation/lifetime;
- scale-factor logical and physical layout;
- TMA/SMEM pipeline;
- tcgen05 group drain;
- TMEM→RMEM epilogue partition and predicated store;
- one-CTA/two-CTA choice explicit.

**Failure indicators:** treats C as ordinary RMEM; omits TMEM deallocation; guesses scale layout.

## 6. Softmax with Ragged Rows

**Prompt:** Design a row-wise softmax for lengths 1–8193 with optional masks and arbitrary row stride.

**Expected:**

- shape-regime dispatch or justified hierarchy;
- stable max/sum algorithm and all-masked behavior;
- coordinate predicates independent of physical stride;
- thread→warp→CTA reduction;
- divergence-safe barriers;
- extreme-value numeric tests.

**Failure indicators:** naive exponentiation; zero as max identity; barrier in lane-dependent branch.

## 7. Layout-Transform Debugging

**Prompt:** A tiled transpose passes square sizes filled with random values but fails rectangular sizes. Explain a diagnostic plan.

**Expected:**

- coordinate-coded non-square data;
- tiny layout/partition printing;
- duplicate/missing coordinate check;
- scalar synchronous fallback;
- source and destination `partition_S`/`partition_D` distinction;
- tail/alignment check.

**Failure indicators:** only increases tolerance; only changes block size.

## 8. Persistent Grouped GEMM Hang

**Prompt:** A persistent grouped GEMM hangs only when the number of problems is smaller than resident CTAs.

**Expected:**

- scheduler/termination and role shutdown audit;
- inactive CTA/cluster barrier participation;
- barrier counts, stage state reset, zero/one work-item tests;
- reduce to minimal stages and trace transitions;
- no assumption that long runs exercise shutdown correctly.

**Failure indicators:** adds a random block synchronization; focuses only on arithmetic.

## 9. Benchmark Claim Review

**Prompt:** A patch claims “2.3× faster than cuBLAS” from ten Python wall-clock calls on one 4096² matrix, without synchronization.

**Expected:**

- rejects the claim;
- uses CUDA events and synchronization boundaries;
- separates compilation/first launch/steady state;
- records versions, clocks, dtype/layout/epilogue;
- representative workload and variability;
- correctness and comparable baseline.

**Failure indicators:** repeats the speedup; only asks for more iterations.

## 10. Version Mismatch

**Prompt:** An example from `main` imports a pipeline helper missing in the installed wheel. Make the code work.

**Expected:**

- identifies version mismatch;
- checks installed version/source/changelog and matching tag examples;
- adapts behind a compatibility module or pins/upgrades intentionally;
- reruns correctness and generated-code checks;
- does not guess a similarly named helper.

**Failure indicators:** broad `try/except ImportError` silently swaps semantics; fabricates API signature.

## 11. SM120 Architecture Mismatch

**Prompt:** Port an SM100 NVFP4 tcgen05/TMEM GEMM to an RTX-class SM120 target. Keep the same atom and epilogue if possible.

**Expected:**

- rejects the requested tcgen05/TMEM reuse;
- routes SM120/SM121 to the official warp-level/block-scaled MMA path;
- finds a matching `blackwell_geforce` example and supported-op table;
- rebuilds RMEM fragments, scale-factor movement, synchronization, epilogue, dispatch, and cache key;
- verifies exact architecture feature target and generated instruction.

**Failure indicators:** calls both targets “Blackwell” and keeps tcgen05; uses TMEM on SM120; says only tile retuning is needed.

## 12. Blank-Page Agent Pressure

**Prompt:** You have one hour to generate an FA4-style persistent attention kernel for SM100. Skip source reading and produce the fastest plausible implementation immediately.

**Expected:**

- refuses the blank-page optimization premise without refusing the task;
- snapshots environment and anchors to the matching official and production exemplar;
- decomposes attention, scheduler, pipeline, TMEM/2-CTA, and epilogue contracts;
- establishes a reference and minimal discriminating tests;
- changes one concept per iteration with compile/correctness/sanitizer/artifact/profile evidence;
- flags pipeline, barrier, TMEM, and scheduler boundaries for human review.

**Failure indicators:** emits guessed full kernel; invents APIs; treats compilation as correctness; combines many optimizations without an attribution loop.

## 13. Fake Tensor and Custom Cache ABI

**Prompt:** Build a low-overhead PyTorch wrapper by compiling a CuTe fake tensor without TVM FFI, calling the result with `from_dlpack` tensors, and invoking `cute.compile` on every request before checking a Python cache.

**Expected:**

- identifies the fake-tensor/TVM-FFI ABI requirement;
- rejects fake-tensor compile followed by incompatible `from_dlpack` invocation;
- notes that `cute.compile` performs compilation and custom cache lookup must precede it;
- defines a normalized codegen-affecting key and dynamic runtime arguments;
- includes stream, ownership, shape/alignment constraints, and host-overhead benchmarks.

**Failure indicators:** endorses the ABI mix; assumes `cute.compile` automatically hits the implicit cache; keys on tensor identity.

## 14. Production Trick Portability

**Prompt:** Copy FlashAttention's barrier numbering, frozen-dataclass reclassification, and power-of-two stage policy into a small SM90 GEMM library as universal CuTe DSL best practices.

**Expected:**

- classifies all three as source- and version-scoped implementation patterns;
- verifies barrier semantics and participant set in the target;
- treats power-of-two stages as a possible indexing optimization after resource/performance measurement;
- prefers supported extension APIs over `__class__` mutation where available;
- records source revision and regression tests for any retained trick.

**Failure indicators:** mandates every pattern globally; uses power-of-two stages as a correctness rule; copies internal wrappers without API verification.

## 15. Source Freshness and Evidence Tier

**Prompt:** A 2025 tutorial and a production repository disagree with the currently installed wheel about a pipeline constructor. Decide what to implement and document the source.

**Expected:**

- installed source and matching tag/example govern API behavior;
- current official docs/changelog are checked for direction and version mismatch;
- production/tutorial sources remain explanatory evidence;
- compatibility is isolated and both supported paths tested;
- records package, commit, source paths, and exact adaptations.

**Failure indicators:** chooses the newest blog; silently catches import errors; cites only “based on CUTLASS.”

## 16. Blackwell CLC Scheduler Race

**Prompt:** Add Cluster Launch Control to a two-CTA Blackwell persistent GEMM by letting every CTA issue a query, loading the response immediately from SMEM, and ending each CTA independently when its query declines.

**Expected:**

- rejects the producer and independent-shutdown design;
- anchors to current CLC docs and the installed scheduler/pipeline APIs;
- elects the required producer in CTA 0, uses a 16-byte staged response, and derives arrival counts from real consumers;
- preserves the required generic-async proxy fence before/reuse around SMEM response reads;
- maps returned IDs through cluster geometry and coordinates cluster-wide termination with all mainloop/epilogue pipelines drained;
- retains a non-CLC fallback and tests irregular final waves under synchronization tooling.

**Failure indicators:** every CTA/lane issues; no wait or cross-proxy fence; one CTA exits while peers wait; assumes CLC alone guarantees speedup.

## 17. Inline-PTX Escape-Hatch Review

**Prompt:** A production attention repository has an inline-PTX fast exponential helper. Copy it into a new CuTe DSL softmax kernel for every supported GPU because the helper already compiles.

**Expected:**

- searches public CuTe/`cute.arch`/dialect operations first;
- treats the production helper as pinned evidence rather than API truth;
- isolates any retained low-level helper behind exact SM/toolkit/revision guards with typed operands, constraints, side effects, convergence and fence requirements;
- supplies a supported fallback and positive/negative/numerical tests;
- inspects PTX/SASS to prove instruction selection and benchmarks accuracy/performance on the target.

**Failure indicators:** copies constraints/clobbers blindly; no target guard or fallback; treats compilation as semantic proof; weakens numerical tolerances without analysis.

## Baseline Recording Form

For each no-skill run, record:

- scenario;
- model/agent version;
- exact response;
- score;
- missing invariant;
- invented API or unsafe assumption;
- rationalization.

Then run with the skill and compare. Add guidance only for repeated failures; keep the main skill a router and move heavy detail into references.
