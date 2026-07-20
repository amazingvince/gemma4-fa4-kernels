# EXP-0019: H100 whole-layer opaque compiler boundary

- Date / author: 2026-07-20 / Codex
- Status: **REJECTED on the declared Inductor graph bound**
- Kernel family: pinned Transformers local-d256 and global-d512 attention
  layers over the retained FA4 forward paths
- Architecture: sm_90
- Starting revision: `292e8923b7e1cc9d45b7c328cfb0125edfe00120`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Starting Transformers patch SHA256:
  `c812937e5a554c1887c2c16a0808f24437cb8b60b561e9fd5eacaa13fb277780`
- H100 FA4 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Starting environment-policy hash:
  `d059cc4e7849c85059d9bfcdc433bf2a2948d6a8eb4ffe5c4b5b967a7434df01`
- Immediate evidence: rejected EXP-0018 local default-Inductor S1023
  `max_abs=0.0703125`, `mean_abs=0.00742268236` against its frozen
  full-layer `atol=0.0625, rtol=0.02`, while direct prepared FA4 references
  and the local/eager S1/S33 compiler smoke passed.

## Invariant changed

No model, mask, prepared-attention numerical, tensor-ownership, CuTe tile,
pipeline, synchronization, backward, or performance invariant changes.
Retain exact BF16 prepared Q/K/V, scale 1.0, distinct K/V, FP32 LSE, local
sliding causality, global causality, the exact 32Q/16KV/d256 and
32Q/4KV/d512 geometries, and every accepted eager/packed/backward path.

Change only the positive no-cache compiler opacity boundary. EXP-0018 left
the pinned Q/K/V projections, FP32 RMS normalization, rotary arithmetic,
reshape/contiguous transport, and output projection visible to Inductor and
made only prepared FA4 opaque. EXP-0019 may make the exact pinned attention
layer algebra opaque as one family-specific, tensor-explicit custom op so the
real body executes the same eager operation order around the retained FA4
kernel. This does not make a whole decoder block, MLP, cache, backward, or
model opaque.

The EXP-0018 mask-boundary cache/origin/recipient proof remains mandatory.
No direct/public/wrong-family/forwarding wrapper may mint compiler authority,
and every non-null cache must still reject before the attention layer.

## Declared envelope

The positive scope remains:

- H100/SM90, BF16 hidden state/weights/output, FP32 LSE, scale exactly 1.0;
- actual pinned `Gemma4TextAttention` layer 0 (local) and layer 5 (global),
  exact locked config, eval inference, no active gradient;
- B1 text-only self-attention, equal query/key lengths, `1 <= S <= 1024`;
- exact zero-based contiguous `position_ids == arange(S)`;
- no cache, padding, vision/document metadata, explicit cu-seqlens/maxima,
  offsets, arbitrary mask, custom overlay, shared prepared KV, fallback,
  export, CUDA graph, or performance claim.

The whole-layer op must take every semantically relevant tensor as an explicit
argument: hidden state, cosine/sine rotary tensors, positions, derived packed
IDs, Q/K/V/output projection weights as applicable, and Q/K RMS scale weights.
The pinned config has no projection bias; a bias, missing/extra weight,
requires-grad tensor, alias, wrong shape/stride/dtype/device, changed module
mode, shared-KV layer, or changed layer index fails closed. Global retains one
shared K-projection source but must create distinct prepared K and V storage
after their different normalizations and rotary treatment.

Public PyTorch 2.8 `torch.compile(fullgraph=True, dynamic=True)` may retain
exactly the declared two graph classes: S1 and one shared S>1 class. The
private scoped size-oblivious one-graph result remains diagnostic only.

## Hypothesis

The EXP-0018 S1023 mismatch is introduced outside the prepared FA4 opaque op
by Inductor's compilation of the surrounding pinned attention-layer algebra.
If that exact algebra is moved into one tensor-explicit whole-layer custom op,
then the actual pinned local/global layers will be bitwise equal to their
uncompiled eager outputs across the full declared ladder while the retained
prepared O/FP32-LSE references, bounded graph/cache classes, mask provenance,
and pre-entry cache rejection remain unchanged.

Falsification is any localization result showing a prepared FA4 semantic
failure rather than outer-layer compiler drift; any non-bitwise pinned-eager
versus whole-op or compiled whole-layer output; a changed direct O/LSE policy;
missing/implicit module state; a graph break; more than S1/S>1 public graph
classes; a missing whole-layer custom-op node; cache/mask admission; an
unbounded FA4 or compiler cache; changed retained FA4 main-object bytes or
resources; or a project-kernel sanitizer finding.

## Single change

Move the compile-only positive boundary from prepared attention to the exact
pinned attention layer:

1. Add a diagnostic mode at local S1023 that compares eager and Inductor for
   prepared Q/K/V, the prepared FA4 O/LSE call with identical operands, the
   output projection with identical operands, and the complete layer. If the
   prepared FA4 result itself violates its frozen policy, reject before
   implementing a larger boundary.
2. Add local/global tensor-explicit whole-layer custom ops with shape-only fake
   implementations and fresh output/FP32-LSE storage. Their real bodies
   validate every tensor/static fact and reproduce the pinned operation order:
   projections, FP32 RMS `pow(-0.5)`, family rotary, distinct prepared K/V,
   retained FA4 forward, contiguous flatten, and output projection.
3. Extend the one pinned Transformers patch only enough for the exact project
   attention interface to expose this early compile-layer hook before Q/K/V
   projection. Other attention implementations and eager execution must be
   byte-for-byte behaviorally unchanged.
4. Keep the prepared custom ops for direct/reference diagnostics and as the
   retained eager implementation substrate. Do not modify a CuTe kernel,
   backward path, mask predicate, dependency revision, or tolerance.
5. Regenerate and hash-lock the exact cumulative Transformers patch; preserve
   strict two-file status unless the declared early hook necessarily changes a
   third pinned file, in which case predeclare that exact file before applying
   the patch. No hidden monkeypatch, module-object registry, captured mutable
   module state, or install-error suppression is allowed.

The implementation may stop after step 1 if localization falsifies the
hypothesis. It may not substitute a post-observation tolerance increase.

## Correctness evidence

- [x] local S1023 localization records eager/Inductor error independently for
      Q, K, V, prepared FA4 O/LSE with identical operands, output projection
      with identical operands, and whole-layer output
- [x] identical prepared operands retain the frozen local O/LSE policy; any
      observed drift is attributed only where the boundary evidence proves it
- [ ] whole-layer real bodies are bitwise equal to the exact pinned eager layer
      for local/global S1 and S33 before compilation
- [x] fake implementations return fresh symbolic BSHD whole-layer output and
      FP32 LSE without entering a real body
- [x] `torch.library.opcheck` passes schema, alias, FakeTensor, dynamic, and AOT
      checks for both family ops and every tensor argument is explicit
- [x] source and tests prove no module object, global module registry, mutable
      weight cache, or implicit training/config state participates in the ABI
- [ ] actual pinned layers 0 and 5 compile fullgraph under eager and default
      Inductor for S1, S32, S33, S1023, and S1024
- [ ] every compiled whole-layer output is bitwise equal to pinned eager output
      at every family/backend/length; no tolerance substitutes for this gate
- [ ] direct prepared local/global O and FP32 LSE pass their unchanged frozen
      reference policies at every length
- [ ] public graphs are bounded exactly to S1 and S>1, zero breaks, and contain
      the expected whole-layer project custom-op node
- [ ] the private size-oblivious diagnostic uses one graph, zero breaks, and is
      not presented as a public requirement
- [ ] repeated/reordered sweeps create no additional graph, Inductor, or FA4
      application-key class
- [ ] default and nondefault CUDA streams pass and repeat bitwise
- [ ] reset positions, B2, unequal lengths, offsets, padding, vision/document
      metadata, explicit sequence metadata, active gradients, altered masks,
      wrong weights, aliases, and config/module mutations fail closed
- [ ] all 16 EXP-0018 real DynamicCache/StaticCache cases still reject before
      layer/cache/custom-op/backend entry with identical cache state
- [ ] the existing eager suite and exact EXP-0016 StaticCache matrix remain
      unchanged and passing
- [ ] a pristine pinned Transformers checkout accepts the revised patch once,
      rejects a second application, reverse-checks, and matches every lock

## Synchronization and generated code

- [ ] memcheck, synccheck, and racecheck pass one default-Inductor local S1024
      whole-layer case with unfiltered output retained
- [ ] memcheck, synccheck, and racecheck pass one default-Inductor global S1024
      whole-layer case with unfiltered output retained
- [ ] any project-symbol sanitizer filter is verified against the retained
      object before filtered output is interpreted
- [ ] retained local/global FA4 main-object hashes, PTX/cubin/SASS bytes,
      register/spill/SMEM/TMEM signatures, and launch geometry are unchanged
- [ ] whole-layer outer-op graphs/cache keys are inventoried separately from
      FA4 application keys and remain within the declared S1/S>1 bound

No absence of a compiler error or graph break substitutes for numerical,
reference, sanitizer, or codegen evidence.

### Localization result

The fixed local/Inductor/S1023 diagnostic passed on implementation revision
`f3e9fa4dd2c89907c0d549df97fe645ad35b2d8f` and confirmed the predeclared
outer-drift hypothesis:

- compiled Q, K, and V were non-bitwise with maximum absolute errors
  `0.03125`, `0.03125`, and `0.0078125` respectively;
- the prepared FA4 custom-op output and FP32 LSE were both bitwise equal for
  identical operands and passed their unchanged references with maximum
  absolute errors `0.015625` and `0.000030517578125`;
- the isolated output projection was bitwise equal for identical operands;
- compiled Q/K/V passed through eager prepared FA4 plus eager output
  projection reproduced maximum absolute error `0.0703125`;
- the complete compiled layer reproduced maximum absolute error `0.0703125`
  and mean absolute error `0.007422682363539934`, outside EXP-0018's frozen
  tolerance;
- QKV, prepared FA4, output projection, and whole-layer diagnostics produced
  one, one, one, and two bounded captures respectively, with zero graph
  breaks and one added FA4 application-key class.

Artifact:
`agent_space/remote-h100-exp0019/h100-exp0019-outer-drift-localization.json`,
SHA256 `647a7669cbb373bba986a2e63391410ff31706d4d049ff6b4fa70842ae861b6a`.
The localization authorizes step 2 of the single change; it is not compiler
acceptance by itself.

### Whole-layer candidate result

Implementation revision
`0adfc0a2fe9df85e01b91d1bc846acf5d2f6ae12` was rejected without
running the remaining positive matrix or sanitizer gates:

- the H100 custom-op and compiler-integration test selection passed `68`, with
  one intentional old-PyTorch skip;
- the local/eager S1/S33 smoke passed direct whole-layer output and FP32-LSE
  bitwise transport, unchanged prepared O/LSE references, exact S1/S>1 public
  graph classes, the private one-graph diagnostic, reset-position/cache
  rejection, and default/nondefault-stream bitwise repeatability;
- default Inductor proved that a metadata-only `detach`, even followed by a
  value-identical clone, may be collapsed at an opaque consumer and pass all
  six live module weights with `requires_grad=True`; the raw whole-layer op
  correctly rejected those operands under this experiment's declared ABI;
- a family-specific opaque ownership snapshot produced fresh detached weights
  and passed FakeTensor, real-H100, and opcheck coverage, but local/Inductor/S1
  then invoked the backend twice for two structurally identical S1 graphs;
- two backend captures for the single S1 coordinate violate the declared
  exactly-one-S1-class bound. Identical node lists do not authorize counting
  two compiler cache entries as one.

Artifacts:

- `agent_space/remote-h100-exp0019/h100-check-exp0019.json`, SHA256
  `4ee189fd65b8377723f8903b7bac3fd56537375a50029e6a0ce7c6594323fc72`;
- `agent_space/remote-h100-exp0019/h100-exp0019-whole-local-eager-smoke.json`,
  SHA256 `377208d22391a00fe0cd6de6efb29befc2799fc1a32fe5492ac3bff2c12b8d31`;
- `agent_space/remote-h100-exp0019/h100-exp0019-whole-compile-matrix-failed.json`,
  SHA256 `6dd011957dc0cc85849318780a4f1595e6a34fb33a86d5b9f1473fe6d400dd89`;
- `agent_space/remote-h100-exp0019/h100-exp0019-inductor-s1-reject.json`,
  SHA256 `50603e7fa6efe060a49ddbf2ae10589689d3345dc2dcf06d47b331dfff88f799`.

The eager numerical evidence supports the whole-layer arithmetic hypothesis,
but the compiler-boundary hypothesis as declared is false. No tolerance,
graph bound, or requires-grad condition was relaxed after observation.

## Measurement

- Clock/power state: not applicable; correctness-only experiment
- Hot/cold L2: not run
- Warmup/repetitions/statistic: not run
- Semantically equivalent baseline: uncompiled exact pinned attention layer
  plus the retained FP32 prepared-attention reference

No timing, speedup, whole-model, compiled-cache, B300, or cross-architecture
claim is authorized.

## Decision

**REJECT.** The local/eager numerical smoke passed, but default Inductor
required either live requires-grad weight metadata at the opaque boundary or
an ownership snapshot that produced two S1 backend captures. Both choices
conflict with a frozen EXP-0019 condition. A later experiment may predeclare
the inference-only weight-metadata contract before removing the snapshot; it
must retain bitwise equality, mask provenance, cache immutability, and the
S1/S>1 graph bound.

## Record

Do not append a result until the declared H100 evidence is complete. Use:

```bash
python scripts/record_result.py EXP-0019 \
  --kernel h100-whole-layer-opaque-compiler \
  --arch sm_90 --decision <accept|reject> \
  --hypothesis '<exact hypothesis above>' --bench <jsonl>
```
