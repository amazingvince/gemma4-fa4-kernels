# EXP-0018: H100 mask-boundary compiler provenance

- Date / author: 2026-07-20 / Codex
- Status: **PREDECLARED; no result recorded**
- Kernel family: Transformers integration over retained local-d256 and
  global-d512 fixed forward paths
- Architecture: sm_90
- Starting revision: `edf5ff404e6dbc34c22a399fc68a53eeb3b3503b`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Starting Transformers patch:
  `patches/transformers/0001-gemma4-forward-vision-block-ids.patch`, SHA256
  `773950a1f1feb04f5f2e6a1d66f8953ff8905e8ca9391f804089f169da59b671`
- Starting H100 FA4 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Starting environment-policy hash:
  `807650a592d428fbfb2f7cacfd43622b008b5ffb68a815a02c37328c5b5a6935`
- Exemplars:
  - pinned Transformers
    `src/transformers/masking_utils.py:create_causal_mask` and
    `create_sliding_window_causal_mask`;
  - pinned Gemma 4 text layers 0 and 5;
  - the retained EXP-0016 eager StaticCache route and retained EXP-0017
    local/global custom ops.

## Invariant changed

No model, mask predicate, numerical, tensor-ownership, CuTe tile, pipeline,
synchronization, backward, or performance invariant changes. Retain exact BF16
prepared Q/K/V, scale 1.0, distinct K/V, FP32 LSE, local sliding causality,
global causality, and the exact 32Q/16KV/d256 and 32Q/4KV/d512 shapes.

Change only the compiler provenance boundary that EXP-0017 proved
insufficient. A compiler origin must be minted at the pinned upstream mask
builder, before a Gemma attention layer can update a cache, rather than inferred
from a callable after the registered mask callback receives it. The builder
must transport the exact `past_key_values` object and a module-private,
identity-checked origin for the exact unmodified plain full-causal or
sliding-causal path. The origin is bound to the mask interface selected by the
builder so an arbitrary registered wrapper cannot forward the same arguments
and acquire project compiler provenance.

Under `torch.compile`, the exact project mask callback rejects every non-null
cache object during mask construction, before any text-layer invocation,
`Cache.update`, or project custom-op invocation. With a null cache it may mint
project compiler provenance only from the matching private upstream
local/global origin. Eager execution does not use this compiler capability and
must preserve the accepted EXP-0016 StaticCache behavior exactly.

## Declared envelope

The accepted positive scope remains deliberately narrow:

- H100/SM90, BF16 inputs, FP32 LSE, scale exactly 1.0;
- actual pinned `Gemma4TextAttention` layer 0 (local) and layer 5 (global);
- inference with no active backward, B1, text-only self-attention;
- equal query/key lengths with `1 <= S <= 1024`;
- exact zero-based contiguous `position_ids == arange(S)`;
- no cache, padding, vision/document metadata, explicit cu-seqlens/maxima,
  offsets, arbitrary 4D mask, custom mask overlay, fallback, export, or
  CUDA-graph claim.

The exact upstream plain origin excludes a non-null attention mask,
`or_mask_function`, `and_mask_function`, `block_sequence_ids`, a bidirectional
path, or any user-supplied mask composition. The standard packed-sequence
helper reached from exact zero-based contiguous positions may remain in the
resulting callable, but it cannot itself mint provenance; the project runtime
still proves the positions and every existing static/runtime scope guard.

Positive full-layer tests use public PyTorch 2.8
`torch.compile(fullgraph=True, dynamic=True)` with both `backend="eager"` and
the default Inductor backend. Each family may use at most two bounded public
graph classes: one for S1 and one shared by every S>1 case. If two classes are
observed, their guards must be exactly that S1/S>1 split; S32, S33, S1023, and
S1024 may not create further classes. A one-graph result obtained only with the
private scoped
`torch.fx.experimental._config.backed_size_oblivious=True` setting is
diagnostic evidence, not an acceptance requirement and not a public compiler
claim.

Padding, packed/varlen framework inputs, reset or noncontiguous positions,
vision/document masking, B>1, unequal Q/K lengths, cache objects, active
gradients, and every unproven mask origin remain fail-closed for the compiler
route. Compiled cache execution, compiled training/autograd, maximum-context
global forward, performance, and SM103/B300 are outside this experiment.

## Hypothesis

If the pinned causal and sliding-causal mask builders transport the exact cache
object plus an interface-bound private plain-mask origin, then the project
callback will reject actual compiled cache requests before mutation, admit only
the exact no-cache local/global mask semantics, and run the public H100
fullgraph matrix within the declared two-class bound while meeting the frozen
prepared-attention references and the predeclared full-layer BF16 tolerance.

Falsification is any cache mutation or custom-op call before cache rejection;
compiler provenance minted through an arbitrary registered wrapper, direct
callback, altered mask, or wrong-family origin; a graph break; more than the
S1/S>1 public graph classes; a missing project custom-op node; full-layer
eager/compiled error outside `atol=0.0625, rtol=0.02`; a direct prepared O/LSE
reference-policy failure; unbounded compiler or FA4 caches; an EXP-0016 eager
regression; a strict patch/hash mismatch; changed retained FA4 generated code;
or a project-kernel sanitizer finding.

## Single change

Extend the one pinned Transformers patch at the mask boundary; do not add
cache or provenance transport to the attention-interface call:

1. In `create_causal_mask` and `create_sliding_window_causal_mask`, preserve
   the exact `past_key_values` object and create a module-private origin only
   for the exact plain full-causal or sliding-causal construction. Bind that
   origin to the exact mask interface selected by the builder and pass both as
   private mask-interface kwargs.
2. In the exact registered project mask callback, compare the received cache
   object and origin by the required identity. During compilation, reject any
   non-null cache immediately. Mint the existing local/global compiler origin
   only when the cache is `None`, the private family origin matches, the bound
   recipient is the exact project callback, and all retained scope checks pass.
3. Preserve the public/direct callback as non-authoritative. A registry entry
   replaced by an arbitrary forwarding wrapper, including one that forwards
   every kwarg to the project callable, must not mint compiler provenance.
4. Keep the real/fake custom-op ABI, FA4 dispatch, kernels, backward paths,
   numerical policies, and eager dispatch unchanged. The private provenance
   kwargs must be harmless to all other pinned mask interfaces.
5. Regenerate the existing Transformers patch against a pristine checkout of
   the pinned revision, update its SHA256 in `upstream.lock.json`, the H100
   environment policy, and the file manifest, and teach strict environment
   verification the exact revised patch file set. Do not add a second
   untracked patch or hide patch/install failures.

This is one conceptual change: move compiler authority to an exact,
cache-aware upstream mask-construction boundary. Test and lock-file changes
exist only to prove and pin that boundary.

## Correctness evidence

- [x] a pristine pinned Transformers checkout accepts the revised patch with
      `git apply --check`, applies it once, rejects a second application, and
      reverse-checks cleanly
- [x] the revised patch SHA256 agrees exactly across the patch file,
      `upstream.lock.json`, `configs/env/h100-compatible.env`, and the file
      manifest; strict H100 environment checking reports only the declared
      patched upstream files
- [x] imports remain safe on unsupported older local PyTorch and execution
      fails closed rather than weakening provenance or opacity
- [x] direct local/global FakeTensor execution returns fresh symbolic BSHD O
      and FP32 LSE without entering a real custom-op body
- [x] `torch.library.opcheck` passes schema, alias, FakeTensor, and dynamic/AOT
      checks for both local and global opaque ops, including S1 and a
      non-singleton case
- [x] actual empty and nonempty `DynamicCache` objects are rejected under
      `fullgraph=True` for both layer families and both compiler backends
      during mask construction, before a layer, `Cache.update`, or project
      custom op is entered
- [x] actual empty and nonempty `StaticCache` objects meet the same compiled
      rejection requirement for both layer families and compiler backends
- [x] cache identity at the callback is the exact caller object, and every
      rejected cache retains identical logical length, counters, tensor
      storage identity, contents/digests, and allocation state before and after
      the attempt
- [x] explicit layer-entry, cache-update, and custom-op counters remain zero
      for every rejected cache case
- [x] an arbitrary registered-wrapper callable cannot mint project compiler
      provenance even when it forwards all private kwargs to the project
      callback; direct/public and wrong-family origins also fail closed
- [ ] altered attention masks, custom `or_mask_function`/`and_mask_function`,
      block/vision metadata, padding, reset positions, B2, unequal lengths,
      offsets, explicit sequence metadata, and active gradients cannot mint or
      use the compiler origin
- [ ] actual pinned local layer 0 and global layer 5 compile with
      `fullgraph=True, dynamic=True` under both `backend="eager"` and default
      Inductor, with every captured graph full, zero breaks, and the expected
      family custom-op node
- [ ] one public dynamic callable per family/backend passes S1, S32, S33,
      S1023, and S1024 with at most two graph classes, split only as S1 versus
      all S>1 lengths; a repeated, reordered sweep creates no new class
- [ ] each compiled full-layer output passes
      `torch.testing.assert_close(eager, compiled, atol=0.0625, rtol=0.02)`;
      bitwise equality is neither required nor claimed
- [ ] direct prepared local/global O and FP32 LSE retain and pass their frozen
      project reference policies without tolerance changes after observation
- [ ] default and nondefault CUDA streams pass the complete positive matrix and
      repeat within the retained forward numerical policies
- [ ] compiler artifacts and FA4 application keys remain bounded by the
      predeclared family/backend and S1/S>1 classes, with no per-length cache
      growth
- [ ] the existing eager suite, all accepted integration paths, and the exact
      EXP-0016 empty/nonempty StaticCache prefill/decode matrix remain
      unchanged and passing

The cache negatives must use real pinned Transformers cache classes and valid
empty/nonempty states; a stand-in object or callback-only test is not evidence.
Snapshot each cache before compiling/calling and after the expected rejection.
For nonempty caches, establish valid prior K/V state before taking the snapshot.
Do not accept an exception raised after a layer, cache update, allocation, or
custom-op counter advances.

The full-layer comparison is deliberately BF16-tolerant because EXP-0017
observed a non-bitwise Inductor result. The tolerance above is fixed before
this run. It does not replace the existing direct prepared O/LSE policies and
may not be loosened after seeing results.

## Synchronization and generated code

- [ ] memcheck passes one default-Inductor local S1024 case
- [ ] synccheck passes the project-owned FA4 kernel in that local case
- [ ] racecheck passes the project-owned FA4 kernel in that local case
- [ ] memcheck passes one default-Inductor global S1024 case
- [ ] synccheck passes the project-owned FA4 kernel in that global case
- [ ] racecheck passes the project-owned FA4 kernel in that global case
- [ ] unfiltered sanitizer output is retained, and any project-symbol filter is
      verified against the retained object before interpreting filtered output
- [ ] retained local/global FA4 main-object hashes, PTX/cubin/SASS bytes,
      register/spill/SMEM/TMEM signatures, and launch geometry are byte-for-byte
      unchanged from the starting revision
- [ ] only bounded outer Dynamo/Inductor artifacts change; their graph guards
      and cache keys are inventoried separately from FA4 application keys

No sanitizer pass or compiler capture is allowed to stand in for numerical
evidence. Because this experiment changes no kernel source or constexpr, any
changed retained FA4 main object falsifies the unchanged-code hypothesis and
must be investigated rather than normalized as expected variance.

## Measurement

- Clock/power state: not applicable; correctness-only experiment
- Hot/cold L2: not run
- Warmup/repetitions/statistic: not run
- Semantically equivalent baseline: uncompiled pinned eager layer plus the
  project FP32 prepared-attention reference

No timing table will be populated. No speed, kernel-performance, B300, or
cross-architecture claim is authorized by this experiment.

## Decision

**REJECTED** for implementation revision
`e9a5af6f88f8d2be74256da1c89a8926d6f89fdd`. The mask-boundary change passed
its provenance and cache-safety objectives, but the complete positive matrix
hit the experiment's frozen numerical falsifier. The public default-Inductor
local layer at S1023 differed from pinned eager output by maximum absolute
error `0.0703125` and mean absolute error `0.00742268236`, exceeding the
predeclared `atol=0.0625, rtol=0.02` comparison. The tolerance was not widened
after observation.

The following bounded evidence passed and remains useful:

- the revised two-file Transformers patch applied exactly once to a pristine
  pinned checkout, rejected a second application, reverse-checked, and matched
  SHA256 `c812937e5a554c1887c2c16a0808f24437cb8b60b561e9fd5eacaa13fb277780`;
- the strict H100 environment reported both exact upstream patch stacks and no
  warnings or errors;
- all 16 local/global x eager/Inductor x Dynamic/Static x empty/nonempty real
  cache cases rejected before layer, cache-update, custom-op, or compiler
  backend entry with unchanged cache identity, contents, allocation, and
  logical length;
- the local/eager S1 and S33 smoke passed FakeTensor, opcheck, direct prepared
  O/LSE references, the public S1/S>1 graph split, the private one-graph
  diagnostic, zero breaks, and exact eager/compiled full-layer output;
- the pre-H100 local repository suite reported **383 passed, 96 skipped**, and
  the focused H100 integration/probe suite reported **104 passed, 1 skipped**.

The complete local/global positive ladder stopped at the first numerical
falsifier. The remaining global/Inductor cases, complete stream matrix,
sanitizers, compiler/codegen inventory, and retained-object comparison were
not run. No compiler compatibility, sanitizer, codegen, performance,
compiled-cache, or B300 acceptance claim is made.

## Record

The schema-validated rejection record was captured on the H100 with:

```bash
python scripts/record_result.py EXP-0018 \
  --kernel h100-mask-boundary-compiler-provenance \
  --arch sm_90 --decision reject \
  --git-sha e9a5af6f88f8d2be74256da1c89a8926d6f89fdd \
  --bench agent_space/h100-exp0018-torch-compile.json \
  --profile agent_space/h100-check-exp0018.json \
  --hypothesis '<pinned mask-boundary hypothesis above>'
```

Retained artifacts:

- `agent_space/remote-h100-exp0018/h100-check-exp0018.json`, SHA256
  `3cef71f936c264dfebc8d521ca61b666dec8793b8152ca82a3f7c05bd52ecc4a`;
- `agent_space/remote-h100-exp0018/h100-exp0018-cache-negative.json`, SHA256
  `08362f1320d192a6476a070a5765fca1925cb276523bc75369c1c00230c96c43`;
- `agent_space/remote-h100-exp0018/h100-exp0018-local-eager-smoke.json`, SHA256
  `11c2554744e0db26a8ff8e1b8c2e1ea0c4e4cae5f95e260a60bcc4679d73250f`;
- `agent_space/remote-h100-exp0018/h100-exp0018-torch-compile.json`, SHA256
  `7e36c2132434428606a5a8c4bf835b3be256c527e101bf0948ee626e211937de`.
