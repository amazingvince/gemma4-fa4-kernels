# EXP-0020: H100 inference-only module-weight boundary

- Date / author: 2026-07-20 / Codex
- Status: **REJECTED on the declared Inductor backend-attempt bound**
- Kernel family: pinned Transformers local-d256 and global-d512 attention
  layers over the retained FA4 forward paths
- Architecture: sm_90
- Starting revision: `256ebfd141c4907290e15b50813a1ad8793b00ff`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Transformers patch SHA256:
  `ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79`
- H100 FA4 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`

## Prior falsifier

EXP-0019 proved that the exact whole-layer arithmetic is bitwise equal under
the eager compiler backend, but its weight-ownership alternatives both failed
a frozen condition. Inductor legally collapsed value-identical detach/copy
operations at an opaque consumer and restored every source Parameter's
`requires_grad=True` metadata. A separate opaque snapshot returned fresh
detached weights, but caused two structurally identical backend captures for
the single S1 coordinate. EXP-0019 was rejected without relabeling those two
cache entries as one class.

## Invariant changed

Change only the dormant module-weight metadata rule for this compile-only
inference path. The exact pinned eval-mode `Gemma4TextAttention` layer 0 or 5
may pass its explicit Q/K/V/output projection and Q/K RMS weights directly to
the family whole-layer custom op while those Parameter tensors retain
`requires_grad=True`, but only when all of the following hold:

- `torch.is_grad_enabled()` is false at the integration hook and real op;
- hidden, cosine, sine, position, and packed-ID tensors do not require grad;
- the exact pinned class/config/layer/projection/norm topology and mask origin
  are proven before entry;
- the custom op has no backward or autograd registration and returns outputs
  that do not require grad;
- weights retain the exact BF16 shapes, device, contiguous layout, distinct
  storage, and Parameter identity required by the locked layer.

Any grad-enabled call, activation requiring grad, altered module, aliased or
non-Parameter weight, cache, shared KV, metadata, or unproven mask still fails
closed. This is not compiled training or backward support. No model, attention,
mask, prepared O/LSE, kernel, synchronization, tolerance, or performance
invariant changes.

## Declared envelope

Retain EXP-0019's positive scope: H100/SM90, BF16, B1 text-only fixed
self-attention, exact zero-based positions, `1 <= S <= 1024`, pinned layers 0
and 5, no cache or shared KV, no vision/document/padding/offset/packed metadata,
and no active gradient. Global K and V remain distinct prepared operands after
their different normalization and rotary treatment.

## Hypothesis

Passing the exact pinned module weights directly across the inference-only
whole-layer opaque boundary will remove the ownership-snapshot recompile while
preserving bitwise pinned-eager output, unchanged prepared O/FP32-LSE
references, exactly S1 and S>1 public graph classes, one private diagnostic
graph, and all retained provenance/cache rejection behavior.

Falsification is any non-bitwise whole-layer result; direct prepared reference
failure; grad-enabled or altered-module admission; missing whole-layer op node;
any weight-snapshot node; more than the declared S1/S>1 public graph classes;
graph break; cache/mask admission; unbounded FA4/compiler application class;
sanitizer finding; changed retained FA4 main object; or tolerance increase.

## Single change

1. Remove the local/global weight-snapshot custom ops and their graph nodes.
2. Pass exact source weight Parameters as explicit whole-layer op arguments.
3. In the real whole-layer op, require global inference mode and reject
   requires-grad activations, while accepting the pinned weights' dormant
   metadata flag under the exact locked shape/layout/storage contract.
4. Preserve the cumulative two-file Transformers patch, family whole-layer
   bodies, prepared ops, FA4 kernels, masks, caches, references, and numerical
   policies unchanged.

## Correctness gates

- [ ] local/global real whole-layer bodies are bitwise equal to pinned eager at
      S1 and S33 before compilation
- [x] local/global snapshot op symbols and graph nodes are absent
- [x] whole-layer opcheck passes with detached direct operands
- [x] inference-mode direct calls accept exact requires-grad weights and return
      no-grad output/LSE; grad-enabled calls reject before FA4 entry
- [ ] altered/aliased/wrong-shape/wrong-dtype weights and requires-grad
      activations reject before FA4 entry
- [ ] actual pinned layers 0 and 5 compile fullgraph under eager and Inductor at
      S1, S32, S33, S1023, and S1024
- [ ] every compiled output is bitwise pinned-eager equal
- [ ] direct prepared local/global O and FP32 LSE retain frozen references
- [ ] public graph/cache classes are exactly S1 and S>1 with zero breaks; the
      private size-oblivious diagnostic is exactly one graph
- [ ] repeated/reordered and default/nondefault-stream sweeps remain bitwise
      and add no graph, Inductor, or FA4 application class
- [ ] all EXP-0018 cache negatives reject before layer/cache/op/backend entry
      with identical state
- [ ] reset positions, B2, unequal lengths, offsets, padding, metadata, shared
      KV, cache, active gradients, altered masks, and module mutations reject
- [ ] existing eager and EXP-0016 StaticCache matrices remain passing

## Synchronization and generated code

- [ ] memcheck, synccheck, and racecheck pass local and global S1024 default
      Inductor whole-layer cases with unfiltered output retained
- [ ] retained local/global FA4 main-object hashes, PTX/cubin/SASS bytes,
      resource signatures, and launch geometry are unchanged
- [ ] outer compiler graphs/cache entries are inventoried separately and stay
      within the exact S1/S>1 bound

## Candidate result

Implementation revision
`f70c828` was rejected at the first discriminating local/Inductor/S1 gate,
without running the wider positive matrix or sanitizer/codegen gates:

- the focused H100 custom-op, compile-integration, and probe tests passed
  `128`, with one intentional compatibility skip;
- the inference-only call boundary is checked before `torch.library.custom_op`
  disables grad mode for its real body; direct local/global calls accept exact
  dormant `requires_grad=True` weights under inference, return no-grad O/LSE,
  and reject a grad-enabled caller before FA4 entry;
- both snapshot custom ops and their symbols are absent; the local eager S1
  graph contains the whole-layer op and no snapshot node;
- local/eager S1 passed bitwise whole-layer output/LSE transport, the unchanged
  prepared O/LSE references, one public and one scoped backend capture, reset
  position and cache rejection, and default/nondefault-stream bitwise replay;
- local/Inductor/S1 invoked the user backend twice. The first identical graph
  reached stock Inductor and raised PyTorch 2.8's internal
  `TensorifyScalarRestartAnalysis`; the second returned successfully. Dynamo
  recorded no guard failure, and both attempts carried the same live dormant
  weight metadata and the same whole-layer op node;
- `TORCH_LOGS=+dynamic` localized that restart to the retained exact
  float-valued config/module attestation, including rope theta, partial rotary
  factor, dropout, logit soft-cap, RMS epsilon, and attention scale. Altering
  how those scalars are specialized is a separate compiler-boundary decision,
  not part of EXP-0020's single weight-metadata change;
- the declared gate counts backend attempts and permits exactly one for S1.
  The internal restart was therefore not relabeled as one graph class after
  observation, and no graph bound or compiler setting was changed.

Artifacts:

- `agent_space/remote-h100-exp0020/h100-check-exp0020.json`, SHA256
  `4ee189fd65b8377723f8903b7bac3fd56537375a50029e6a0ce7c6594323fc72`;
- `agent_space/remote-h100-exp0020/h100-exp0020-local-eager-s1.json`, SHA256
  `ee309681a82695c20cb8ea6b64fbb3ce3d35af654530d510032d2313ad9a0ca9`;
- `agent_space/remote-h100-exp0020/h100-exp0020-local-inductor-s1-reject.json`,
  SHA256
  `f114de70dc6a0dc7c98c18edcf5ae840f0d52a992cb8c00134cd842f5650e2af`.

The strict environment report records both pinned patch stacks applied
exactly and empty warnings/errors. The retained FA4 and Transformers patch
hashes did not change.

## Measurement

Correctness-only. No timing, speedup, whole-model, compiled-cache, B300, or
cross-architecture claim is authorized.

## Decision

**REJECT.** Removing the ownership snapshot and admitting exact live pinned
weights under inference preserved the eager numerical and ABI evidence, but
stock Inductor still required two backend attempts for S1 because its scalar
tensorification pass restarted on the retained float-valued module/config
proof. That violates the frozen exactly-one-attempt condition. A later
experiment may predeclare a scalar-attestation/static-module policy, but it
must retain exact mutation rejection, bitwise equality, mask provenance,
cache immutability, and the S1/S>1 bound.

## Record

```bash
python scripts/record_result.py EXP-0020 \
  --kernel h100-inference-module-weight-boundary \
  --arch sm_90 --decision reject \
  --hypothesis '<exact hypothesis above>' --bench <jsonl>
```
