# EXP-0020: H100 inference-only module-weight boundary

- Date / author: 2026-07-20 / Codex
- Status: **PREDECLARED; no result recorded**
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
- [ ] local/global snapshot op symbols and graph nodes are absent
- [ ] whole-layer opcheck passes with detached direct operands
- [ ] inference-mode direct calls accept exact requires-grad weights and return
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

## Measurement

Correctness-only. No timing, speedup, whole-model, compiled-cache, B300, or
cross-architecture claim is authorized.

## Decision

**PENDING.** Accept only if every gate passes without changing bitwise
equality, graph bounds, prepared references, provenance, cache immutability,
or the inference-only restriction. Otherwise reject.

## Record

```bash
python scripts/record_result.py EXP-0020 \
  --kernel h100-inference-module-weight-boundary \
  --arch sm_90 --decision <accept|reject> \
  --hypothesis '<exact hypothesis above>' --bench <jsonl>
```
