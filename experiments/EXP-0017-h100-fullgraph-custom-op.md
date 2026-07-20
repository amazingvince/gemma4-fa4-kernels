# EXP-0017: H100 no-cache fullgraph custom-op boundary

- Date / author: 2026-07-19 / Codex
- Kernel family: Transformers integration over retained local-d256 and
  global-d512 fixed forward paths
- Architecture: sm_90
- Starting revision: `e72b2a20d271397faa75234d91ceb5482a63584b`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Starting H100 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `807650a592d428fbfb2f7cacfd43622b008b5ffb68a815a02c37328c5b5a6935`
- Exemplars:
  - pinned PyTorch `torch.library.custom_op`, `register_fake`, `opcheck`, and
    Dynamo/Inductor APIs;
  - pinned Transformers
    `.upstream/transformers/src/transformers/models/gemma4/modeling_gemma4.py`;
  - the accepted project integration and fixed H100 paths at the starting
    revision.

## Invariant changed

No model, mask, numerical, ownership, CuTe tile, pipeline, synchronization, or
backward invariant changes. Retain exact BF16 prepared Q/K/V, scale 1.0,
distinct K/V, FP32 LSE, local sliding causality, global causality, and the exact
32Q/16KV/d256 and 32Q/4KV/d512 shapes.

Change only the no-cache inference compiler boundary. The ordinary eager
dispatcher performs storage, mask-closure, scalar, overlap, packing, and route
proofs that correctly fail on FakeTensor or cause Dynamo graph breaks. Do not
weaken those proofs or pretend they are traceable. Instead, a pinned-origin
mask token may admit one deliberately narrow compile branch before eager
dispatch. That branch invokes a real `torch.library.custom_op` whose real H100
implementation reuses the already accepted fixed FA4 route and whose fake
implementation returns only symbolic output metadata.

The accepted candidate may expose separate local/global opaque forward ops.
Each consumes prepared BHSD Q/K/V plus the runtime position tensor and returns
fresh, non-aliasing BSHD O and FP32 LSE. The real body revalidates every runtime
shape, dtype, device, stride, position, and no-grad requirement before launch.
The fake body must never run a CuTe kernel or read tensor data.

## Declared envelope

This experiment admits only:

- H100/SM90, BF16 inputs, FP32 LSE, scale exactly 1.0;
- actual pinned `Gemma4TextAttention` layer 0 (local) and layer 5 (global);
- no cache, inference/no active backward, B1, text-only self-attention;
- equal query/key lengths with `1 <= S <= 1024`;
- exact zero-based contiguous `position_ids == arange(S)`;
- no padding, vision/document metadata, explicit cu-seqlens/maxima, offsets,
  arbitrary 4D mask, fallback, export, or CUDA-graph claim.

Padding, packed/varlen framework input, reset or noncontiguous positions,
vision/document masking, B>1, unequal Q/K lengths, cache objects, active
gradients, and every unproven mask origin remain fail-closed. Existing eager
fixed, padded, packed, multimodal, and StaticCache behavior must remain
unchanged. Compiled StaticCache and compiled training/autograd are separate
future experiments; a forward custom op does not inherit the inner FA4
autograd function.

## Hypothesis

A pinned-origin, no-cache local/global `torch.library.custom_op` boundary with
a shape-only fake implementation will let the actual pinned Gemma 4 attention
layers execute as one full Dynamo graph under default Inductor for every
declared length, with bitwise eager/compiled layer output, reference-valid
prepared O/FP32 LSE, zero graph breaks, bounded compiler and FA4 cache classes,
and no changed CuTe main-kernel object.

Falsification is any missing opaque node, fake-body real launch, graph break,
data-dependent trace guard, output/input alias, silent active-gradient path,
unsupported mask/cache admission, non-bitwise eager/compiled layer result,
reference-policy failure, wrong O/LSE dtype or shape, unbounded length-derived
graph or FA4 application keys, changed retained main-object bytes/resources,
or project-kernel sanitizer finding.

## Single change

Add one bounded no-cache inference compiler ABI:

1. mark only plans produced by the registered pinned Gemma mask callback with
   an internal local/global compile-origin token;
2. before ordinary eager dispatch, recognize only the declared compile scope
   using Python/static facts and call the corresponding opaque op;
3. validate runtime tensor facts and zero-based contiguous positions inside
   the real opaque body, then invoke the retained fixed FA4 route;
4. register a symbolic fake implementation with fresh O/LSE outputs;
5. keep imports safe and execution explicitly unavailable when the installed
   PyTorch lacks the required custom-op APIs.

Do not use `torch._dynamo.disable`, `allow_in_graph`, an identity decorator, or
a decomposition that exposes the eager dispatcher. Do not change a CuTe
kernel, FA4 patch, dependency revision, mask predicate, backward path, or
performance setting.

## Correctness evidence

- [ ] imports remain safe on unsupported older local PyTorch and execution
      fails closed rather than becoming non-opaque
- [ ] direct local/global fake execution returns symbolic BSHD O and FP32 LSE
      without entering a real body
- [ ] `torch.library.opcheck` passes schema, alias, FakeTensor, and dynamic/AOT
      checks for both ops
- [ ] actual pinned layers 0 and 5 compile with `fullgraph=True` under both the
      eager backend and default Inductor
- [ ] captured FX/Inductor graphs contain the expected project custom-op node,
      one full graph, and zero graph breaks
- [ ] one dynamic compiled callable per layer family passes S1, S32, S33,
      S1023, and S1024 without a length-derived graph class
- [ ] eager and compiled full-layer outputs are bitwise equal at every length
- [ ] direct prepared O and FP32 LSE pass the frozen local/global references
- [ ] default and nondefault CUDA streams pass and repeat within the declared
      deterministic forward contract
- [ ] forged/direct plans, padding, arbitrary masks, vision/document metadata,
      reset positions, B2, unequal lengths, cache, offsets, explicit sequence
      metadata, and active gradients fail closed
- [ ] all accepted eager and EXP-0016 StaticCache tests remain unchanged

The compiler branch may not reuse the ordinary eager mask-closure equality or
future-mask scalar checks on FakeTensor. The internal origin token is necessary
but not sufficient: every static and runtime scope guard above remains
mandatory. No absence of a graph break or crash is accepted as numerical
evidence.

## Synchronization and generated code

- [ ] memcheck passes one compiled local S1024 case
- [ ] synccheck passes the project-owned FA4 kernel in that local case
- [ ] racecheck passes the project-owned FA4 kernel in that local case
- [ ] memcheck passes one compiled global S1024 case
- [ ] synccheck passes the project-owned FA4 kernel in that global case
- [ ] racecheck passes the project-owned FA4 kernel in that global case
- [ ] outer compiler artifacts are inventoried separately from FA4 application
      keys and remain bounded across the declared length matrix
- [ ] retained FA4 main-object hashes, PTX/cubin/SASS bytes, and resource
      signatures are unchanged

Unfiltered tooling may traverse Inductor, projection, or other vendor kernels;
any filter used to isolate the project FA4 symbol must be verified from the
retained object and the unfiltered result must remain disclosed.

## Measurement

- Clock/power state: not applicable; correctness-only experiment
- Hot/cold L2: not run
- Warmup/repetitions/statistic: not run
- Semantically equivalent baseline: uncompiled pinned eager layer plus the
  project FP32 prepared-attention reference

No performance measurement or speed claim is authorized by this experiment.

## Decision

**PENDING.** Accept only if the complete declared compiler, numerical,
negative-scope, cache/version, sanitizer, and generated-code matrix passes on
the pinned H100 environment. Backend-eager compilation alone, a direct custom
op call, or a toy module does not establish actual pinned-layer compatibility.

Compiled StaticCache, compiled training/autograd, padded or packed framework
inputs, multimodal compilation, export, CUDA graphs, maximum-context global
forward, performance, and B300 remain outside this decision.

## Record

No result record exists while the decision is pending.
