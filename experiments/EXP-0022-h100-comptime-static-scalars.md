# EXP-0022: H100 compile-time static scalar guards

- Date / author: 2026-07-20 / Codex
- Status: **PREDECLARED — no candidate result yet**
- Kernel family: pinned Transformers local-d256 and global-d512 attention
  layers over the retained FA4 forward paths
- Architecture: sm_90
- Starting revision: `3c2c68163e5491f60af0af6ac7e86178c75254fe`
- Model-contract lock SHA256:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Transformers patch SHA256:
  `ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79`
- H100 FA4 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Compiler source anchor: pinned
  `torch/_dynamo/comptime.py`, SHA256
  `94296e5c1e942dc1e3a0d0f8ced799f9620ac93ae17ea4156659f138abdd9971`

## Prior falsifier

EXP-0020 passed live source Parameters directly to the whole-layer custom op,
and EXP-0021 additionally routed all eleven float-valued config/module fields
through CPU FP64 scalar tensors. Both reached the local/eager/S1 custom-op and
reference gates, but stock Inductor invoked the backend twice. The first
capture retained backed `SymFloat` inputs and raised
`TensorifyScalarRestartAnalysis`; the second specialized those values and
returned. EXP-0021 was rejected and its added scalar-tensor transport was
removed in `3c2c681`.

The pinned PyTorch 2.8 `torch._dynamo.comptime` module describes itself as a
public compile-time interface. `ComptimeVar.force_static()` evaluates a
symbolic scalar expression, and its documented contract is: “Forces that a
value is static, inducing a guard on its specific value.” This is materially
different from EXP-0021's runtime tensor transport and from
`assume_constant_result`: the original source value remains guarded, while no
scalar tensor is added to the graph or whole-layer ABI.

## Invariant changed

Change only the compiler ownership of the eleven exact Python float values
that already gate the pinned no-cache whole-layer route:

```text
0  config.rope_parameters[sliding_attention].rope_theta = 10000.0
1  config.rope_parameters[full_attention].partial_rotary_factor = 0.25
2  config.rope_parameters[full_attention].rope_theta = 1000000.0
3  config.attention_dropout = 0.0
4  config.final_logit_softcapping = 30.0
5  config.rms_norm_eps = 1e-6
6  module.scaling = 1.0
7  module.attention_dropout = 0.0
8  module.q_norm.eps = 1e-6
9  module.k_norm.eps = 1e-6
10 module.v_norm.eps = 1e-6
```

Each source value is first bound to a named local, passed once to public
`comptime.force_static`, and then compared exactly with the locked value.
Eager validation remains the existing exact comparison. Structural
validation, source Parameter ownership, family geometry, mask origin, cache
rejection, inference-only policy, custom-op ABI, FA4 application keys, and all
numerical contracts remain unchanged.

No value becomes a tuning key, tensor operand, constant-result assertion, or
unchecked snapshot. A guard failure after later mutation must retrace only
far enough to reject the invalid value; it must not reach the backend or FA4.

## Declared envelope

Retain EXP-0020's positive scope: H100/SM90, BF16, B1 text-only fixed
self-attention, exact zero-based positions, `1 <= S <= 1024`, pinned layers 0
and 5, no cache or shared KV, no vision/document/padding/offset/packed
metadata, and no active gradient. Global K and V remain distinct prepared
operands after their different normalization and rotary treatment.

## Hypothesis

Forcing each of the eleven already-required pinned floats static through the
documented PyTorch 2.8 comptime interface before exact comparison will install
value guards and remove all backed `SymFloat` inputs from the first graph, so
stock Inductor reaches the local/Inductor/S1 backend exactly once while every
later one-field mutation rejects before backend and FA4 entry.

Falsification is any first-capture `SymFloat` placeholder, scalar extraction
or tensorification node; backend-attempt count other than one; a
`TensorifyScalarRestartAnalysis`; any mutation admitted or reaching the
backend/custom op; new backend graph, Inductor cache class, or FA4 application
key for a mutation; restored exact value failing to reuse the original graph;
non-bitwise whole-layer output; prepared O/FP32-LSE reference failure; missing
whole-layer op; snapshot node; graph break; widened shape/cache/mask/gradient
scope; sanitizer finding; changed retained FA4 generated object; tolerance
increase; or any global/private compiler configuration change.

## Single change

1. Add one version-pinned helper that calls
   `torch._dynamo.comptime.comptime.force_static` for a scalar only while
   compiling and otherwise preserves the ordinary eager value.
2. Bind and force-static precisely the eleven fields above before their
   existing exact validation. Do not alter any structural or tensor check.
3. Extend the compiler probe to inventory scalar graph inputs/nodes and run a
   one-at-a-time mutation/restoration sweep with backend, custom-op, Inductor
   cache, and FA4 application-key entry counters.
4. Do not change the whole-layer tensor ABI, use `mark_static`, set
   `specialize_float`, disable tensorification, use
   `assume_constant_result`, add a registry, freeze/monkeypatch upstream
   classes, alter graph counting, or change either retained upstream patch.

## Correctness gates

- [ ] local/Inductor/S1 reaches the user backend exactly once; its first graph
      has no `SymFloat` placeholder, scalar extraction/tensorization/stack
      node, graph break, or snapshot node and does contain the family
      whole-layer custom op
- [ ] local/eager/S1 remains exactly one backend attempt and bitwise equal
- [ ] mutating each source float individually after compiled-callable creation
      rejects before backend and whole-layer/FA4 custom-op entry, adds no
      backend graph, Inductor cache file, or FA4 application key, and restoring
      the exact value reuses the original graph
- [ ] wrong scalar type, NaN, positive/negative infinity, and signed/ordinary
      non-equal finite values reject without reaching the retained op
- [ ] actual pinned layers 0 and 5 compile fullgraph under eager and Inductor
      at S1, S32, S33, S1023, and S1024
- [ ] every compiled output is bitwise pinned-eager equal
- [ ] direct prepared local/global O and FP32 LSE retain frozen references
- [ ] whole-layer fake registration and opcheck retain the complete unchanged
      tensor-explicit ABI
- [ ] public graph/cache classes remain exactly S1 and S>1 with zero breaks;
      the scoped size-oblivious diagnostic remains exactly one graph
- [ ] repeated/reordered and default/nondefault-stream sweeps remain bitwise
      and add no graph, Inductor, or FA4 application class
- [ ] all EXP-0018 cache negatives reject before layer/cache/op/backend entry
      with identical cache state
- [ ] reset positions, B2, unequal lengths, offsets, padding, metadata, shared
      KV, cache, active gradients, altered masks, and non-float module
      mutations reject
- [ ] existing eager and EXP-0016 StaticCache matrices remain passing

## Synchronization and generated code

- [ ] memcheck, synccheck, and racecheck pass local and global S1024 default
      Inductor whole-layer cases with unfiltered output retained
- [ ] retained local/global FA4 main-object hashes, PTX/cubin/SASS bytes,
      resource signatures, and launch geometry are unchanged
- [ ] outer compiler graphs/cache entries and exact scalar guards are
      inventoried separately and stay within the declared S1/S>1 bound

## Measurement

Correctness-only. No timing, speedup, whole-model, compiled-cache, B300, or
cross-architecture claim is authorized.

## Decision

Pending. Run only local/Inductor/S1 first. Reject immediately if its user
backend is invoked other than exactly once or if its captured graph retains
any scalar input/tensorification node. Run the wider matrix only after that
discriminator passes without changing the gates.

## Record

```bash
python scripts/record_result.py EXP-0022 \
  --kernel h100-comptime-static-scalars \
  --arch sm_90 --decision <accept|reject|refine> \
  --hypothesis '<exact hypothesis above>' --bench <jsonl>
```
