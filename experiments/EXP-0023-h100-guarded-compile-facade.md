# EXP-0023: H100 guarded tensor-only compile facade

- Date / author: 2026-07-20 / Codex
- Status: **ACCEPT — guarded facade only**
- Kernel family: pinned Transformers local-d256 and global-d512 attention
  layers over the retained FA4 forward paths
- Architecture: sm_90
- Starting revision: `a918120c5a72bc75f1e3839c7e693f8da7ee175f`
- Candidate implementation revision:
  `1756ec0df13d25ac8bd48d9c018ec438402db428`
- Final evidence/probe revision:
  `f592971c09da16fc68db15ad588f94f6e1bde0be`
- Model-contract lock SHA256:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Transformers patch SHA256:
  `ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79`
- H100 FA4 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`

## Prior falsifier

EXP-0020, EXP-0021, and EXP-0022 successively tested exact live float
comparisons, runtime CPU-FP64 tensor transport, and public comptime static
guards inside a raw fullgraph `torch.compile(layer)` trace. All three failed
the frozen local/Inductor/S1 one-attempt gate. EXP-0022 removed every visible
`SymFloat`, scalar-extraction, and scalar-tensor node from FX, but the first
identical graph still raised `TensorifyScalarRestartAnalysis`; the second
backend attempt returned. Those candidates were removed.

The evidence means another in-frame scalar representation is not a new
hypothesis. This experiment instead changes the ownership boundary: exact
live-object validation happens outside Dynamo on every user call, and only a
tensor-explicit inner function is compiled.

## Invariant changed

Add one explicitly named project-owned compile facade for the already-declared
H100, inference-only, B1 text, no-cache attention-layer envelope. Construction
binds an actual pinned `Gemma4TextAttention` layer and compiles a module-level
family function whose complete runtime ABI is tensors:

```text
local:
  hidden, cos, sin, position_ids,
  q_weight, k_weight, v_weight, o_weight, q_norm_weight, k_norm_weight

global:
  hidden, cos, sin, position_ids,
  q_weight, shared_kv_source_weight, o_weight,
  q_norm_weight, k_norm_weight
```

The inner function derives zero-reset packed IDs from exact zero-based
`position_ids` and calls the retained local/global whole-layer custom op. It
does not close over or read the module, config, rope dictionaries, Python
floats, mask objects, cache objects, tuning data, or registries.

Before every invocation of that compiled function, the ordinary eager facade
must re-read and exactly validate the pinned config/module structure, all
eleven live float fields, family, layer index, training state, projection/norm
ownership, source Parameter identity/shape/dtype/device/layout, input tensor
contract, no-grad/inference state, and zero-based no-cache text envelope. Only
after validation passes may it increment the compiled-entry counter and call
the tensor-only function with the current live Parameters.

This is a new public facade, not transparent support for raw
`torch.compile(layer)`, a whole-model compiler, arbitrary masks, or compiled
cache. Its API, report, docs, and result record must state that distinction.
No kernel, attention, mask, arithmetic, tolerance, compile key, or performance
invariant changes.

## Declared envelope

H100/SM90, pinned PyTorch 2.8, BF16, global inference/no-grad, B1 text-only
fixed self-attention, exact zero-based contiguous positions, `1 <= S <= 1024`,
pinned exemplar layers 0 and 5, no cache/shared prepared KV, no vision,
documents, padding, offsets, packed resets, fallback, or active gradient.
Global K and V remain distinct prepared operands after different
normalization and rotary treatment.

Supporting the other 58 layer indices, full-model compilation, varlen facade
inputs, or StaticCache is a later separately declared widening. This
experiment must not claim those scopes from the layer-0/layer-5 evidence.

## Hypothesis

Validating all live Python/object state before every facade call and compiling
only the tensor-explicit family function will give stock PyTorch 2.8 Inductor
exactly one backend attempt for local/Inductor/S1, while any later one-field
scalar mutation rejects before compiled-function and FA4 entry and restoring
the exact value reuses the original graph and application key.

Falsification is a backend-attempt count other than one; any Python scalar or
module/config source in the inner FX graph; any
`TensorifyScalarRestartAnalysis`; source mutation admitted or observed only
after compiled entry; new backend graph, Inductor cache class, or FA4
application key for a mutation or restoration; non-bitwise whole-layer output;
prepared O/FP32-LSE reference failure; missing whole-layer op; snapshot node;
graph break; widened layer/mask/cache/gradient scope; sanitizer finding;
changed retained FA4 generated object; tolerance increase; global/private
compiler setting; registry; or wording that promotes the facade to raw
`torch.compile(layer)` compatibility.

## Single change

1. Add `compile_gemma4_fa4_h100_layer(...)` and a small callable facade that
   owns per-call eager validation and the separately compiled tensor-only
   family function.
2. Reuse the existing exact module/config/weight validators outside Dynamo;
   add exact facade-input validation and prohibit all cache/mask/fallback
   arguments by API construction.
3. Compile only the module-level local/global explicit-weight functions with
   `fullgraph=True`, `dynamic=True`, and the selected stock backend. Preserve
   the existing S1/S>1 public graph policy.
4. Extend the probe with a facade mode, backend/compiled-entry counters,
   inner-graph source inventory, exact mutation/restoration sweep, isolated
   Inductor/FA4 cache inventories, and direct prepared references.
5. Do not alter raw `gemma4_fa4_compile_layer`, the retained custom-op ABI,
   the Transformers patch, compiler settings, graph counting, numerical
   policies, or any FA4 kernel.

## Correctness gates

- [x] facade construction rejects wrong module/config class, family/layer,
      training state, weight ownership, device/dtype/layout, or unsupported
      PyTorch/custom-op environment before compilation
- [x] local/Inductor/S1 reaches the user backend exactly once with zero graph
      breaks, no scalar/module/config source, no scalar tensorification node,
      no snapshot node, and exactly one local whole-layer op node
- [x] local/eager/S1 also uses exactly one backend attempt and is bitwise equal
      to the same pinned eager layer
- [x] mutating each of the eleven source floats individually after facade
      construction rejects before compiled-entry/backend/FA4 entry, adds no
      graph, Inductor cache file, or FA4 application key, and restoring the
      exact value reuses the original compiled graph bitwise
- [x] equal non-float types, NaN, positive/negative infinity, and ordinary
      unequal finite values reject before compiled entry
- [x] actual pinned exemplar layers 0 and 5 pass eager and Inductor at S1,
      S32, S33, S1023, and S1024
- [x] every facade output is bitwise equal to the pinned eager layer
- [x] direct prepared local/global O and FP32 LSE retain frozen references;
      K/V are distinct after their locked preparation
- [x] public default dynamic graphs remain exactly the S1 and S>1 classes;
      the scoped size-oblivious diagnostic remains exactly one graph
- [x] repeated/reordered and default/nondefault-stream sweeps remain bitwise
      and add no graph, Inductor, or FA4 application class
- [x] reset positions, B2, unequal lengths, offsets, padding, metadata, shared
      KV, cache, active gradients, altered module structure, and malformed
      tensors reject before compiled entry
- [x] existing eager framework, raw-compile negative, and EXP-0016 StaticCache
      matrices remain unchanged and passing in their documented classifications

## Synchronization and generated code

- [x] memcheck, synccheck, and racecheck pass local and global S1024 default
      Inductor facade cases with unfiltered output retained
- [x] retained local/global FA4 main-object hashes, PTX/cubin/SASS bytes,
      resource signatures, and launch geometry are unchanged
- [x] facade inner graphs/cache entries and FA4 application keys are
      inventoried separately and remain within the exact S1/S>1 bound

## Measurement

Correctness-only. No timing, speedup, whole-model, raw
`torch.compile(layer)`, compiled-cache, B300, or cross-architecture claim is
authorized.

## Evidence

The first local/Inductor/S1 discriminator reached the user backend exactly
once, produced zero graph breaks, contained one whole-layer custom-op node and
no forbidden Python/module/config source, and was bitwise equal to the pinned
eager layer. The expanded local S1 sweep rejected all eleven one-field float
mutations, five malformed scalar forms, twelve tensor/input violations, eight
unsupported API requests, and an altered layer index before compiled entry;
restoration, replay, reorder, and a nondefault CUDA stream reused the same
graph and application key bitwise.

The full public-default matrix passed local/global × eager/Inductor at
S={1,32,33,1023,1024}. It retained exactly two public graph classes per
family/backend (S1 and S>1), zero graph breaks, one whole-layer op, bitwise
facade/eager equality, and one bounded FA4 application key per family reused
across backends. The separately requested, explicitly nondefault
`backed_size_oblivious=True` plus `mark_dynamic` diagnostic passed the same
matrix with exactly one graph per family/backend. It is evidence about that
opt-in policy, not the public default.

The strict H100 environment check reported no warnings or errors. The complete
H100 suite reported `505 passed, 17 skipped, 1 xfailed`; the expected xfail is
the pinned Transformers generic FA4 mask adapter's inability to encode the
local vision future exception. The focused FakeTensor matrix reported
`16 passed, 111 deselected`. Local and global S1024 memcheck reported zero
errors. Project-kernel-filtered synccheck reported zero errors, and filtered
racecheck reported zero hazards for both families; the complete retained logs
record the filter because unfiltered sync/race runs enter cuBLASLt projection
kernels outside the project-owned FA4 kernel.

Generated-code inventory at `f698da05047eca84dd8f4f895f4dc0c25413ef58`
proves unchanged retained launch/device objects. Local host object, PTX, and
inner cubin hashes are bitwise equal to the EXP-0001/EXP-0004 baselines;
global equivalents are bitwise equal to EXP-0002. Both families retain 168
registers, zero stack/local bytes, 1024 static shared bytes, and no SASS
`LDL`/`STL`. The later `f592971` change adds only the opt-in diagnostic.

Hash-locked evidence lives under `agent_space/remote-h100-exp0023/`:

- `h100-exp0023-expanded-local-inductor-s1.json`
- `h100-exp0023-expanded-guarded-facade-matrix.json`
- `h100-exp0023-scoped-one-graph-matrix.json`
- `h100-check-exp0023.json`
- `sanitizers/`
- `codegen-inventory.json` and `codegen/`

Two probe-only corrections preceded the final matrix: cross-backend key
accounting originally counted an already-warmed family key as newly added, and
the first S1 noncontiguous fixture could not be noncontiguous because its
singleton dimension made the transpose contiguous. Neither changed the
candidate, arithmetic, guard policy, or declared acceptance bound.

## Decision

**ACCEPT**, only for the declared project-owned guarded facade on actual pinned
layers 0/local and 5/global, B1 BF16 no-cache text inference/no-grad, exact
zero-based positions, and `1 <= S <= 1024`. The candidate passed the frozen
first discriminator before the wider matrices ran, then passed every declared
correctness, rejection, sanitizer, cache, and unchanged-codegen gate.

This does not accept raw `torch.compile(layer)`, other layer indices, a full
model, compiled `StaticCache`, varlen facade inputs, arbitrary masks, active
gradients, performance, B300, or any scope outside the declared envelope.

## Record

```bash
python scripts/record_result.py EXP-0023 \
  --kernel h100-guarded-compile-facade \
  --arch sm_90 --decision accept \
  --git-sha f592971c09da16fc68db15ad588f94f6e1bde0be \
  --hypothesis '<exact hypothesis above>' \
  --profile agent_space/remote-h100-exp0023/h100-check-exp0023.json
```
