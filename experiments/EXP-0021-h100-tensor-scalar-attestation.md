# EXP-0021: H100 tensor-explicit scalar attestation

- Date / author: 2026-07-20 / Codex
- Status: **REJECTED on the declared Inductor backend-attempt bound**
- Kernel family: pinned Transformers local-d256 and global-d512 attention
  layers over the retained FA4 forward paths
- Architecture: sm_90
- Starting revision: `99c97bbb2311fe76ff47ea4b7bb9dc280589468b`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Transformers patch SHA256:
  `ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79`
- H100 FA4 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Compiler source anchors: pinned PyTorch
  `torch/fx/passes/_tensorify_python_scalars.py`,
  `torch/_dynamo/variables/builder.py`, and `torch.library.custom_op`

## Prior falsifier

EXP-0020 removed the ownership snapshot and passed exact live dormant source
Parameters through the inference-only whole-layer op. Its local eager S1 and
focused ABI gates passed, but stock Inductor invoked the user backend twice:
the first identical graph raised `TensorifyScalarRestartAnalysis`, and the
second compiled. PyTorch 2.8's scalar pass found already-specialized backed
`SymFloat` inputs created by the exact float-valued config/module checks and
restarted Dynamo so those inputs could be specialized away. The frozen S1
gate permits exactly one backend attempt, so EXP-0020 was rejected.

Pinned source inspection also rules out two tempting shortcuts. Marking an
`nn.Module` class static keeps integer attributes constant but
`VariableBuilder` still wraps floats as `SymFloat` while
`specialize_float=False`. `torch.compiler.assume_constant_result` explicitly
does not validate its soundness. This experiment uses neither mechanism and
does not change any global or private compiler setting.

## Invariant changed

Change only how the exact float-valued pinned module/config contract crosses
the no-cache fullgraph boundary. Compiler-visible Python must not branch on a
source float. Instead, the hook constructs one ordered, contiguous CPU FP64
attestation tensor from eleven 0-D `torch.scalar_tensor` values:

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

The family whole-layer custom op receives that tensor as one additional
explicit operand and checks its shape, CPU device, FP64 dtype, contiguity,
no-grad status, finiteness, and exact ordered values before projection or FA4
entry. Structural validation retains the exact config/module classes, rope
dictionary keys and string modes, layer topology, integer/boolean fields,
projection/norm types, weight identity, mask origin, and every existing
inference-only guard.

A later mutation of any source float must change the runtime attestation and
must be rejected by the opaque boundary before FA4 entry without compiling a
new graph or adding an FA4 application key. The values do not become tuning
keys or generated-code policy. No model, attention, mask, prepared O/LSE,
kernel, synchronization, tolerance, or performance invariant changes.

## Declared envelope

Retain EXP-0020's positive scope: H100/SM90, BF16, B1 text-only fixed
self-attention, exact zero-based positions, `1 <= S <= 1024`, pinned layers 0
and 5, no cache or shared KV, no vision/document/padding/offset/packed
metadata, and no active gradient. Global K and V remain distinct prepared
operands after their different normalization and rotary treatment.

## Hypothesis

Making every pinned Python float feed a tensor-producing operation and
validating the resulting explicit CPU FP64 vector inside the opaque
whole-layer boundary will let stock PyTorch 2.8 Inductor compile each declared
S1/S>1 graph class in one backend attempt while preserving bitwise pinned-eager
output, exact later-mutation rejection, unchanged prepared O/FP32-LSE
references, and all retained provenance/cache guards.

Falsification is any S1 backend-attempt count other than one; scalar
tensorification restart; source-float mutation admitted or observed only
after FA4 entry; new graph or FA4 class for a value mutation; non-bitwise
whole-layer result; direct prepared reference failure; grad-enabled or altered
module admission; missing whole-layer op node; snapshot node; more than the
declared S1/S>1 public graph classes; graph break; cache/mask admission;
sanitizer finding; changed retained FA4 main object; tolerance increase; or a
global/private compiler configuration change.

## Single change

1. Split current pinned config/module validation into structural fields and
   the ordered eleven-float extraction above; preserve every non-float check.
2. Materialize each source float with public `torch.scalar_tensor(...,
   dtype=torch.float64, device="cpu")`, stack exactly once, and pass the
   contiguous vector to the existing family whole-layer op.
3. Extend the local/global whole-layer custom-op schema, fake implementation,
   direct wrapper, and opcheck matrix by that one tensor; validate exact values
   in every real entry before any FA4 call.
4. Do not call `torch._dynamo.mark_static`, set `specialize_float`, disable the
   tensorification pass, use `assume_constant_result`, add a module registry,
   freeze or monkeypatch the upstream classes, change graph counting, or alter
   the retained Transformers/FA4 patches and numerical policies.

## Correctness gates

- [ ] local/global direct whole-layer bodies accept only the exact attestation
      vector and remain bitwise equal to pinned eager at S1 and S33
- [ ] wrong shape/device/dtype/layout/grad flag, NaN/Inf, reordered entries,
      and every one-field value mutation reject before FA4 entry
- [x] whole-layer opcheck passes the complete tensor-explicit ABI
- [ ] actual pinned layers 0 and 5 compile fullgraph under eager and Inductor
      at S1, S32, S33, S1023, and S1024
- [ ] local/Inductor/S1 reaches the user backend exactly once with no
      `TensorifyScalarRestartAnalysis`
- [ ] every compiled output is bitwise pinned-eager equal
- [ ] direct prepared local/global O and FP32 LSE retain frozen references
- [ ] public graph/cache classes are exactly S1 and S>1 with zero breaks; the
      private size-oblivious diagnostic is exactly one graph
- [ ] after compiled-callable creation, mutating each of the eleven source
      floats rejects before FA4 and adds no backend graph, Inductor cache
      class, or FA4 application key; restoring the exact value reuses the
      original graph
- [ ] repeated/reordered and default/nondefault-stream sweeps remain bitwise
      and add no graph, Inductor, or FA4 application class
- [ ] all EXP-0018 cache negatives reject before layer/cache/op/backend entry
      with identical state
- [ ] reset positions, B2, unequal lengths, offsets, padding, metadata, shared
      KV, cache, active gradients, altered masks, non-float module mutations,
      and malformed attestation tensors reject
- [ ] existing eager and EXP-0016 StaticCache matrices remain passing

## Synchronization and generated code

- [ ] memcheck, synccheck, and racecheck pass local and global S1024 default
      Inductor whole-layer cases with unfiltered output retained
- [ ] retained local/global FA4 main-object hashes, PTX/cubin/SASS bytes,
      resource signatures, and launch geometry are unchanged
- [ ] outer compiler graphs/cache entries and the scalar-attestation nodes are
      inventoried separately and stay within the exact S1/S>1 bound

## Candidate result

Implementation revision
`d35a97d6bbe52421f7a04e7d2ab196a2c251a733` is rejected at the first
local/Inductor/S1 discriminator. The larger positive matrix and
sanitizer/codegen gates were not run:

- focused local validation passed `110` tests with `21` hardware/API skips;
  the real H100 custom-op, opcheck, compile-integration, and probe suite passed
  `130` tests with one intentional compatibility skip;
- local/eager/S1 passed the exact direct prepared O/LSE reference, bitwise
  whole-layer output/LSE, one public and one scoped graph, reset/cache
  rejection, opcheck, and nondefault-stream replay;
- the eager graph exposed all eleven source floats, eleven scalar extraction
  nodes, eleven `scalar_tensor` nodes, one stack, and the family whole-layer
  op. This confirms the declared tensor-explicit path was the path tested;
- local/Inductor/S1 still invoked the user backend twice. The first graph
  retained the eleven float placeholders and extraction/tensorization nodes,
  and stock Inductor raised `TensorifyScalarRestartAnalysis`. The second graph
  specialized those float inputs away, retained constant scalar-tensor/stack
  construction, and returned successfully;
- Dynamo recorded no guard failure. The failed result is therefore the same
  frozen two-attempt condition as EXP-0020, now with direct evidence that
  routing each value through `torch.scalar_tensor` does not make PyTorch 2.8's
  tensorification pass complete in its first analysis;
- no graph bound, compiler setting, numerical policy, or result classifier was
  changed after observation. The unsuccessful scalar tensor transport will be
  removed rather than retained as production overhead.

Artifacts:

- `agent_space/remote-h100-exp0021/h100-check-exp0021.json`, SHA256
  `4ee189fd65b8377723f8903b7bac3fd56537375a50029e6a0ce7c6594323fc72`;
- `agent_space/remote-h100-exp0021/h100-exp0021-local-eager-s1.json`, SHA256
  `52fe132f5bffb58a871018f3179981141bd1dbe5d1ec9a3053a492f5fd40814e`;
- `agent_space/remote-h100-exp0021/h100-exp0021-local-inductor-s1.json`,
  SHA256
  `d5698bfed53bf975c69a800ae27e051e34597a69759888ce9d1b409f9c69a7f1`.

The strict report records the same exact pinned H100 environment and patch
stacks as EXP-0020, with empty warnings/errors.

## Measurement

Correctness-only. No timing, speedup, whole-model, compiled-cache, B300, or
cross-architecture claim is authorized.

## Decision

**REJECT.** Explicit CPU FP64 scalar transport preserved the eager numerical
and ABI gates but did not avoid PyTorch 2.8's scalar tensorification restart,
so S1 still required two backend attempts. The candidate adds eleven scalar
construction operations without advancing the compiler boundary and will be
reverted. A later experiment must choose a different predeclared ownership or
guard boundary while retaining the EXP-0018 through EXP-0020 evidence.

## Record

```bash
python scripts/record_result.py EXP-0021 \
  --kernel h100-tensor-scalar-attestation \
  --arch sm_90 --decision reject \
  --hypothesis '<exact hypothesis above>' --bench <jsonl>
```
