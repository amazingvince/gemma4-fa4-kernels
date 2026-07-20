# EXP-0028: H100 local cache counter transaction

- Date / author: 2026-07-20 / Codex
- Status: **ACCEPTED — scoped local compiled-cache decode envelope**
- Kernel family: pinned Transformers local-d256 one-token decode over the
  retained native packed-varlen local text forward
- Architecture: sm_90
- Starting revision: `704d5de79ec76545bbbd265311fb3f5432751ccb`
- Model-contract lock SHA256:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Transformers patch SHA256:
  `ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79`
- H100 FA4 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`

## Prior boundary and rejection

EXP-0016 accepts eager layer-0 `StaticSlidingWindowLayer` semantics.
EXP-0025 accepts non-static full-storage views of upstream-marked cache roots
as explicit FX placeholders, and EXP-0026 accepts the resulting global
`StaticLayer` envelope. EXP-0027's first local candidate correctly reached the
K1024 boundary but declared K, V, and the CUDA counter mutable in one opaque
custom-op schema. PyTorch therefore changed the counter version on the first
saturated rollover even though the body left its bytes at 1024. Pinned
upstream performs no counter operation on that branch, so EXP-0027 was
rejected before its wider matrix.

## Invariant changed

Keep the exact EXP-0027 local layer-0, B1/Q1 BF16 inference/no-grad, text-only,
eager-prefill envelope and the same three explicit cache placeholders. Change
only mutation ownership:

- the opaque local custom op reads the explicit CUDA counter but declares and
  performs mutation only on K and V;
- after a successful compiled call, the guarded eager facade first proves K/V
  mutation and a byte/version-unchanged CUDA counter;
- for absolute positions below 1024, the facade then increments the CUDA
  counter exactly once; at positions 1024 and later it performs no counter op;
- only after validating that tensor transition does the facade increment
  `cumulative_length_int` exactly once and revalidate the complete binding.

The custom op still owns exact K/V slot-or-roll mutation and attention over
the resulting ordered active window. The facade already owns both pinned
counter identities outside Dynamo, so this transaction introduces no Python,
cache, or module state into FX and no second compiled graph.

No attention kernel, numerical policy, metadata path, dependency, compiled
prefill, raw/full-model compile, training, performance, or B300 behavior
changes.

## Hypothesis

A K/V-only local custom-op mutation schema followed by an eager guarded
counter transaction will preserve the pinned CUDA-counter byte and version
protocol through underfill, boundary fill, and repeated saturated rollover,
while retaining one tensor-only graph with explicit K/V/counter placeholders,
bitwise eager cache/output equality, and the accepted native local-varlen
application/codegen class.

Falsification is a second graph; graph break; cache `get_attr`; missing counter
placeholder; Python/cache/module source in FX; more than one local cache op;
counter mutation during the custom op; absent or extra facade counter mutation;
wrong slot/roll order; cache/output byte disagreement; numerical policy
failure; hostile-tail influence; address/alias change; unsupported input
entering compiled code; sanitizer finding; changed native local-varlen
application/code/resource/launch signature; or any widened claim.

## Single change

1. Remove `cache_length` from the local custom op's `mutates_args` set without
   removing it from the tensor ABI or graph.
2. Remove counter mutation from the opaque body; derive the underfill slot and
   active length from the already validated absolute position.
3. Split local facade post-validation into K/V-op validation, conditional
   eager CUDA-counter update, and eager Python-count update.
4. Update the EXP-0027 probe/schema assertions to identify EXP-0028 and prove
   the refined transaction. Do not change a CuTe kernel.

## First discriminator

After eager K1023 prefill, one Inductor facade runs absolute positions 1023,
1024, and 1025. Require one backend attempt, zero breaks, one K/V-mutating
local cache op, three explicit cache placeholders, zero cache `get_attr`, and
no forbidden Python/module/cache source. Whole-layer output and complete cache
state must be bitwise equal to a weight-identical pinned eager twin. The root
CUDA counter version must change only at position 1023; bytes transition 1023
to 1024 and remain 1024; Python length advances 1023 through 1026.

Stop and record a rejection before the wider matrix if this discriminator
fails.

## Correctness gates

- [x] schema marks only K/V mutable; FakeTensor returns fresh BF16 output and
      FP32 `(1,32,1)` LSE; all 13 tensor arguments remain explicit
- [x] first discriminator passes one graph, zero breaks, one cache op, three
      cache placeholders, zero cache `get_attr`, and no forbidden source
- [x] boundary fill, first rollover, and repeated rollover match pinned eager
      whole-layer output and complete cache bytes bitwise
- [x] CUDA counter bytes/versions and Python absolute count follow the exact
      pinned transition protocol and update only after successful prior stages
- [x] independent underfill K33/K34 pass eager and Inductor; hostile unwritten
      tails cannot affect prepared O/LSE or projected O
- [x] prepared Q/K/V are exact; local packed O and FP32 LSE pass frozen
      reference policies at underfill, boundary, and rollover
- [x] root/view addresses remain stable, K/V remain distinct, and different
      seeds, reverse order, and one nondefault stream add no graph/application
      class
- [x] malformed position/counters, B2, Q2, metadata, grad, lazy/offloaded/
      foreign/reset/rebound/forged/wrong-class/wrong-capacity inputs reject
      before compiled entry with byte-identical state
- [x] EXP-0016, EXP-0023, EXP-0025, EXP-0026, raw-compile negative, fixed,
      packed-varlen, mask, and backward regressions remain unchanged

## Synchronization and generated code

- [x] unfiltered memcheck passes one repeated local rollover call
- [x] project-kernel-filtered synccheck passes the same call
- [x] project-kernel-filtered racecheck passes the same call
- [x] native local-varlen application object and retained PTX/cubin/SASS/
      resources/launch geometry remain equal to the accepted predecessor
- [x] Dynamo graph, Inductor files, root/view mutation state, both counters,
      and FA4 application keys are separately inventoried

## Measurement

Correctness-only. No timing, speedup, compiled-prefill, whole-model,
raw-`torch.compile(layer)`, cached vision/document metadata, training, B300,
or cross-architecture claim is authorized.

## Decision

**ACCEPT** at implementation source
`829dc5bf2691d47349f8e94ebbe616c260740569` for only the pinned local
layer-0, B1/Q1 BF16 inference/no-grad, text-only, eager-prefill envelope.
The first discriminator passes positions 1023, 1024, and 1025 through one
Inductor graph with zero breaks, one local cache op, 13 tensor plus four
symbolic placeholders, and zero `get_attr`. Whole-layer output and complete
cache bytes are bitwise to the pinned eager twin. The CUDA counter changes
bytes and version only at the boundary fill and stays saturated at 1024
through both rolls; the Python absolute count advances once per successful
transaction.

Default and reverse matrices use different seeds and opposite orders. Each
passes eager and Inductor K33/K34 underfill, hostile-tail underfill, a
nondefault stream, and K1024 plus two rolls. The two matrices retain one
semantic/runtime graph signature and the accepted native local-varlen decode
application key
`e7b213f0ae59536df7feec9f0202f6cdace2105999b143dc3c133cbda041f176`.
The largest prepared-output absolute error is 0.015625 and the largest FP32
LSE absolute error is 0.00002288818359375, inside the frozen policies.

Sixteen malformed or out-of-scope cases reject before compiled entry without
state mutation or a new graph/application. Unfiltered memcheck, filtered
synccheck, and filtered racecheck pass the repeated-roll case. The retained
native host object remains bitwise equal to EXP-0016; the extracted cubin uses
168 registers, zero stack/local bytes, 1024 static shared-memory bytes, and
the retained HGMMA/TMA/barrier instruction class. No PTX payload is embedded.

This decision does not accept compiled prefill, B>1/Q>1, cached
vision/document metadata, facade training/backward, other layer indices,
raw/full-model compilation, performance, or B300.

## Evidence

- `agent_space/remote-h100-exp0028/first-discriminator.json`
- `agent_space/remote-h100-exp0028/local-envelope-default.json`
- `agent_space/remote-h100-exp0028/local-envelope-reverse.json`
- `agent_space/remote-h100-exp0028/negative-matrix.json`
- `agent_space/remote-h100-exp0028/sanitizer-case.json`
- `agent_space/remote-h100-exp0028/sanitizers/`
- `agent_space/remote-h100-exp0028/codegen-inventory.json`
- `agent_space/remote-h100-exp0028/h100-check.json`
- `agent_space/remote-h100-exp0028/exp0023-regression.json`
- `agent_space/remote-h100-exp0028/exp0025-regression.json`
- `agent_space/remote-h100-exp0028/exp0026-regression.json`

Local verification at evidence commit `49da9b8d8a199354d0ccdb8bd271b4d2301dc5f6`
reported `415 passed, 106 skipped, 8 warnings`. After the seven affected
Python files were mechanically Ruff-formatted, the discriminator, both full
matrices, the independent negative matrix, all three sanitizers, and the
EXP-0023/0025/0026 regressions were rerun and durably collected at final
source `ebcc1fc2623e565991151d9beeea29cbe17bfbd8`. The complete pinned-H100
bundle verifier then reported
`508 passed, 17 skipped, 1 xfailed, 8 warnings`, 264/264 manifest hashes,
clean lint/format/shell/static/JSON checks, and 27 valid pre-record entries.
The one xfail is the retained generic Transformers FA4 mask limitation, not a
project-kernel failure. The strict H100 environment check has empty warnings
and errors. The schema-enforcing recorder subsequently appended this result,
bringing the ledger to 28 entries.

## Record

```bash
python scripts/record_result.py EXP-0028 \
  --kernel h100-local-cache-counter-transaction \
  --arch sm_90 --decision accept \
  --hypothesis '<exact hypothesis above>' \
  --profile agent_space/remote-h100-exp0028/h100-check.json
```
