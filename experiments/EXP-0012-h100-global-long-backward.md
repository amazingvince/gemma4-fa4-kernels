# EXP-0012: H100 long global d512 backward

- Date / author: 2026-07-19 / Codex
- Kernel family: global-d512-bwd and framework integration
- Architecture: sm_90
- Implementation revision: `ebe993c5b23aae66ecbcf90ee737988546482b9a`
- Result record: schema-valid `EXP-0012` entry in `experiments/results.jsonl`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Starting H100 patch:
  `patches/flash-attention/0002-sm90-gemma4-d512-forward-backward.patch`,
  SHA256 `df345b01e4fab6d077898f642ac3ba40effffc6f2291803bc93ae1b0e38ec294`
- Accepted H100 patch SHA256:
  `521a4e5eeff8c4750fa9ee20499c3bc2ed6597396fee58a585201aac02766abe`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `d1d5e107c2af10140c3866761240be8322420fb6329f896fc153432044ace229`
- Strict environment artifact: `agent_space/h100-check-exp0012.json`, SHA256
  `3667043819fed0b5bcf6abaffa05a3aa2b2ae7157d22f37abd09918d82c65205`
- Exemplar: the accepted EXP-0006 split dKV/dQ SM90 implementation and the
  EXP-0011 fixed/lower-right/packed global composition

## Invariant changed

No model, mask, ownership, precision, or synchronization invariant changes.
Retain exact BF16 B1 per fixed subcall, 32 query heads, 4 distinct K/V heads,
GQA-8, d512, causal lower-right masking, scale 1.0, FP32 accumulation, and
separate dQ, dK, and dV.

Raise only the historical `S <= 1024` evidence and guard boundary to
`S <= 2048`, and add a fail-closed resource preflight. The existing M64 x N32
split backward scheduler, ownership variants, and compile-time keys remain
unchanged. Fixed, lower-right, and packed-varlen framework calls may enter the
training composition only after the project admission check reserves retained
forward state plus anticipated backward scratch. The patched low-level
backward separately rechecks its scratch budget immediately before allocation.

## Hypothesis

Because every `S >= 65` call has the same accepted non-single-block compile
class, the EXP-0006 runtime scheduler will preserve finite O/FP32-LSE and
separate reference-bounded dQ/dK/dV at S1025 and S2048 without creating a
new sequence-specialized main-kernel artifact. A conservative allocation
preflight will reject unsafe longer calls before forward state or backward
scratch can exhaust H100 HBM.

Falsification is any numerical-policy failure, illegal access, barrier or
race report, new length-keyed main-kernel artifact, incorrect lower-right or
packed segment ownership, non-finite result, or allocation proceeding when
the estimated peak exceeds its declared free-HBM budget.

## Single change

Raise the fixed global-backward sequence cap from 1024 to 2048, add fail-closed
project admission, and add a low-level scratch recheck. Do not change an
accepted tile, stage, warp-group count, barrier protocol, accumulator ownership
map, postprocess tiler, or model semantic. Do not tune performance.

The first boundary probe is deliberately limited to S1025 and S2048. The
accepted direct and composed training envelope therefore stops at K2048.
Native packed-varlen backward and K above 2048 require a separate ABI and
scheduler experiment; they are not inferred from this result.

## Correctness evidence

- [x] locked contract and pinned Transformers oracle
- [x] S1025 and S2048 fixed O/LSE/dQ/dK/dV reference checks
- [x] lower-right Q33/K1025 combined-output reference
- [x] packed Q=[33,65], K=[1025,2048] combined-output reference and isolation
- [x] dO-only, LSE-only, and combined dO+dLSE gradients
- [x] invalid shapes and preflight rejection before allocation
- [x] repeated-run and nondefault-stream checks
- [x] full local and H100 test suites

The unchanged EXP-0002/0004 numerical policy passed. Representative maximum
absolute errors were:

| case | O | LSE | dQ | dK | dV |
|---|---:|---:|---:|---:|---:|
| fixed S1025, dO | 0.015625 | 0.00011444092 | 0.56811523 | 1.03125 | 0.125 |
| fixed S2048, dO | 0.015625 | 0.00014495850 | 0.55712891 | 1.0 | 0.125 |
| fixed S1025, LSE only | 0.015625 | 0.00011444092 | 0.0625 | 0.25 | 0.0 |
| fixed S1025, dO+dLSE | 0.015625 | 0.00011444092 | 0.56640625 | 1.0 | 0.125 |
| lower-right Q33/K1025 | 0.015625 | 0.00012207031 | 0.5 | 0.35253906 | 0.03125 |
| packed mixed lengths | 0.015625 | 0.00011444092 | 0.59375 | 0.59375 | 0.03125 |

Every error was inside the independently established upstream-relative BF16
limit. LSE-only dV was exactly zero. The S1025 structured probe retained
V-slab superposition and isolated query-head 9 to KV-head 1 with every
inactive head exactly zero. The mixed packed probe proved exact-zero dQ/dK/dV
outside the segment selected by the upstream gradient.

Three S2048 nondefault-stream repeats independently passed. O, LSE, and dV
were bitwise equal; FP32 atomic reduction order made dQ and dK non-bitwise,
as already documented by EXP-0006. No deterministic-gradient claim is made.

Representative commands:

```bash
python scripts/probe_h100_global_backward.py \
  --seqlen 1024 --seqlen 1025 --seqlen 2048 --reference
python scripts/probe_h100_global_backward.py \
  --seqlen 2048 --reference --repeats 3 --nondefault-stream
python scripts/probe_h100_global_backward.py \
  --seqlen 2048 --compile-only --record-memory
python scripts/probe_h100_global_backward.py \
  --seqlen 1025 --reference --structured
python scripts/probe_h100_global_backward.py \
  --seqlen 1025 --reference --gradient-source lse
python scripts/probe_h100_global_backward.py \
  --seqlen 1025 --reference --gradient-source out_lse
python scripts/probe_h100_transformers_integration.py \
  --case global-lower-right-long-backward --seed 12012
python scripts/probe_h100_transformers_integration.py \
  --case global-packed-long-backward --seed 12012

for tool in memcheck synccheck racecheck; do
  compute-sanitizer --tool "$tool" \
    --report-api-errors no --error-exitcode 99 \
    python scripts/probe_h100_global_backward.py \
      --seqlen 1025 --compile-only
  compute-sanitizer --tool "$tool" \
    --report-api-errors no --error-exitcode 99 \
    python scripts/probe_h100_transformers_integration.py \
      --case global-packed-long-backward --seed 12012
done
```

## Synchronization and generated code

- [x] memcheck at the first above-boundary tail and packed mixed lengths
- [x] synccheck at the first above-boundary tail and packed mixed lengths
- [x] racecheck at the first above-boundary tail and packed mixed lengths
- [x] compile-cache inventory across S65, S1024, S1025, and S2048
- [x] generated object comparison against the accepted long compile class
- [x] registers, spills, dynamic SMEM, and static SMEM unchanged

All six sanitizer runs passed. Memcheck and synccheck each reported
`ERROR SUMMARY: 0 errors`; racecheck reported
`0 hazards displayed (0 errors, 0 warnings)` for fixed S1025 and the packed
Q=[33,65], K=[1025,2048] framework case.

A fresh S65 cache created the expected forward, one dO-only preprocess, three
postprocess, and three split-main objects. S1024, S1025, and S2048 added no
path or object. Exercising dLSE added exactly one expected preprocess object;
lower-right values, packed totals/order, sanitizers, and repeats added none.
The final inventory is 18 paths (nine objects plus nine lock files).

The three multi-block main keys remain:

| variant | key | cached object SHA256 |
|---|---|---|
| dKV | `06f1572a86d0f8550caaf4f8d4ba89802176e5701d014161ccb524144e5c336b` | `0276afb6725c86a69d00c51aea739953fbac11fc0751db6796d693711d83d21b` |
| dQ low | `0fc78fca6e5779b659481c8d9583a6809e21c90235db71bab9d8e832851ee9d8` | `e8648710d55d0da92a8411ea54ff1a9b6151af8fad1eb6587a6ab405c4ac4ca6` |
| dQ high | `d777f652c51d05c0487eef07494b31ef3f85327e8c833a246ff121291abfb502` | `a4d1b2063870248463a5a754c1db08c6e5fae6435a47975edf5c5c3126315889` |

Those object hashes are byte-identical to the retained EXP-0006 S>=65
objects. Consequently the recorded 168 registers, zero stack/local memory,
1 KiB static shared memory, and 222,208-byte dKV / 218,112-byte dQ dynamic
shared memory remain exact; no PTX/SASS or resource decision changed.

## Measurement

No performance measurement is authorized for this experiment. Timing output,
if any tool emits it incidentally, is not a benchmark or speed claim.

The conservative fixed-backward scratch estimate is

```text
147456*S + 66048*round_up(S,64) + 16384*round_up(S,32)
```

Forward admission additionally reserves retained O/LSE and anticipated
dO/dLSE, with both an 80%-of-free-HBM ceiling and a 2 GiB reserve. At S2048,
the measured candidate-only peak increment was 538,181,632 bytes, below the
605,552,640-byte estimate. The patched low-level function rechecks the
backward scratch budget immediately before allocation.

## Decision

**ACCEPT.** The exact H100 fixed global backward and framework composition are
promoted from K1024 through K2048. The implementation is a resource-guarded
extension of the existing runtime scheduler, not a new or tuned kernel.

This does not accept native packed-varlen global backward, K>2048 training,
maximum-context square backward, deterministic gradients, FakeTensor/
`torch.compile`, static-cache model execution, performance, B300, dropout,
or any model geometry other than exact Gemma 4 global attention.

## Record

The schema-valid EXP-0012 record names implementation
`ebe993c5b23aae66ecbcf90ee737988546482b9a`, the updated H100 environment
policy, and `agent_space/h100-check-exp0012.json`.

Final repository gates on the implementation tree:

- local: `180 passed, 75 skipped, 9 warnings`;
- H100: `251 passed, 8 skipped, 1 xfailed, 9 warnings`;
- pinned Transformers model oracle: status `ok`;
- strict H100 environment: no warnings or errors;
- compileall and Ruff: pass.
