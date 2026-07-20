# EXP-0006: H100 split global d512 backward

- Date / author: 2026-07-19 / Codex
- Kernel family: global-d512-bwd
- Architecture: sm_90
- Source commit: `185f11cbda15ae7bd4841968c3dd46f95b670282`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Active H100 patch:
  `patches/flash-attention/0002-sm90-gemma4-d512-forward-backward.patch`,
  SHA256 `df345b01e4fab6d077898f642ac3ba40effffc6f2291803bc93ae1b0e38ec294`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- QuACK runtime helpers: `quack-kernels==0.5.3`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `4e8cd07ca240508e8bd2bd90f4d681b0c05c4e75a72e5b4e53fbbfbb05890d18`
- Strict environment artifact: `agent_space/h100-check-exp0006.json`, SHA256
  `b41270f33f67dda21af8c8b45d7f76c5287daa2c5a10598f3887f9a56cddc9d4`
- Exemplar path and revision: pinned `flash_attn/cute/flash_bwd_sm90.py`,
  `flash_bwd_postprocess.py`, `interface.py`, and their pinned helpers

## Invariant changed

No model invariant changes. Keep B=1, fixed-length contiguous BF16 BSHD,
32 query heads, 4 KV heads, GQA-8, d512 Q/K/V, causal masking, scale 1.0,
distinct K/V, FP32 score and gradient accumulation, and separate dQ/dK/dV.

Change only backward ownership for the two accepted V256 forward slabs. Each
slab gets one dK/dV-only launch and two D256 dQ-only launches. The outer custom
autograd function preserves FP32 sums across both slabs before the sole BF16
conversion:

```text
dQ = concat(dQ00 + dQ10, dQ01 + dQ11)
dK = dK0 + dK1
dV = concat(dV0, dV1)
```

## Hypothesis

An M64 x N32, one-stage SM90 mainloop split at compile time into dK/dV-only
and D256 dQ-only variants will remain below the H100 232,448-byte dynamic
shared-memory limit and pass the exact B1/32Q/4KV/GQA-8/d512 causal backward
matrix under the numerical policy frozen in EXP-0004.

The falsifying observations were any compile/launch error, realized shared
storage over the limit, non-finite or mis-owned gradients, a numerical-policy
failure, or a sanitizer report.

## Single change

The exact specialization is:

- `tile_m=64`, `tile_n=32`, Q/dO/PdS stages `1/1/1`;
- `SdP_swapAB=False`, `dKV_swapAB=True`, `dQ_swapAB=False`;
- all atom-layout factors 1, with two MMA warp groups plus one producer group;
- dK/dV-only codegen omits dQ and uses an explicit 64 KiB FP32 shared arena;
- two dQ-only variants omit dK/dV and own D256 offsets 0 and 256;
- three separately cached compiled functions preserve ownership, slice offset,
  PdS stage, layout, and broadcast decisions in their keys;
- a default-off SM90 dKV postprocess tiler preserves the producer's two-WG
  accumulator layout while keeping physical WGMMA M at 64.

N16 remains rejected because pinned QuACK cannot form its per-WG N8 BF16
fragment-A. The first real S128 run exposed an incorrect diagnostic workaround
that decoded the two-WG dKV accumulator with 8/4 postprocess warp groups; it
produced non-finite dK and wrong dV. Keeping two WGs and using the physical
`(64, logical-N)` tiler fixed the ownership map. Every result below was rerun
after that fix.

## Correctness evidence

- [x] exact fake-tensor composed backward compile
- [x] S=`1,31,32,33,63,64,65,127,128,129,511,512,513,1024`
- [x] exact B1/32Q/4KV/GQA-8/d512 shapes and scale 1.0
- [x] distinct finite BF16 dQ, dK, and dV
- [x] independent full-d512 FP32 reference and EXP-0004 numerical rule
- [x] half-zero dO-slab superposition and dV-placement checks
- [x] isolated query-head to KV-head ownership check
- [x] three repeated S128 runs and a nondefault-stream run
- [x] invalid/fallback guard tests

Representative commands were:

```bash
env FLASH_ATTENTION_FAKE_TENSOR=1 \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0006-global-bwd-fake-v2 \
  python scripts/probe_h100_global_backward.py --seqlen 128

python scripts/probe_h100_global_backward.py \
  --seqlen 1 --seqlen 31 --seqlen 32 --seqlen 33 \
  --seqlen 63 --seqlen 64 --seqlen 65 \
  --seqlen 127 --seqlen 128 --seqlen 129
python scripts/probe_h100_global_backward.py \
  --seqlen 511 --seqlen 512 --seqlen 513 --seqlen 1024
python scripts/probe_h100_global_backward.py \
  --seqlen 128 --repeats 3 --nondefault-stream
python scripts/probe_h100_global_backward.py \
  --seqlen 33 --compile-only --structured
```

All 14 lengths passed. Across that matrix the largest candidate-vs-FP32
absolute errors were dQ `0.61621094`, dK `1.0`, and dV `0.125`, all inside the
predeclared upstream-relative rule. S128 was dQ `0.5`, dK `0.75`, dV `0.125`.
The structured S33 superposition errors were dQ `0.25`, dK `0.25`, and dV
`0.0078125`; inactive query/KV heads stayed exactly zero.

Repeated O and LSE were bitwise equal. Parallel FP32 bulk/atomic reduction
order made dQ/dK/dV non-bitwise-identical across repeats; dK/dV additionally
reduce across the eight GQA query heads. Every repeat independently passed the
numerical rule.

## Synchronization and generated code

- [x] memcheck at S128 and S129
- [x] synccheck at S128 and S129
- [x] racecheck at S128 and S129
- [x] PTX/SASS instruction path recorded
- [x] registers, spills, static and dynamic SMEM recorded for all main variants

Each tool was run at both lengths:

```bash
compute-sanitizer --tool <memcheck|synccheck|racecheck> \
  --report-api-errors no --error-exitcode 99 \
  python scripts/probe_h100_global_backward.py --seqlen <128|129> --compile-only
```

Memcheck and synccheck reported `ERROR SUMMARY: 0 errors`; racecheck reported
`0 hazards displayed (0 errors, 0 warnings)` at both lengths.

The retained PTX is version 8.8 targeting `sm_90a`. `cuobjdump` reports 168
registers, 1 KiB static shared memory, zero stack, and zero local memory for
all three main variants. Retained MLIR records 222,208 bytes dynamic shared
memory for dKV and 218,112 bytes for each dQ variant. The dKV SASS contains 72
HGMMA, 36 UTMALDG, four UBLKRED.G.S.ADD.F32, and four WARPGROUP.DEPBAR
instructions; each dQ SASS contains 50 HGMMA and 36 UTMALDG instructions.

Fresh S128 compile keys and artifact hashes are:

| variant | compile key | PTX SHA256 | CUBIN SHA256 | SASS SHA256 |
|---|---|---|---|---|
| dKV | `06f1572a86d0f8550caaf4f8d4ba89802176e5701d014161ccb524144e5c336b` | `4d1fe394fec90057ecd9bad5cae6a358ca457d8efd1e23841de105cc872a871e` | `fd43378dde8e0351d24ef4f26b9271d272a06ab884a188c73aa2e7678aa6db81` | `1232b3c384a9eb5847e87c64af8b2bb76436c15569231b8ef47f3f37925a5640` |
| dQ low | `0fc78fca6e5779b659481c8d9583a6809e21c90235db71bab9d8e832851ee9d8` | `2d375823bed71fc5fc4a31dfbb9dfe00235e89d553c041a807f71e934f35361d` | `3b41ac32ba4298b9fb843b2f032c0b47d1318f1da2d64b5b7aec3b88cf9e4ca9` | `10184b28aad296496f3545abe15b70961b7193578dc28ccd638a1f0b51aa9f91` |
| dQ high | `d777f652c51d05c0487eef07494b31ef3f85327e8c833a246ff121291abfb502` | `b4515996822d3ae802d00f94e2388a24d8f30398e4197964034c6f90efb57a45` | `19720a622a91fd9a2645997cc66be5b2d32bde95abc36f6ea078d99366cfe374` | `4c5b96c112b7663fcff8dcfa8845d1ef729046e035e1f53f4a24a9febc2e9396` |

Raw generated files remain ignored under
`agent_space/exp-0006-artifacts-variants/` on the H100.

## Measurement

No performance measurement is authorized or claimed. A full d512 backward
uses six correctness-first main launches and intentionally recomputes score
work.

## Decision

**ACCEPT.** The exact H100 prepared global d512 forward/backward envelope is
promoted. EXP-0005 remains the historical rejection of the unchanged direct
asymmetric-GQA path. This acceptance does not cover multimodal local masking,
varlen, dropout, batch greater than one, B300, or performance.

## Record

The schema-validated EXP-0006 acceptance is appended to
`experiments/results.jsonl`. The next ordered H100 gate is multimodal local
masking forward/backward correctness; performance work remains deferred.
