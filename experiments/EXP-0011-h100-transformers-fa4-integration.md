# EXP-0011: pinned Transformers to H100 FA4 integration

- Date / author: 2026-07-19 / Codex
- Kernel family: framework integration
- Architecture: sm_90
- Implementation revision: `e7f26bba9b6795e3022c733cff39e060075daf57`
- Result record: schema-valid `EXP-0011` entry in `experiments/results.jsonl`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Pinned Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Transformers patch:
  `patches/transformers/0001-gemma4-forward-vision-block-ids.patch`
- Transformers patch SHA256:
  `773950a1f1feb04f5f2e6a1d66f8953ff8905e8ca9391f804089f169da59b671`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar: pinned Transformers attention/mask registries and mask
  combinators, plus the project-owned fixed and packed FA4 adapters

## Invariant changed

The accepted low-level H100 calls gain one eager, model-specific framework
boundary. Pinned Transformers prepared Q/K/V remain distinct BHSD operands.
The adapter creates a no-copy BSHD view when its real layout satisfies the
pinned CuTe 128-bit contract, constructs exact packed THD streams only when
padding, reset positions, documents, or explicit cumulative arrays require
them, and exposes its selected route.

One hash-locked Transformers patch creates or accepts a single authoritative
vision-block tensor before mask construction. The same tensor builds the mask
and reaches the registered attention interface, including when generation
supplies a prebuilt mask mapping. Explicit IDs take precedence over derivation
from multimodal token types.

## Hypothesis

A uniquely registered `gemma4_fa4_h100` attention/mask pair preserves
O/FP32-LSE/separate dQ/dK/dV for the locked 50-local/10-global layer pattern,
fixed and packed-varlen calls, padding, lower-right offsets, and multimodal
masking without an unconditional input copy, a dropped mask, K/V aliasing, or
misreporting LSE as attention weights. Exact no-grad global forward can extend
through K262144 with memory preflight, while unsupported gradient cases fail
closed.

Falsification is any generic-FA4 registry override, incorrect layer route,
lost vision/document/padding boundary, accepted non-unit scale, hidden fixed
view copy, incorrect lower-right coordinate, arbitrary mask entering a native
route, nonfinite accepted result, unbounded compile variants, or an
unsupported path executing without an explicit diagnostic.

## Single change

Add the pinned eager framework adapter and compact metadata transport. Do not
change an accepted local/global kernel tile, stage, barrier protocol, model
geometry, or scale. Global batching and varlen are exact compositions of the
already accepted calls. No performance work is part of EXP-0011.

## Framework and mask contract evidence

- `gemma4_fa4_h100` is registered in both pinned Transformers registries;
  registration is idempotent for the project callables and rejects collisions.
- Generic `flash_attention_4` attention and mask entries are not replaced.
- The locked layer index routes all 60 layers as 50 local and 10 global;
  mutable layer labels cannot change the locked geometry.
- The Transformers interface returns `(BSHD output, None)`. Internal
  `Gemma4DispatchResult` retains FP32 LSE when available and the route name.
- The mask plan retains callable, padding/static mask, and Q/K offset data.
  Native dispatch requires an exact structural fingerprint of the pinned
  causal/sliding/packed/vision combinators and matching captured metadata.
- Arbitrary callables, floating additive 4D score masks, mismatched closures,
  and unclassifiable expressions never enter a native route. Their exact plan
  reaches only the inference fallback, or raises when fallback is disabled.
  Boolean/integer 4D masks and noncontiguous repeated document IDs fail closed.
- Batch-1 position IDs and metadata broadcast across the prepared batch.
  Explicit cumulative arrays cannot cross batch rows. Short static 2D masks
  are extended with masked positions; holey/static layouts retain the exact
  fallback plan.
- Prepared K and V must be distinct allocations. Invalid scale, dropout,
  requested attention weights, last-D stride, geometry, alignment, overlap,
  and unsupported architecture reject before dispatch.
- Framework FakeTensor/`torch.compile` tracing is explicitly unsupported and
  fails closed.

## Layout and copy evidence

The canonical BHSD-to-BSHD transpose shares storage for Q, K, and V. The H100
probe verifies storage identity and legal noncontiguous view strides on fixed
local S65, fixed global S33, and lower-right local Q3/K9. Fixed routes do not
perform an unconditional contiguous conversion.

Copies are localized to valid-token gathers/scatters for padding/packing,
zero-Q prefixes used to establish lower-right coordinates in composed global
training, and the required V256 slabs. Padded result tails are exact zero O and
`-inf` LSE.

## H100 correctness evidence

The normal probe passed all eight cases:

```bash
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_transformers_integration.py --case all
```

| Case | Accepted route and evidence |
|---|---|
| local fixed strided S65 | `fa4_local_fixed`; O/LSE and separate gradients; zero-copy view |
| local padded B2 lengths `[6,3]` | `fa4_local_varlen`; O/LSE and separate gradients; zero-O/`-inf`-LSE tails |
| local lower-right Q3/K9 | `fa4_local_varlen`; exact lower-right O/LSE and separate gradients |
| global fixed strided S33 | `fa4_global_fixed`; O/LSE and separate gradients; zero-copy view |
| global B2/S5 | exact per-segment `fa4_global_varlen`; O/LSE and separate gradients |
| global no-grad Q1/K2048 | `fa4_global_forward_only`; O/LSE reference |
| global no-grad packed Q33/K2048 | `fa4_global_varlen_forward_only`; O/LSE reference |
| pinned HF transport | paired registration, derived/explicit/prebuilt vision IDs, registered local FA4 call, and actual `Gemma4TextAttention` forward/backward |

Training-capable global composition is accepted only when every K segment is
at most 1024. Fixed/rectangular and packed global forward-only routes accept
nonempty `1 <= Sq <= Sk <= 262144` with autograd disabled and preflight a
conservative composed output/LSE allocation estimate against 80% of current
free HBM.

The separately guarded maximum-context command passed:

```bash
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_transformers_integration.py \
    --case global-forward-only-max-context
```

For Q1/K262144 with zero scores and a value sentinel of 64 at key zero, the
route was `fa4_global_forward_only`, output was exactly `64/262144`, and FP32
LSE matched `log(262144)` within `1e-4` absolute tolerance.

The pinned optional Transformers checks also pass for the project-specific
mask plan and explicit generation-mask IDs. The strict xfail for the generic
FA4 mask adapter remains intentionally unchanged because that generic adapter
cannot encode Gemma's local vision future-token exception.

## Rejected Flex backward

The first D512 Flex fallback attempted a default M64/N32 tile and failed
explicitly because it requested 299,008 shared-memory bytes against H100's
232,448-byte limit. A smaller compiled B2/S5 fallback candidate then produced
a non-finite dQ. That backward is rejected evidence, not a route to tune into
acceptance inside this experiment.

The production fallback is therefore inference-only. If autograd is enabled
and any prepared operand requires a gradient, the adapter raises before
calling FlexAttention. This leaves global B2/S5 training on the passing exact
per-segment FA4 composition.

## Synchronization and generated-code evidence

- [x] memcheck: composed global B2/S5 and packed global Q33/K2048 no-grad
- [x] synccheck: the same two cases
- [x] racecheck: the same two cases
- [x] isolated compile-cache object inventory and exact hashes
- [x] final repository-wide verification and result-schema record

Representative command form:

```bash
bash scripts/remote/run.sh h100 compute-sanitizer --tool <tool> \
  --report-api-errors no --error-exitcode 99 \
  python scripts/probe_h100_transformers_integration.py \
    --case <global-varlen-batch|global-varlen-forward-only-long>
```

Memcheck and synccheck reported zero errors, and racecheck reported no hazards
or errors for both cases. EXP-0011 changes framework composition and opens the
pinned no-grad global forward entry points; it does not claim new native
varlen generated-code counts.

The final candidate passed the isolated real-H100 cache probe in the fresh
path
`/workspace/.cache/exp-0011-transformers-cache-019f77fa-2659-7e33-9934-eb367ad23a55-d`:

- fixed backward S<=32: 0 to 8 object paths;
- fixed backward S33..64: 8 to 11;
- fixed backward S>=65: 11 to 14;
- composed segment order, values, legal strides, and B1/B3 runtime batch:
  unchanged at 14;
- long fixed forward-only B1/S1025 and B2/S1033: unchanged at 14;
- native packed long forward: 14 to 15;
- packed order, values, legal strides, and B1/B2 runtime batch: unchanged at
  15.

The final inventory is 15 paths, nine unique object contents, and 976,336
bytes. The cache namespace is
`6167b82729b6e91949486c7a469474318b0906e396849651633c18d6f82aef92`.
Exact keys and object SHA256 values are:

| Family / class | Cache key | Object SHA256 |
|---|---|---|
| fixed forward | `9d458b84f3527bebfcb7ee487274eb29d88742098ee88282c56264cae78ef0f3` | `3fe53c5e9ded56603c53b5aedda602868c565ac3d983ce80ad422d3c96707018` |
| native-varlen forward | `44952ddb12c947e354dbd0e9d04d6d54281270f597adb1d8b7fe0ca6eabd8b8b` | `ffaeca2b51fc02c73135f21259fc99a7587c5975266e42224ae3e69507b07bba` |
| backward preprocess | `7e9444f215f0390e61d727e7ada6dab49cf3d232cedf1529aa7619feea03c3d2` | `8a4f244f2358089eaf08f9edad6870aef39628d0ee24f78887bad29155147084` |
| backward postprocess | `993a57b2ec3617d2937b8079b9b9480886b14a96a8d3e0e8129a1fb9a6a1cf7c` | `2aac61ebcbce2c7c3f14ef9fa59c8c87e19d57dcf0d37216d92cef74d780e746` |
| backward postprocess | `ed0e318bf7baf45f045036c12ae5865e8e6b35c05648ccbfe453fb28117fe5dc` | `040303d8505264ac36f3a38bb19113b73bde2f1383638118b4e695889a1f9bae` |
| backward postprocess | `f6c4dbb89ebd141b5f184a2d55d6f709cc3608211943e7b71f30a28c922e179d` | `cf5148f524365e0eedc6324a7d4456f528e892bf665f9c24bd6a6bd9412ced7e` |
| dKV S<=32 / S33..64 / S>=65 | `37fa3700bfa906585f0ad44a3bef4c49b73f7c090c60b75c924f158f86caba39`, `e5863e871869920561f8d049b746336569f69e3b59d56a0bb0f70f27689e4261`, `06f1572a86d0f8550caaf4f8d4ba89802176e5701d014161ccb524144e5c336b` | `0276afb6725c86a69d00c51aea739953fbac11fc0751db6796d693711d83d21b` |
| dQ-low S<=32 / S33..64 / S>=65 | `f52e2dee79a411d7fcfad55720ef1e388ca1521ddef31d95186e25d76e5f8428`, `b9f5c437f4e7e77a220037d92fea2890cb1b2f3acf55561fed3fddd10146311e`, `0fc78fca6e5779b659481c8d9583a6809e21c90235db71bab9d8e832851ee9d8` | `e8648710d55d0da92a8411ea54ff1a9b6151af8fad1eb6587a6ab405c4ac4ca6` |
| dQ-high S<=32 / S33..64 / S>=65 | `ed516908ab3b98699de259100246b4c357ed7f0fb5181885c1ec85a79fc81945`, `5b7399d239551fd08c5d8495843eafdaac17e023221d169aebb7a93bbbc3fd01`, `d777f652c51d05c0487eef07494b31ef3f85327e8c833a246ff121291abfb502` | `a4d1b2063870248463a5a754c1db08c6e5fae6435a47975edf5c5c3126315889` |

The fixed and native-varlen forward objects are distinct (87,344 versus
99,408 bytes). The three class keys per backward role have byte-identical
objects. These are bounded compile/artifact facts, not timing or performance
evidence.

## Measurement

No benchmark ran. No speed, efficiency, fused-d512, full-checkpoint,
FakeTensor/compiled-model, static-cache, or B300 claim is authorized for
EXP-0011.

## Remaining H100 work

1. Design exact global backward above K1024.
2. Design a separate FakeTensor/`torch.compile` and compiled/static-cache ABI
   with an explicit compile-key contract.

## Decision

**ACCEPT (scoped eager H100 integration).**

The H100 eager compatibility hypothesis is supported by the recorded probe,
maximum-context sentinel, focused sanitizer evidence, and bounded cache
inventory. The committed implementation passed `175 passed, 75 skipped`
locally and `246 passed, 8 skipped, 1 xfailed` on H100; the checksum-locked
bundle verifier and schema validation also passed on H100. Acceptance remains
limited to the eager envelopes stated above. Global backward above K1024,
FakeTensor/`torch.compile`, compiled static-cache execution, performance, and
B300 remain excluded.
