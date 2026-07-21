# EXP-0043: H100 all-layer compiled StaticCache dispatch

- Date / author: 2026-07-20 / Codex
- Kernel family: hf-cache-integration
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Environment: `configs/env/h100-compatible.env`
- Design brief: `docs/h100-all-layer-compiled-cache-design.md`

## Invariant changed

Widen only the accepted guarded compiled-cache facade's layer-index set from
`{0, 5}` to the locked range `0..59`. Preserve every cache layout, tensor,
counter, decode, compiler, and inference invariant.

## Hypothesis

Capturing the exact construction-time cache layer index admits all 60 pinned
text layers without adding layer-index-dependent graph or FA4 specialization
classes, while same-family module-index mutation still rejects before compiled
entry and before cache mutation.

## Single change

Parameterize global/local cache binding and live revalidation by the captured
layer index. Do not widen compiled prefill, raw/full-model compilation,
cached multimodal metadata, varlen facade inputs, training, or performance.

## Acceptance gates

- [x] exhaustive CPU/FakeTensor construction and mutation guards
- [x] all 60 actual pinned cache layers under eager facade at Q1/K33
- [x] all 60 actual pinned cache layers under Inductor facade at Q1/K33
- [x] bitwise outputs/cache state plus exact LSE/reference evidence
- [x] exactly two family cache graphs and bounded FA4 application keys
- [x] EXP-0026 global and EXP-0028 local envelope regressions
- [x] local and H100 full suites, strict environment, oracle, and bundle

The shared-cache sweep produces 120 bitwise decode comparisons across the two
backends. Every target cache has exactly slot 32 written, exact K/V and counter
state versus its independent eager twin, bitwise prepared FP32 LSE, and passing
FP32 reference tolerances. Both backends retain two family cache graphs and
zero graph breaks. The eager sweep adds one local FA4 application class and no
global class; the Inductor sweep adds none after eager warming, so layer index
does not create an application class. The fresh FA4 cache contains six files
(270,048 bytes) and the Inductor cache contains six files (243,490 bytes).

The complete EXP-0026 global K33/K34/K1025 and EXP-0028 local
underfill/boundary/saturated-rollover matrices pass unchanged. The local suite
reports `449 passed, 106 skipped`; the fresh-cache H100 suite reports
`542 passed, 17 skipped, 1 xfailed`. Compileall, Ruff check/format, the locked
model contract, strict exact-patch environment, and pinned Transformers oracle
pass; the focused oracle reports `5 passed, 1 xfailed` for the declared generic
FA4 mask-adapter gap. The schema result is pinned to implementation `6fd7dbe`;
final bundle verification passes with that record and the refreshed manifest.

## Decision

**ACCEPT.** Admit every locked cache-layer index only through the explicitly
named guarded Q1 facade. Retain exact construction-index and cache-layer
identity validation plus the accepted global/local counter transactions.
Compiled prefill, raw/full-model compilation, cached multimodal metadata, and
varlen facade inputs remain separate work.
