# EXP-0042: H100 all-layer compiled dispatch

- Date / author: 2026-07-20 / Codex
- Kernel family: hf-integration
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Environment: `configs/env/h100-compatible.env`
- Design brief: `docs/h100-all-layer-compiled-dispatch-design.md`

## Invariant changed

Widen only the accepted guarded no-cache compiled facade's layer-index set
from `{0, 5}` to the locked range `0..59`. All tensor, model, mask, inference,
shape, and compiler boundaries remain unchanged.

## Hypothesis

Capturing and revalidating the exact construction-time layer index admits all
60 pinned text layers without adding a layer-index-dependent graph or FA4
specialization class, while same-family index mutation still fails before
compiled entry.

## Single change

Pass an explicit expected layer index through the whole-layer validator and
the guarded facade. Validate the pinned terminal-family
`store_full_length_kv` marker exactly at layers 58 and 59; the locked checkpoint
has no KV-sharing consumers. Do not widen raw whole-layer compilation,
StaticCache, compiled prefill, varlen facade inputs, or multimodal metadata.

## Acceptance gates

- [x] exhaustive CPU/FakeTensor construction and mutation guards
- [x] all 60 actual pinned layers under eager facade at S1
- [x] all 60 actual pinned layers under Inductor facade at S1
- [x] representative S33/S1024 local and global regression
- [x] bounded family-only graph and FA4 application keys
- [x] local and H100 full suites, strict environment, oracle, and bundle

The S1 sweep produces 120 bitwise eager comparisons, zero graph breaks, two
captured family graphs per backend, one whole-layer custom op per graph, and
two total FA4 forward application classes. The fresh cache contains four FA4
files (164,392 bytes) and 25 Inductor files (354,767 bytes). The representative
S33/S1024 matrix is bitwise under both backends and preserves the EXP-0023
negative, mutation, replay, and nondefault-stream gates.

The local suite reports `446 passed, 106 skipped`; the fresh-cache H100 suite
reports `539 passed, 17 skipped, 1 xfailed`. Compileall, Ruff check/format,
the locked local and pinned-Transformers model-contract verifiers, and the
strict exact-patch environment check pass. The focused pinned-Transformers
oracle reports `5 passed, 1 xfailed` for the declared generic FA4 mask-adapter
gap. The schema result is pinned to implementation `a813edf`; final bundle
verification passes with that record and the refreshed manifest.

## Decision

**ACCEPT.** Admit every locked layer index only through the explicitly named
no-cache guarded facade. Retain exact construction-index revalidation and the
pinned terminal-family marker. Raw whole-layer compilation, StaticCache layer
widening, compiled prefill, and full-model execution remain separate work.
