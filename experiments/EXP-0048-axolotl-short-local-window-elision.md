# EXP-0048: Axolotl short local-window specialization elision

- Date / author: 2026-07-21 / Codex
- Kernel family: integration
- Architecture: sm_90
- Candidate FA4 revision: `17bf9cb7d0812c5fdbb7ca7ed3d65837d6ad79c1`
- Model: `google/gemma-4-12B-it` revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`

## Invariant changed

For causal local attention with `K <= 1024`, the 1024-token left window cannot
exclude any legal key. The integration may therefore request FA4's ordinary
causal path instead of a local-window specialization without changing the
Gemma mask.

## Hypothesis

Eliding the redundant local specialization for the S512 training workload
will remove the trajectory drift observed in EXP-0047.

## Single change

Map local `window_size` to `(None, None)` when `kv_length <= sliding_window`;
retain `(1023, 0)` above the boundary. Global attention remains `(None, None)`.

## Correctness evidence

- [x] CPU boundary tests cover K1024 and K1025.
- [x] A 30-update full-BF16 all-FA4 diagnostic completes with exact routes,
  native geometries, distinct K/V, finite metrics, and bounded memory.
- [ ] The change alone aligns gradients. Against the same-schedule
  FA2-local/FA4-global discriminator, loss mean is 4.81% high and gradient P95
  is 4.11x (`2650.0` versus `644.0`), with a maximum norm of `3712.0`.

## Decision

**REJECT AS A COMPLETE FIX / RETAIN AS AN EXACT ROUTING SIMPLIFICATION.**
The equivalence is valid and the boundary test remains, but automatic local
packed-GQA still fails the gradient discriminator. EXP-0049 changes only that
remaining choice.
