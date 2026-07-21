# EXP-0047: Axolotl full-BF16 pure-SDPA control

- Date / author: 2026-07-21 / Codex
- Kernel family: integration
- Architecture: sm_90
- Official FA4 base: `2409214a03797b168f648ea30df1adbc09ce658a`
- Candidate FA4 revision: `17bf9cb7d0812c5fdbb7ca7ed3d65837d6ad79c1`
- CuTe DSL / CUDA / PyTorch: CuTe DSL `4.6.0.dev0`, CUDA `12.8`,
  PyTorch `2.11.0+cu128`
- Model: `google/gemma-4-12B-it` revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`

## Invariant changed

The comparison control is pure PyTorch SDPA for all 48 text layers. It cannot
register or call a project FA4 or FA2 route. Both control and candidate store
all 11,907,350,272 trainable parameters in BF16 before Trainer creates the
optimizer; the vision tower remains frozen and no adapter is present.

Axolotl normally leaves SDPA's tied token embedding and normalization modules
in FP32. The harness records and explicitly aligns those 292 parameters
(`1,007,410,432` elements) to BF16 so attention backend is the only intended
precision difference.

## Hypothesis

A matched 100-update pure-SDPA control and all-FA4 candidate on a pinned real
dataset will keep mean loss within 5%, final-10 median loss within 0.15, and
control-relative median/P95 gradient norms within the declared bounds.

## Single change

Add a fail-closed full-training harness with a pure `sdpa` backend, pinned
dataset generation, per-update loss/gradient/CUDA-event capture, exact route
and parameter evidence, and a comparison command. This experiment tests the
then-current automatic packed-GQA local FA4 route without altering it.

## Correctness evidence

- [x] Pure SDPA completes 100 real optimizer updates with 100 finite loss,
  gradient-norm, learning-rate, and CUDA-event records.
- [x] SDPA records `{}` project routes and zero native geometry layers.
- [x] Both reports contain exactly 11,907,350,272 trainable BF16 parameters,
  zero adapter parameters, and 52,379,904 frozen parameters.
- [x] The candidate records 8,000 local d256 and 1,600 global d512 FA4 calls,
  exactly two calls per layer/update under checkpoint recomputation.
- [x] All 48 candidate layers record native shapes and distinct K/V storage.
- [ ] The automatic packed-GQA candidate trajectory passes. It does not: mean
  loss is `2.1235171` versus SDPA `1.4909978` (+42.42%), with additional
  gradient spikes beginning near update 21.

Dataset provenance is pinned to `tatsu-lab/alpaca` revision
`dce01c9b08f87459cf36a430d809084718273017`, source SHA-256
`06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073`,
and deterministic 256-record JSONL SHA-256
`acb6da27e9617284c2da74ed1dea014c95f09d42cc42ee137e1ac65271fd5afc`.

## Measurement

The H100 lease was exclusive. Five updates were excluded as warmup; the
remaining 95 synchronized CUDA-event samples are retained.

| route | mean loss | median grad norm | P95 grad norm | median step ms |
|---|---:|---:|---:|---:|
| pure SDPA | 1.490998 | 53.500 | 409.900 | 4023.049 |
| all FA4, automatic packed GQA | 2.123517 | 103.750 | 741.800 | 946.375 |

Both peak at `63,982,582,272` allocated bytes and
`68,371,349,504` reserved bytes.

## Decision

**REJECT.** The speed and memory results do not compensate for the failed
training trajectory. The pure-SDPA control is retained and no production
claim is made for automatic packed-GQA local FA4 training.

Machine-readable evidence is under
`agent_space/remote-h100-exp0047/`.
