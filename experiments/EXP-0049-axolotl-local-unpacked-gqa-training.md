# EXP-0049: Axolotl local unpacked-GQA full training

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

Local `16Q/8KV/d256` forward and backward keep the public GQA geometry
explicit. Forward no longer auto-packs query heads while backward necessarily
uses unpacked ownership. The global `16Q/1KV/d512` path was already explicit
and is unchanged.

## Hypothesis

Passing `pack_gqa=False` for local FA4 will remove the full-BF16 trajectory and
gradient drift without a material step-time or memory regression.

## Single change

Set `pack_gqa=False` for both Gemma layer families at the model integration
boundary. Public Q/K/V shapes, kernel implementation, masks, scale, optimizer,
dataset, and training schedule remain unchanged.

## Correctness evidence

- [x] Same-schedule 30-update discriminator: mean-loss delta versus
  FA2-local/FA4-global falls from 4.81% to 0.82%; paired loss MAE falls to
  `0.07741`.
- [x] The 30-update median gradient ratio is `0.9632`; P95 ratio is `1.0174`.
- [x] Corrected all-FA4 completes 100 real updates with all metrics finite,
  8,000 exact local routes, 1,600 exact global routes, and 48 native geometry
  records with distinct K/V.
- [x] Control and candidate have identical BF16 trainable/frozen parameter
  evidence and identical peak allocated/reserved memory.
- [x] Mean loss differs by 0.503%; paired per-step loss MAE is `0.10648`.
- [x] Candidate/control gradient median, P95, and maximum ratios are
  `1.0257`, `1.2181`, and `0.8539`.
- [ ] The frozen aggregate comparator passes. It does not. Its absolute
  greater-than-10x-median spike rule rejects the pure-SDPA control itself
  (`1424.0 / 53.5`), and the final-10 median absolute loss delta is
  `0.153649`, 0.003649 above the frozen 0.15 cutoff. Neither check is waived.

The pure-SDPA control contains no FA4 route. Both branches train all
11,907,350,272 language-model parameters in BF16 with batch 1, no adapter,
Adafactor, cosine scheduling, scale 1.0, and the same deterministic real-data
order.

## Measurement

Five updates were excluded as warmup. The following are single matched
100-update jobs with 95 synchronized CUDA-event samples, not repeated
fresh-process confidence intervals.

| route | mean loss | last-10 median loss | grad median/P95/max | step median/mean ms |
|---|---:|---:|---:|---:|
| pure SDPA | 1.490998 | 1.188397 | 53.500 / 409.900 / 1424 | 4023.049 / 3306.893 |
| all FA4, unpacked GQA | 1.483499 | 1.342046 | 54.875 / 499.300 / 1216 | 957.013 / 975.166 |

The median and mean step ratios are 4.204x and 3.391x. End-to-end Trainer
runtime is 341.6 seconds for SDPA and 102.9 seconds for all FA4 (3.32x). Peak
allocated and reserved bytes are identical at `63,982,582,272` and
`68,371,349,504`.

## Decision

**REFINE / RETAIN THE UNPACKED-GQA CHANGE.** The earlier 42.4% loss failure is
removed, the control-relative full-trajectory and gradient distributions are
close, and the measured throughput gain is large. The existing aggregate gate
still says false, so this is not promoted to a production-ready or training-
convergence claim. Before promotion, predeclare a control-valid stability
statistic, repeat fresh-process control/candidate runs, and close the 31B
checkpoint gate.

Machine-readable evidence is under
`agent_space/remote-h100-exp0047/`.
