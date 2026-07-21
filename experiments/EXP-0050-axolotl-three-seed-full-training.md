# EXP-0050: Axolotl prospective three-seed full training

- Date / author: 2026-07-21 / Codex
- Kernel family: integration
- Architecture: sm_90
- Candidate FA4 revision: `17bf9cb7d0812c5fdbb7ca7ed3d65837d6ad79c1`
- Model: `google/gemma-4-12B-it` revision
  `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`

## Invariant changed

No attention invariant changes. This experiment replaces the single-run
judgment in EXP-0049 with three new fresh-process pairs. Every control uses
PyTorch SDPA for all 48 text layers and must record empty project routes,
per-layer routes, and native geometry. Every candidate uses native FA4 for
all 40 local d256 and 8 global d512 layers with explicit unpacked GQA.

The three seeds are frozen before execution as `1729`, `31415`, and `65537`.
Each seed selects one deterministic 256-record real-data sample and controls
the matched Axolotl processes. Both members of a pair use the same data,
parameter storage, optimizer, schedule, and example order.

## Hypothesis

Across three new matched 100-update pairs, all-FA4 preserves the robust loss
curve and gradient distribution of pure SDPA while remaining at least 5%
faster at equal peak memory.

## Prospective gate

Each pair must complete 100 finite updates, improve from its initial-20 to
final-20 median loss, and satisfy:

- mean-loss relative delta at most 3%; three-seed median at most 2%;
- paired per-step loss MAE normalized by control mean at most 15%; median at
  most 10%;
- final-20 median-loss relative delta at most 20%; median at most 15%;
- gradient median ratio in `[0.75, 1.33]` and P95 ratio in `[0.50, 2.00]`;
- candidate/control allocated and reserved peak-memory ratios at most 1.01;
- control/candidate median step-time speedup at least 1.05 for every seed and
  in the seed-level median.

The gate deliberately uses gradient quantiles rather than the historical
absolute max/median spike rule because EXP-0047 showed that rule rejects the
pure-SDPA control itself. These thresholds are committed before any EXP-0050
training result is observed and will not be changed after execution.

## Execution

```bash
GEMMA4_FA4_SOURCE_REVISION=<committed-sha> \
  bash scripts/axolotl/run_h100_full_training_matrix.sh
```

The runner acquires `/workspace/.h100-codex.lock`, checks for active compute
processes, and holds the lease across all six fresh Axolotl processes.
It retains each historical single-pair comparison artifact but defers the
matrix exit status to the prospective aggregate above. This is necessary
because the historical max/median rule is known to reject pure SDPA itself.

## Decision

**ACCEPT FOR THE GEMMA 4 12B S512 FULL-BF16 TEXT-TRAINING GATE.** All three
fresh pairs pass every per-seed and aggregate threshold.

## H100 results

All six processes used integration revision
`e4d974cb4e6ab06bd6b6c21f22f8db499ff4f798`. Every control records `{}`
routes, `{}` layer routes, and zero native geometry. Every candidate records
8,000 local d256 plus 1,600 global d512 FA4 calls, all 48 native prepared
geometries, and distinct K/V. Each matched pair has identical dataset,
11,907,350,272 trainable BF16 parameters, 52,379,904 frozen parameters, and
zero adapter parameters.

| seed | mean-loss delta | paired loss NMAE | final-20 delta | grad median / P95 ratio | step speedup | allocated / reserved ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 1.718% | 6.030% | 3.247% | 1.223 / 0.864 | 4.137x | 1.0000 / 0.9994 |
| 31415 | 1.758% | 5.869% | 1.371% | 1.076 / 1.346 | 4.254x | 1.0000 / 1.0000 |
| 65537 | 0.740% | 3.857% | 3.455% | 1.056 / 1.080 | 4.185x | 1.0000 / 1.0003 |

The seed-level medians are 1.718% mean-loss delta, 5.869% paired loss NMAE,
3.247% final-20 delta, and 4.185x median-step speedup. Both control and
candidate improve from their initial-20 to final-20 median loss for every
seed. Peak allocated memory is byte-identical; the largest reserved-memory
ratio is 1.00034.

The first launch attempt completed seed 1729 but stopped when the historical
single-pair comparator rejected SDPA's own gradient spike. Its results are
excluded. Revision `e4d974c` changed only orchestration so the matrix retains
that artifact and continues. After all six clean-restart reports completed,
the aggregate CLI initially lacked `ROOT/src` in its standalone environment;
the committed comparator was then run unchanged with the explicit path and
passed. The runner now supplies that path directly.

This accepts the all-FA4 12B S512 training trajectory and performance gate. It
does not establish 31B training, S1024/S2048 full training, multimodal masks,
or universal production readiness. Retained machine-readable evidence is in
`agent_space/remote-h100-exp0050/`.
