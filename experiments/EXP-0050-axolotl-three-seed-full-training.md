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

**PENDING HARDWARE EXECUTION.**
