# Experiment ledger

Copy `TEMPLATE.md` to a monotonically numbered `EXP-NNNN-<slug>.md`. Record
accepted, rejected, refined, and baseline experiments in `results.jsonl` using
`scripts/record_result.py`.

An experiment changes one invariant or tuning decision. Correctness and
synchronization evidence precede timing. Baselines must match the Gemma scale,
mask, prepared K/V operands, geometry, dtype, and timed mode.
