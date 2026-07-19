# AGENTS.md — Gemma 4 FA4 kernel lab operating contract

## Mission

Build exact BF16 forward and backward attention kernels for Gemma 4 31B on
H100 (SM90a) and B300 (SM103), then optimize only after the semantic and
synchronization contracts are proven.

## Mandatory reading order

1. `docs/model-contract.md`
2. `docs/status.md`
3. `docs/design.md`
4. `docs/experimental-plan.md`
5. `skills/gemma4-kernel-project/SKILL.md`
6. **REQUIRED SUB-SKILL for kernel work:**
   `skills/writing-cute-dsl-kernels/SKILL.md`, plus the references it routes.

The complete user-supplied CuTe DSL skill is vendored as a directory. Do not
copy or read only its entry point when a referenced architecture, layout,
pipeline, JIT, correctness, or profiling file applies.

## Non-negotiable model invariants

- 60 text layers: 50 sliding and 10 full; full at indices 5, 11, ..., 59.
- 32 query heads in both layer families.
- Sliding: 16 KV heads, QK/V d=256, window 1024, GQA ratio 2.
- Full: 4 KV heads, QK/V d=512, GQA ratio 8.
- Attention scale is exactly `1.0` for both layer families.
- Sliding mask is `k > q - 1024 AND (k <= q OR same nonnegative vision block)`.
- The local overlay is a left bound only: same-block vision keys may be
  arbitrarily far in the future, while far-past keys remain window-limited.
- Full mask is causal; vision bidirectionality does not apply to full layers.
- Global K and V share a projection source but are distinct FMHA operands:
  K receives K RMSNorm + partial/proportional RoPE; V receives a different
  RMSNorm and no RoPE.
- Never require `k is v`; never merge dK and dV in an attention-only backward.
- Global partial rotary factor is 0.25: 128 of 512 channels are rotary.
- The checkpoint sets `num_kv_shared_layers=0`; no text layer reuses prepared
  KV from another layer. Keep this distinct from the global shared projection source.
- Dropout is zero; BF16 input/output and FP32 score/LSE/gradient accumulation
  are the first implementation target.

The machine-readable source is `configs/model/gemma4-31b.lock.json`.

## Evidence loop

Every kernel iteration follows:

1. Create `experiments/EXP-NNNN-<slug>.md` from the template.
2. State one falsifiable hypothesis tied to a counter or invariant.
3. Change one tuning or structural decision.
4. Compile the exact specialization and record the cache key.
5. Pass reference tests and relevant optional HF-oracle tests.
6. Run memcheck/synccheck/racecheck for new memory or barrier protocols.
7. Inspect IR/PTX/SASS when the hypothesis depends on an instruction path.
8. Benchmark at locked, recorded clocks with semantically equivalent baselines.
9. Record accepted and rejected results.

A compile is not a correctness proof. A random square test is not a layout
proof. A performance number is invalid if scale, mask, K/V preparation,
dtype, or gradient contract differs from the model.

## Upstream and version discipline

- `upstream.lock.json` pins Transformers and FlashAttention revisions.
- `configs/env/latest-compatible.env` pins the reviewed environment policy.
- `scripts/setup_env.sh` checks out the exact revisions under `.upstream/`.
- Any upstream refresh is a separate reviewed change: update locks, rerun the
  model oracle, compile matrix, sanitizer matrix, and baselines.
- Every constexpr or codegen-changing option belongs in the compile-cache key.
  Runtime tensors and per-step lengths never belong in it.

## Remote GPU workflow

Copy `remote/<profile>.env.example` to `remote/<profile>.env`, fill SSH values,
and keep it untracked.

```bash
bash scripts/remote/sync.sh h100
bash scripts/remote/bootstrap.sh h100
bash scripts/remote/check.sh h100
bash scripts/remote/run.sh h100 python scripts/verify_model_contract.py --transformers
bash scripts/remote/collect.sh h100
```

System CUDA installation is explicit and guarded. Driver upgrades are not
automated because they can disrupt shared machines and require a reboot.

## Commands

```bash
python scripts/verify_model_contract.py
pytest -q
ruff check .

# Optional pinned Transformers oracle
python scripts/verify_model_contract.py --transformers
pytest -q tests/test_hf_oracle_optional.py

# GPU environment
python scripts/check_env.py --expect-arch sm_90 --strict
python scripts/check_env.py --expect-arch sm_103 --strict

# Benchmark modes are separate
python benchmarks/bench_attention.py --ladder smoke --impl fa4 --mode fwd
python benchmarks/bench_attention.py --ladder smoke --impl fa4 --mode bwd
python benchmarks/bench_attention.py --ladder smoke --impl fa4 --mode fwd_bwd
```

## Current boundary

The starter bundle establishes the contract and workflow. It does not claim an
SM90/SM103 d=512 kernel, local SM103 d=256 support, sanitizer-clean GPU code,
or any speedup. See `docs/status.md` for the exact next hardware actions.
