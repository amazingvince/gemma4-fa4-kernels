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

- `upstream.lock.json` pins Transformers and FlashAttention revisions plus the
  reviewed H100 patch path/hash.
- `configs/env/h100-compatible.env` pins the CUDA-12 Hopper policy;
  `configs/env/latest-compatible.env` pins the CUDA-13 B300 policy.
- `scripts/setup_env.sh` checks out the exact revisions under `.upstream/` and
  applies only the profile-declared patch stack.
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
python scripts/check_env.py --profile h100 --expect-arch sm_90 --strict
python scripts/check_env.py --profile b300 --expect-arch sm_103 --strict

# Benchmark modes are separate
python benchmarks/bench_attention.py --ladder smoke --impl fa4 --mode fwd
python benchmarks/bench_attention.py --ladder smoke --impl fa4 --mode bwd
python benchmarks/bench_attention.py --ladder smoke --impl fa4 --mode fwd_bwd
```

## Current boundary

H100 fixed-length local d256 text forward/autograd backward and an exact
two-launch global d512 text forward composition are validated over their
declared M1 envelopes. The latter is not a fused or optimized d512 kernel.
EXP-0003's fixed elementwise dQ/dK envelope remains rejected; EXP-0004 kept
that result intact and accepted the unchanged backward kernel under a
predeclared upstream-relative BF16 oracle, including boundary, stream,
repeat, sanitizer, and generated-code gates. Global d512 backward is the active
ordered gate. Multimodal kernels, benchmarks, and SM103/B300 remain unrun. See
`docs/status.md` before hardware work and never loosen a recorded experiment's
policy after observing its result. EXP-0005 rejects direct GQA-8 backward for
the asymmetric d512-QK/d256-V slabs at the pinned constructor. Head expansion
alone still exceeds the monolithic register/SMEM budgets; the next global
backward experiment must preserve model geometry while structurally splitting
dQ from dKV or chunking dQ in D.
