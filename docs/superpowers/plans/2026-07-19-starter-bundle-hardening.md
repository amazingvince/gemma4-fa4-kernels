# Gemma 4 FA4 Starter Bundle Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use the project's
> `writing-cute-dsl-kernels` skill for GPU work and verification-before-completion
> before any completion claim.

**Goal:** Establish an executable, revision-pinned Gemma 4 contract and a
reproducible local/remote research environment before kernel implementation.

**Architecture:** The model contract is isolated in `src/gemma4_fa4`; remote
transport is separated from host system provisioning and user-space setup;
upstream sources are exact detached checkouts; GPU work consumes the full CuTe
DSL skill and experiment ledger.

**Tech Stack:** Python 3.12, PyTorch 2.13.0 cu132, CUDA Toolkit 13.3,
CuTe DSL 4.6.0.dev0, pytest, ruff, SSH/rsync.

## Global Constraints

- Prepared global K and V are distinct operands.
- Softmax scale is 1.0.
- No GPU correctness/performance claim without target-hardware evidence.
- Upstream and environment revisions are pinned.
- Remote credentials remain untracked.

### Task 1: Model contract and reference

**Files:** `configs/model/gemma4-31b.lock.json`, `src/gemma4_fa4/*`, `tests/*`.

- [x] Lock exact 31B geometry, layer pattern, scale, RoPE, and explicitly disabled cross-layer KV sharing.
- [x] Implement exact local/global masks and high-precision attention.
- [x] Test scale, window edge, vision block, GQA, LSE, and separate gradients.
- [x] Add optional pinned Transformers oracle.

### Task 2: Versioned environment and remote profiles

**Files:** `configs/env/*`, `upstream.lock.json`, `scripts/setup_env.sh`,
`scripts/check_env.py`, `scripts/remote/*`, `remote/*`.

- [x] Pin the reviewed latest-compatible environment.
- [x] Separate driver/toolkit administration from user setup.
- [x] Add SSH sync, bootstrap, check, run, and collection placeholders.
- [ ] Execute strict checks on actual H100 and B300 hosts.

### Task 3: Measurement and experiment contract

**Files:** `benchmarks/*`, `experiments/*`, `scripts/record_result.py`.

- [x] Separate fwd, bwd, and fwd+bwd timings.
- [x] Reject semantically unequal baselines.
- [x] Add memory preflight, L2 modes, robust statistics, and provenance.
- [ ] Capture M0 hardware baselines and profiles.

### Task 4: Agent workflow and CuTe skill

**Files:** `AGENTS.md`, `skills/*`, `prompts/*`, `docs/*`.

- [x] Vendor and validate the complete skill package.
- [x] Route project tasks through the model contract and complete skill.
- [x] Add bootstrap, M0, first-kernel, and d512-design prompts.
- [ ] Run fresh-context skill pressure evaluations on GPU tasks.

### Task 5: Hardware implementation

- [ ] B300 local d256 forward correctness.
- [ ] B300 local backward and multimodal boundaries.
- [ ] H100/B300 global d512 forward design and implementation.
- [ ] Owner-computes dQ/dK/dV.
- [ ] Framework integration and end-to-end model-weighted measurement.
