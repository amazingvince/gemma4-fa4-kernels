# Gemma 4 × FlashAttention-4 kernel lab

A correctness-first starter repository for developing BF16 forward and backward
CuTe DSL attention kernels for **Gemma 4 31B**, targeting **H100 (SM90a)** and
**B300 / Blackwell Ultra (SM103)**.

The repository deliberately separates two kernel families:

| Layer family | Count | Full-model shape | Mask |
|---|---:|---|---|
| Sliding | 50 | 32 Q heads, 16 KV heads, d=256 | `(k > q-1024) AND (causal OR same vision block)` |
| Full | 10 | 32 Q heads, 4 KV heads, d=512 | causal |

## Correctness warning that drives the design

`attention_k_eq_v=true` means the **global K projection output is reused as the
source for V**. It does not mean K and V are equal at the FMHA boundary. K is
K-normalized and partially rotated; V is separately normalized and not
rotated. Attention-only kernels must accept separate K and V and return
separate dK and dV.

Gemma 4 also uses `softmax_scale=1.0`, not the usual `1/sqrt(d)`.

Read [`docs/model-contract.md`](docs/model-contract.md) before editing a
kernel. The contract is executable in `src/gemma4_fa4/` and `tests/`.

## What is included

- a revision-pinned Gemma 4 31B model contract;
- a high-precision reference and exact local multimodal mask;
- CPU tests plus an optional Transformers oracle suite;
- a semantics-aware benchmark skeleton with distinct fwd/bwd/fwd+bwd modes;
- pinned upstream revisions for Transformers and FlashAttention;
- a complete `writing-cute-dsl-kernels` agent skill and project router;
- H100/B300 SSH profile placeholders, remote sync/run/collect scripts, and a
  guarded CUDA 13.3 toolkit installer for Ubuntu 24.04;
- prompts for bootstrap, M0 baselines, and the first kernel task.

## Environment policy reviewed 2026-07-19

- host toolkit: **CUDA 13.3 GA**;
- PyTorch: **2.13.0**, official **cu132** wheel;
- FA4-pinned `nvidia-cutlass-dsl`: **4.6.0.dev0**;
- Python: **3.12**;
- full CUDA 13.3 feature policy: NVIDIA driver **610.43.02 or newer**.

The host toolkit and PyTorch wheel runtime are intentionally different minor
versions. See [`docs/environment.md`](docs/environment.md).

## Local contract check

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'
bash scripts/verify_bundle.sh
```

## Remote GPU quick start

```bash
cp remote/h100.env.example remote/h100.env
cp remote/b300.env.example remote/b300.env
# Fill SSH details; never commit the non-example files.

bash scripts/remote/probe.sh h100
bash scripts/remote/sync.sh h100
bash scripts/remote/bootstrap.sh h100
bash scripts/remote/check.sh h100
bash scripts/remote/run.sh h100 pytest -q
```

Use the same commands with `b300`. System-level CUDA/driver setup is separated
from user-space Python setup and never runs implicitly.

## Start here

For the line-by-line Transformers semantic audit, read
[`docs/hf-implementation-audit.md`](docs/hf-implementation-audit.md).

1. `AGENTS.md`
2. `docs/model-contract.md`
3. `docs/design.md`
4. `docs/experimental-plan.md`
5. `skills/writing-cute-dsl-kernels/SKILL.md`
6. `prompts/README.md` and `prompts/00-bootstrap-and-m0.md`
7. `BUNDLE_MANIFEST.md` and `VERIFICATION.md`
8. `docs/source-index.md`

No target GPU execution is claimed by the starter bundle itself. Hardware
correctness, sanitizer evidence, and performance begin in M0/M1.
