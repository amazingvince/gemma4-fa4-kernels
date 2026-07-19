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
- a hash-locked H100 FA4 patch plus validated fixed and packed local-d256
  forward/autograd-backward adapters, exact local vision/document masking,
  and an exact composed global-d512 text forward/backward adapter;
- a complete `writing-cute-dsl-kernels` agent skill and project router;
- H100/B300 SSH profile placeholders, remote sync/run/collect scripts, and a
  guarded target-specific CUDA toolkit installer for Ubuntu 24.04;
- prompts for bootstrap, M0 baselines, and the first kernel task.

## Environment policies reviewed 2026-07-19

- H100/SM90: **CUDA 12.8**, PyTorch **2.8.0+cu128**, FA4 `[dev]`;
- B300/SM103: **CUDA 13.3**, PyTorch **2.13.0+cu132**,
  FA4 `[dev,cu13]`;
- both: Python **3.12** and FA4-pinned `nvidia-cutlass-dsl`
  **4.6.0.dev0**, with the successfully resolved `quack-kernels` runtime
  helper fixed at **0.5.3**.

Profiles use separate `.venv-h100` and `.venv-b300` environments. See
[`docs/environment.md`](docs/environment.md).

## Local contract check

```bash
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
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
3. `docs/status.md`
4. `docs/design.md`
5. `docs/experimental-plan.md`
6. `skills/gemma4-kernel-project/SKILL.md`
7. `skills/writing-cute-dsl-kernels/SKILL.md`
8. `prompts/README.md`
9. `BUNDLE_MANIFEST.md`, `VERIFICATION.md`, and `docs/source-index.md`

The current H100 gate results are recorded in `docs/status.md`. Fixed local
d256 text and multimodal paths, packed local d256 native/custom paths, and
composed global d512 text forward/backward pass their declared M1 envelopes.
EXP-0005 remains the historical rejection of unchanged unequal-dimension
GQA-8 backward; EXP-0006 accepts the correctness-first split composition.
EXP-0007 accepts exact fixed B1 vision masking, and EXP-0008 accepts nonempty
packed local self-attention with `B>=1` and `1 <= Sq <= Sk <= 1025`, including
lower-right alignment and K-stream vision/document IDs. These are scoped
correctness results, not performance or B300 claims. Production local context
above 1025, generic framework dispatch, and all tuning remain unverified.
