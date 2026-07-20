# Axolotl Gemma 4 12B FA4 smoke harness

This harness performs eight real BF16 LoRA forward/backward steps on
`google/gemma-4-12B-it`, with three warmups and five measured steps. The learning
rate is exactly zero, so it exercises training and gradients without evolving the
weights or pretending to be a useful long training run.

It compares three paths on the identical B1/S1024 dataset:

- `hybrid`: Axolotl's Gemma 4 default, using FlashAttention-2 for local d256 and
  packing-aware SDPA for global d512.
- `sdpa`: Transformers SDPA throughout, used as a second correctness baseline.
- `project_12b_compat`: this repository's H100 FA4 local/global kernels through an
  explicit 12B compatibility mapping.

The generator emits the modern top-level `messages` schema required by Gemma 4
Unified's runtime multimodal collator. Each record is exactly 1024 tokens under
the pinned tokenizer. The collator therefore consumes raw messages with
`skip_prepare_dataset: true`; no truncation, dropped sample, or variable batch
width is hidden in the measurement.

## What the 12B compatibility route proves

The accepted kernels are specialized for Gemma 4 31B geometry. Gemma 4 12B uses
16 query heads, 8 local KV heads, and 1 global KV head. The adapter maps local
16Q/8KV to 32Q/16KV with zero heads. For global attention it maps 16Q/1KV to
32Q/4KV as `[K, K, 0, 0]` and `[V, V, 0, 0]`; the two copies give the first 16
query heads the original GQA-16 semantics under the kernel's GQA-8 mapping.
Outputs are sliced back to 16 heads, and autograd sums the two global KV-gradient
copies. CPU reference tests cover O, dQ, dK, and dV for both layer families.

This proves real-model integration and semantics. It is not a native 12B kernel:
the zero-padded work can make it slower. The harness reports that outcome directly.

## Pinned boundary and prerequisites

The lock is in `configs/model/gemma4-12b-harness.lock.json`. The run requires one
H100 with at least 70 GiB visible HBM and access to Google's gated model. Set an
Hugging Face token after accepting the model license:

```bash
export HF_TOKEN=...
bash scripts/axolotl/bootstrap_env.sh
AXOLOTL_VENV_DIR=.venv-h100-axolotl-baseline \
  INSTALL_AXOLOTL_FA2=1 INSTALL_PROJECT_FA4=0 \
  bash scripts/axolotl/bootstrap_env.sh
.venv-h100-axolotl/bin/python scripts/axolotl/check_env.py
.venv-h100-axolotl-baseline/bin/python scripts/axolotl/check_env.py --skip-fa4
```

The bootstraps use separate environments because current Axolotl requires newer
PyTorch than the accepted project H100 profile. It installs Axolotl at the locked
commit and the patched Transformers revision in both. The baseline venv owns FA2;
the project venv owns FA4. They must remain separate because both distributions own
the `flash_attn` Python package. These are EXP-0036 candidate environments only;
they do not widen `configs/env/h100-compatible.env`.

## Run locally on an H100

```bash
bash scripts/axolotl/run_h100_matrix.sh
```

Or use the configured remote workflow:

```bash
bash scripts/remote/sync.sh h100
bash scripts/remote/run.sh h100 bash scripts/axolotl/bootstrap_env.sh
bash scripts/remote/run.sh h100 bash scripts/axolotl/run_h100_matrix.sh
bash scripts/remote/collect.sh h100
```

The runner creates a timestamped ignored directory under
`agent_space/axolotl-exp0036/`. Each backend gets a log and JSON report, followed
by `hybrid-vs-project.json` and `sdpa-vs-project.json`. Its FA4 compile cache is
isolated under that timestamped directory so concurrent tasks do not share the
harness's cache. Both comparison JSON files are written even when a correctness
gate rejects the candidate; the runner still exits nonzero.

## Acceptance gates

The comparator exits nonzero unless all gates pass:

- model revision, dataset hash, sequence length, batch/accumulation, step count,
  warmup count, and seed are identical;
- every process resets LoRA A from stable per-parameter name seeds, zeros LoRA B,
  and records the same SHA-256 initialization fingerprint before training;
- all 48 project attention layers are observed, with 40 local fixed FA4 routes and
  8 global fixed FA4 routes, and aggregate counts match per-layer evidence;
- no project fallback, FlexAttention, eager, or SDPA route is observed;
- measured losses satisfy `atol=5e-3, rtol=2e-3`;
- the bounded, deterministic LoRA-gradient sketch has cosine at least `0.999` and
  relative L2 at most `0.01`;
- both reports contain five positive CUDA-event step timings.

Only after those gates pass does the comparator compute median, p25, p75, IQR, and
baseline-median / candidate-median speedup. A speedup below 1.0 is a valid rejection
of the compatibility route's performance, not a correctness failure.

To compare existing reports directly:

```bash
python scripts/compare_axolotl_runs.py baseline/report.json project/report.json
```
