# H100 M0 status and handoff

**Status date:** 2026-07-19

The repository has a revision-pinned Gemma 4 31B attention contract, CPU
reference, optional pinned Transformers oracle, FA4 integration audit,
environment policy, remote workflow, benchmark scaffolding, and CuTe DSL skill
package. The established semantics include scale `1.0`, distinct prepared K
and V operands, separate dK/dV attention gradients, the strict local window
and vision overlay, GQA geometry, and disabled cross-layer KV sharing.

## Current H100 probe: non-policy smoke evidence only

| Item | Observed 2026-07-19 | Policy status |
|---|---|---|
| GPU | H100, 80 GB HBM3 | Target architecture: compute capability 9.0 |
| NVIDIA driver | 580.126.09 | Below the required `>=610.43.02` |
| CUDA toolkit | 12.8 | Below the required 13.3 |
| PyTorch | 2.8.0+cu128 | Not the required 2.13.0+cu132 wheel |
| CuTe DSL | Not validated on this image | Required: 4.6.0.dev0 |

Strict environment validation failed on this current image against the
unchanged policy: CUDA 13.3, driver `>=610.43.02`, PyTorch 2.13.0+cu132, and
CuTe DSL 4.6.0.dev0. Consequently, this repository has no H100 FA4 compile,
execution, sanitizer, correctness, or performance claim.

The current image ran only the model-contract/CPU suite and exploratory
SDPA/roofline commands. Collected artifacts are ignored, unlocked,
current-image smoke data; they cannot support an FA4 comparison or a
performance claim.

## Active H100 M0 sequence

Do not bootstrap or run FA4 validation on the current noncompliant image.
First probe and check the H100 profile:

```bash
bash scripts/remote/probe.sh h100
bash scripts/remote/check.sh h100

# Only after the host/image satisfies the unchanged policy:
bash scripts/remote/bootstrap.sh h100
bash scripts/remote/run.sh h100 python scripts/verify_model_contract.py --transformers
bash scripts/remote/run.sh h100 pytest -q tests/test_hf_oracle_optional.py
bash scripts/remote/run.sh h100 env FLASH_ATTENTION_FAKE_TENSOR=1 \
  pytest -q .upstream/flash-attention/tests/cute/test_flash_attn.py
bash scripts/remote/run.sh h100 \
  pytest -q .upstream/flash-attention/tests/cute/test_flash_attn.py
```

Only after the real FA4 correctness pass succeeds, run the sanitizer follow-up
for each newly exercised memory or synchronization protocol:

```bash
bash scripts/remote/run.sh h100 compute-sanitizer --tool memcheck \
  pytest -q .upstream/flash-attention/tests/cute/test_flash_attn.py
bash scripts/remote/run.sh h100 compute-sanitizer --tool synccheck \
  pytest -q .upstream/flash-attention/tests/cute/test_flash_attn.py
bash scripts/remote/run.sh h100 compute-sanitizer --tool racecheck \
  pytest -q .upstream/flash-attention/tests/cute/test_flash_attn.py
```

Only after correctness and sanitizer passes, record unlocked status or lock
clocks according to lab policy, then collect denominators and semantically
equivalent H100 measurements in separate modes:

```bash
bash scripts/remote/run.sh h100 python benchmarks/roofline.py \
  --out agent_space/h100-roofline.json
bash scripts/remote/run.sh h100 python benchmarks/bench_attention.py \
  --ladder smoke --impl fa4 --mode fwd
bash scripts/remote/run.sh h100 python benchmarks/bench_attention.py \
  --ladder smoke --impl fa4 --mode bwd
bash scripts/remote/run.sh h100 python benchmarks/bench_attention.py \
  --ladder smoke --impl fa4 --mode fwd_bwd
```

## Scope boundary

The long-term program retains both H100 and B300/SM103 targets, but B300/SM103
actions are explicitly deferred from this active H100 M0 handoff. No H100
result may be generalized to the deferred target.
