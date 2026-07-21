# H100 production-candidate handoff

This document is the release boundary for the exact-BF16 Gemma 4 31B H100
work through EXP-0044. It describes what is accepted, how to reproduce it,
and what remains outside the claim. It does not claim B300 compatibility,
general FlashAttention semantics, training convergence, or an FP8 path.

## Pinned source stack

- Project branch: `codex/h100-exp0038-dq-single-launch`
- FlashAttention base: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- H100 cumulative patch:
  `patches/flash-attention/0004-sm90-gemma4-forward-d512-single-launch.patch`
- H100 patch SHA256:
  `9d14635e23199200f0b25cbd9d33f464d1dd119a527838cc96098e9b8b3d91dd`
- Transformers base: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Transformers integration patch:
  `patches/transformers/0001-gemma4-forward-vision-block-ids.patch`
- Transformers patch SHA256:
  `ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79`
- Gemma 4 31B model revision:
  `2d418d1b7ed8c04d732c3359e19a11fbc85b6842`

`scripts/check_env.py --expect-arch sm_90 --strict` is the authoritative
check that both upstream trees match these bases and exact patch diffs, that
imports resolve from the pinned checkouts, and that no extra upstream changes
are present.

## Accepted H100 boundary

All accepted routes use BF16 inputs and outputs, FP32 score/LSE/gradient
accumulation where declared, dropout zero, and `softmax_scale=1.0`.

| Layer family | Exact geometry | Accepted fixed/native boundary |
| --- | --- | --- |
| Local | 32 Q heads, 16 KV heads, d256, GQA 2, sliding/vision mask | Exact text and multimodal fixed paths; native packed paths through S262144 subject to the declared schedule, metadata, INT32, and HBM admission guards |
| Global | 32 Q heads, 4 KV heads, d512, GQA 8, causal, distinct K/V | Cooperative fixed/native forward; fixed training through S2048 and native packed training through S262144 subject to the declared HBM guards |

The global fast default has one cooperative forward launch and two main
backward launches: one full-D dQ launch plus one owner-computes full-D dKV
launch. The dKV owner route removes the internal FP32 dK/dV workspace and its
postprocess launches. The dQ FP32 workspace remains, so the complete backward
does not yet have a bounded-memory claim.

The pinned Transformers integration accepts eager execution for every one of
the 60 text layers. The explicitly guarded no-cache facade accepts all 60
layer indices for B1 text at S1..1024 under eager and Inductor. The guarded
cache facade accepts Q1 decode for all 60 layer indices after eager prefill.
These claims do not imply raw `torch.compile(layer)`, compiled prefill, or
full-model compilation.

## Tested rollback controls

Each accepted global structural change retains its exact previously accepted
route:

- `FLASH_ATTENTION_GEMMA4_EXPERIMENT_FORWARD_D512_SINGLE_LAUNCH=0` restores
  the exact two-V256 global forward composition.
- `FLASH_ATTENTION_GEMMA4_EXPERIMENT_DQ_D512_SINGLE_LAUNCH=0` restores the
  accepted split-dQ global backward route.
- `FLASH_ATTENTION_GEMMA4_EXPERIMENT_OWNER_DKV=0` restores the accepted
  full-D dKV route with FP32 dK/dV workspace and postprocessing.
- `deterministic=True` selects the separately validated, bitwise-repeatable
  global backward route; it is intentionally not the fast default.

These controls are rollback mechanisms, not independent supported products.
Changing more than one at a time requires rerunning the applicable experiment
gates rather than assuming compositional coverage.

## Release evidence

EXP-0044 is the final production-soak gate. Its accepted replay records:

- local CPU suite: `451 passed, 106 skipped`;
- fresh-cache H100 suite: `544 passed, 17 skipped, 1 xfailed`;
- pinned Transformers oracle: `5 passed, 1 xfailed` for the documented
  generic FA4 mask-adapter gap;
- result-schema validation: 41 records;
- compileall, Ruff check/format, locked model-contract verification, strict
  exact-patch environment verification, and the complete bundle: passed;
- global S65536 forward+backward: two warmups plus five samples completed;
- global Q1/K262144 analytic backward: exact and bitwise across three
  nondefault-stream repetitions;
- local Q=K=262144: three fresh seeds and streams passed after HBM preflight;
- every soak child exited zero, released its allocation, and left no compute
  process; the FA4 disk cache stabilized at 22 files and 1,199,032 bytes.

The reported S65536 timing in EXP-0044 is a lifecycle diagnostic only. It is
not a new speedup claim. Full commands, outputs, cache inventory, and strict
environment report are under `agent_space/remote-h100-exp0044/` and
`experiments/EXP-0044-h100-production-soak.md`.

## Remaining production work

The following remain deliberately outside the accepted boundary:

- B300/SM103 implementation and validation;
- raw or full-model `torch.compile`, compiled prefill, cached multimodal
  decode, and varlen facade integration;
- deterministic local gradients and all-empty physical packed workloads;
- schedules rejected by the HBM or padded-work admission guards;
- bounded-memory global dQ backward;
- FP8 backward/training and any FP8 d512 single-warp proposal;
- training-convergence validation and end-to-end serving or training SLOs.

Start any widening as a new predeclared experiment. Preserve the model lock,
semantic oracles, sanitizer gates, exact upstream patch checks, and shared-GPU
lease wrapper described in `AGENTS.md` and `docs/status.md`.
