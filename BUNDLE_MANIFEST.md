# Starter bundle manifest

## Purpose

This bundle is a correctness-first launch point for Gemma 4 31B
FlashAttention-4 forward/backward research on H100 and B300. It does not contain
a claimed optimized kernel.

## Included contracts and tooling

- exact, revision-pinned checkpoint and Transformers attention contract;
- separate prepared Q/K/V semantics and separate dK/dV;
- explicit scale 1.0 and exact local vision-mask predicate;
- full CuTe DSL skill package v1.1.0 with its original checksums;
- pinned FlashAttention/Transformers revisions and an exact hash-locked,
  one-file H100 Transformers integration patch;
- exact CuTe DSL and QuACK runtime-helper versions;
- one hash-locked, license-noticed H100 FlashAttention patch;
- validated fixed and packed H100 local-d256 plus composed global-d512 text
  forward and autograd-backward adapters, with exact local vision/document
  masking and O/LSE/gradient evidence;
- an eager pinned-Transformers integration under the unique
  `gemma4_fa4_h100` backend name, with exact fixed and packed-varlen dispatch
  over its declared H100 envelope;
- reviewed target-specific CUDA/PyTorch/DSL environment policies;
- SSH/module/Slurm-capable remote profile placeholders;
- guarded Ubuntu host-prerequisite and CUDA-toolkit-only installers;
- offline verifier, optional HF oracle, CPU CI, lint/static checks;
- semantics-aware benchmark ladders and experiment records;
- starter prompts and architecture design briefs.

## Deliberately not claimed

- H100 fixed/packed local d256 and composed global d512 text
  forward/autograd backward passed their declared compile, numerical, stream,
  sanitizer, cache, and generated-code gates; the global path is a
  correctness-first multi-launch composition, not an optimized fused kernel;
- no CUDA/driver changes were made on a remote host;
- the pinned Transformers oracle is skipped in the local environment and runs
  against the exact patched checkout on the remote H100;
- the H100 CUDA 12.8/PyTorch 2.8 cu128/FA4 `[dev]` gate passed; the B300 CUDA
  13.3/PyTorch 2.13 cu132/FA4 `[dev,cu13]` gate remains unrun;
- EXP-0003's frozen elementwise dQ/dK envelope remains rejected; EXP-0004
  separately accepts the unchanged local backward under the predeclared
  upstream-relative BF16 oracle, exact model GQA-2 boundary matrix,
  stream/repeat checks, sanitizers at S128/S129, and generated-code inspection;
- EXP-0005 preserves the unchanged direct global-backward rejection; EXP-0006
  accepts the split dKV-only/D256-dQ-only H100 path through S1024;
- EXP-0007 accepts fixed B1 local multimodal masking; EXP-0008 accepts
  nonempty packed local self-attention through per-sequence S1025;
- EXP-0009 accepts native packed local text through per-sequence S262144 on
  H100 SM90, with no deterministic-gradient or performance claim;
- EXP-0010 accepts exact packed local vision/document metadata through
  per-sequence S262144 when the tile schedule fits the declared padded-work,
  metadata, and free-HBM safety envelope;
- EXP-0011 accepts eager pinned-Transformers dispatch under the
  unique `gemma4_fa4_h100` backend after all 8 integration probe cases passed;
  its exact one-file patch SHA256 is
  `773950a1f1feb04f5f2e6a1d66f8953ff8905e8ca9391f804089f169da59b671`;
- the global EXP-0011 gates include B2/S5 composed training, Q33/K2048 packed
  no-grad, and the exact Q1/K262144 forward-only sentinel; memcheck,
  synccheck, and racecheck are clean for the first two varlen routes;
- the isolated integration compile cache contains 15 paths representing 9
  unique contents and 976336 bytes;
- empty segments, over-budget sparse schedules, `torch.compile`, static-cache
  support, global backward beyond K1024, and benchmarks remain unclaimed;
- no speedup or B300 correctness claim exists.

See `VERIFICATION.md` for the assembly evidence and explicit unrun checks.

## Verification command

```bash
bash scripts/verify_bundle.sh
```

On a bootstrapped GPU host, additionally run:

```bash
python scripts/check_env.py --profile h100 --expect-arch sm_90 --strict \
  --require-transformers --require-profilers
# or: --profile b300 --expect-arch sm_103
python scripts/verify_model_contract.py --online --transformers
```
