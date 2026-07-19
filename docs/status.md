# Status

## Established in the starter bundle

- revision-pinned Gemma 4 31B contract;
- corrected scale, K/V preparation, local vision mask, GQA geometry, and
  explicit confirmation that cross-layer KV sharing is disabled;
- CPU reference and optional pinned Transformers oracle;
- complete CuTe DSL skill package integrated;
- pinned Transformers implementation audit, including a strict expected failure for the generic FA4 vision-mask adapter;
- latest-compatible environment policy and remote-host placeholders;
- benchmark and experiment scaffolding.

## Not established without target hardware

- FA4 import/compile success on CUDA 13.3 + PyTorch 2.13.0;
- SM90/SM103 fake-tensor and real execution matrices;
- local d=256 B300 correctness;
- any dense d=512 kernel correctness;
- sanitizer cleanliness;
- register/TMEM/SMEM budgets from generated code;
- performance or end-to-end speedups.

## First hardware actions

1. Bootstrap each host and save `scripts/check_env.py --strict` output.
2. Run `python scripts/verify_model_contract.py --transformers`.
3. Run the upstream FA4 fake-tensor compile pass, then real tests using the
   persistent cache.
4. Capture rooflines and semantically valid baselines.
5. Start M1 with B300 local d=256 fixed-length BF16 forward correctness.
6. In parallel, complete design briefs for H100 and B300 global d=512 forward.
