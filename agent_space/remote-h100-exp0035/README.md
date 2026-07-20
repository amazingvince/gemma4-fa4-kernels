# EXP-0035 H100 scratch evidence

This directory contains the isolated experimental delta and H100 gate outputs
for the accepted exact-BF16 global d512 D256-streaming path.

- `candidate.patch` is the EXP-0035 delta that applies on top of the prior
  accepted upstream patch; the repository's combined patch now includes it.
- The stream path is enabled by default. Set
  `FLASH_ATTENTION_GEMMA4_EXPERIMENT_DQ_D256_STREAM=0` for the retained slab
  dQ fallback.
- Files suffixed `barrierfix` are the final sanitizer-clean implementation.
- S8K results use 10 warmups/30 repetitions. Final S64K `confirm-barrierfix`
  rows also use 10/30; `screen-barrierfix` rows use 2/5.
- Clocks were unlocked. Claims are scoped to the recorded H100, exact BF16
  global causal Gemma 4 contract, and semantically equivalent baselines.
