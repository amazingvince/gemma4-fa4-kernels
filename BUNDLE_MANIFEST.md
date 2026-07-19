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
- pinned FlashAttention/Transformers revisions;
- reviewed CUDA/PyTorch/DSL environment policy;
- SSH/module/Slurm-capable remote profile placeholders;
- guarded Ubuntu host-prerequisite and CUDA-toolkit-only installers;
- offline verifier, optional HF oracle, CPU CI, lint/static checks;
- semantics-aware benchmark ladders and experiment records;
- starter prompts and architecture design briefs.

## Deliberately not claimed

- no H100 or B300 code was compiled or executed while assembling this bundle;
- no CUDA/driver changes were made on a remote host;
- the optional pinned Transformers oracle requires that checkout to be
  installed and is skipped in a minimal CPU environment;
- CUDA 13.3 + PyTorch 2.13 cu132 + FA4/DSL compatibility must be verified on
  both target machines;
- no correctness, sanitizer, or performance claim exists for a new kernel.

See `VERIFICATION.md` for the assembly evidence and explicit unrun checks.

## Verification command

```bash
bash scripts/verify_bundle.sh
```

On a bootstrapped GPU host, additionally run:

```bash
python scripts/check_env.py --expect-arch sm_90 --strict --require-transformers
# or sm_103
python scripts/verify_model_contract.py --online --transformers
```
