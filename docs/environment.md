# Environment policy — CUDA, PyTorch, and FA4

**Reviewed:** 2026-07-19

## Target-specific compatible stacks

| Component | H100 / SM90 | B300 / SM103 |
|---|---|---|
| Python | 3.12 | 3.12 |
| Host CUDA Toolkit | 12.8 | 13.3 |
| Driver policy | 570.26.00 or newer | 610.43.02 or newer |
| PyTorch | 2.8.0 | 2.13.0 |
| Official wheel runtime | cu128 / CUDA 12.8 | cu132 / CUDA 13.2 |
| FA4 extra | `[dev]` | `[dev,cu13]` |
| FA4 upstream | `77aacb68...` | `77aacb68...` |
| Project patch | exact combined SM90 global forward/backward patch | none |
| `nvidia-cutlass-dsl` | 4.6.0.dev0 | 4.6.0.dev0 |
| `quack-kernels` | 0.5.3 | 0.5.3 |
| Transformers oracle | `7ea2320c...` | `7ea2320c...` |

The split is required by the pinned FA4 package itself:
`flash_attn/cute/README.md` specifies `[dev]` on CUDA 12.x and reserves `cu13`
for CUDA 13.x. H100 uses the upstream-tested CUDA 12.x path; B300 keeps the
reviewed CUDA 13.x policy. Do not install B300 extras on H100.

Do not force a CUDA toolkit minor version to equal the wheel label by
installing an unofficial PyTorch build.

## Host administrator setup

Required host capabilities:

- supported NVIDIA datacenter GPU and production driver;
- target-policy CUDA Toolkit (12.8 on H100, 13.3 on B300), including `nvcc`,
  `ptxas`, `nvdisasm`, `cuobjdump`, and Compute Sanitizer;
- Nsight Compute and Nsight Systems;
- Python 3.12, git, build tools, and enough local disk for JIT caches;
- clock/power controls where the lab permits them.

For Ubuntu 24.04, an explicit toolkit-only helper is provided:

```bash
sudo ALLOW_SYSTEM_CHANGES=1 bash scripts/remote/install_cuda_ubuntu.sh h100
# or: ... install_cuda_ubuntu.sh b300
```

It does not replace the NVIDIA driver. Driver upgrades can affect all users,
require a reboot, and must be coordinated with the host owner.

## User-space setup

```bash
bash scripts/setup_env.sh h100
source .venv-h100/bin/activate
python scripts/check_env.py --profile h100 --expect-arch sm_90 --strict --require-transformers

# On B300 instead:
bash scripts/setup_env.sh b300
source .venv-b300/bin/activate
python scripts/check_env.py --profile b300 --expect-arch sm_103 --strict --require-transformers
```

`--strict` enforces the correctness toolchain. Add `--require-profilers` before
the benchmark gate; it additionally requires Nsight Systems.

The exact official URLs and revisions are listed in `docs/source-index.md`.

The setup script uses separate `.venv-h100` and `.venv-b300` environments so
CUDA-13 DSL libraries cannot leak into the Hopper environment. It:

1. installs the target profile's pinned PyTorch wheel;
2. installs this package and development tools;
3. checks out pinned FlashAttention and Transformers revisions under
   `.upstream/`;
4. on H100, verifies and applies the one hash-locked combined patch; strict
   checks reject a missing patch or any extra checkout change;
5. installs FA4 using the target's exact extras (`dev` on H100,
   `dev,cu13` on B300) and pins its QuACK helper to the H100-resolved 0.5.3;
6. runs the offline contract, optional Transformers oracle, a strict
   GPU/tool/runtime/import-provenance check, and the full static bundle
   verifier. The remote wrapper repeats the strict check after setup.

Set `INSTALL_HF_ORACLE=0` to omit the Transformers checkout.

## Recommended runtime variables

```bash
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/fast/local/cache/fa4
export CUTE_DSL_KEEP_PTX=1        # only when inspecting codegen
export CUTE_DSL_LINEINFO=1        # profiling/sanitizer sessions
```

Keep cache paths architecture- and revision-specific when hosts share storage.
Never reuse an SM90 cubin as SM103 or across incompatible DSL/toolkit revisions.

## Upgrade procedure

“Latest” is a reviewed compatibility set, not an automatic floating install.
To refresh:

1. update one component in a branch;
2. update the affected target policy (`configs/env/h100-compatible.env` or
   `configs/env/latest-compatible.env`) and `upstream.lock.json`;
3. run model-oracle, compile, sanitizer, benchmark, and cache-reuse matrices on
   both GPUs;
4. record the environment change as an experiment boundary;
5. only then replace the default policy.
