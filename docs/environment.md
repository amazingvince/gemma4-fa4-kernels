# Environment policy — CUDA, PyTorch, and FA4

**Reviewed:** 2026-07-19

## Selected latest-compatible stack

| Component | Policy |
|---|---|
| Python | 3.12 |
| Host CUDA Toolkit | 13.3 GA |
| Driver for full CUDA 13.3 feature set | 610.43.02 or newer on Linux |
| PyTorch | 2.13.0 |
| Official PyTorch wheel runtime | cu132 / CUDA runtime 13.2 |
| FA4 upstream | `77aacb68...` |
| `nvidia-cutlass-dsl` | 4.6.0.dev0, matching FA4 upstream |
| Transformers oracle | `7ea2320c...` |

CUDA 13.4 is a developer preview in the archive at this audit date, so it is
not the default research environment. PyTorch 2.13.0 publishes an official
CUDA 13.2 wheel, not a cu133 wheel. It is normal to compile CuTe DSL kernels
with the host CUDA 13.3 toolchain while PyTorch ships cu132 runtime libraries;
the NVIDIA driver provides compatibility.

Do not force a CUDA toolkit minor version to equal the wheel label by
installing an unofficial PyTorch build.

## Host administrator setup

Required host capabilities:

- supported NVIDIA datacenter GPU and production driver;
- CUDA Toolkit 13.3 including `nvcc`, `ptxas`, `nvdisasm`, `cuobjdump`, and
  Compute Sanitizer;
- Nsight Compute and Nsight Systems;
- Python 3.12, git, build tools, and enough local disk for JIT caches;
- clock/power controls where the lab permits them.

For Ubuntu 24.04, an explicit toolkit-only helper is provided:

```bash
sudo ALLOW_SYSTEM_CHANGES=1 bash scripts/remote/install_cuda_ubuntu.sh
```

It does not replace the NVIDIA driver. Driver upgrades can affect all users,
require a reboot, and must be coordinated with the host owner.

## User-space setup

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_env.py --expect-arch sm_90 --strict --require-transformers   # H100
python scripts/check_env.py --expect-arch sm_103 --strict --require-transformers  # B300
```

The exact official URLs and revisions are listed in `docs/source-index.md`.

The setup script:

1. installs PyTorch 2.13.0 from the official cu132 index;
2. installs this package and development tools;
3. checks out pinned FlashAttention and Transformers revisions under
   `.upstream/`;
4. installs FA4 using its pinned CuTe DSL dependency and `cu13` extra;
5. runs the offline contract, optional Transformers oracle, full static bundle
   verifier, and then the remote wrapper runs a strict GPU/tool/upstream check.

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
2. update `configs/env/latest-compatible.env` and `upstream.lock.json`;
3. run model-oracle, compile, sanitizer, benchmark, and cache-reuse matrices on
   both GPUs;
4. record the environment change as an experiment boundary;
5. only then replace the default policy.
