# Remote H100 and B300 connection placeholders

The repository contains transport and bootstrap templates, never credentials.
The actual `remote/*.env` profiles are ignored by git. Treat profile contents
as trusted local shell configuration: `REMOTE_INIT_COMMAND`,
`REMOTE_LAUNCHER`, and `REMOTE_EXTRA_SSH_ARGS` intentionally permit
site-specific shell syntax.

## 1. Create untracked profiles

```bash
cp remote/h100.env.example remote/h100.env
cp remote/b300.env.example remote/b300.env
```

Fill the following fields:

- `REMOTE_HOST`, `REMOTE_USER`, `REMOTE_PORT`;
- optional `REMOTE_IDENTITY_FILE` and `REMOTE_PROXY_JUMP`;
- `REMOTE_ROOT`, using a simple absolute path or `~/relative/path` with no
  spaces;
- `REMOTE_INIT_COMMAND` for modules, Conda, proxy setup, or site environment;
- optional `REMOTE_LAUNCHER` for a scheduler allocation. The helper appends
  `bash -lc <payload>` to it;
- an architecture-local FA4 JIT-cache directory, preferably node-local NVMe.
- a shared `REMOTE_GPU_LOCK_FILE` for every task targeting the same physical
  GPU, plus the desired lease wait and idle-process policy.

Examples:

```bash
REMOTE_INIT_COMMAND='source /etc/profile && module purge && module load cuda/12.8'  # H100
REMOTE_LAUNCHER='srun --partition=blackwell --gres=gpu:b300:1 --time=01:00:00'
```

For a direct SSH GPU host, leave `REMOTE_LAUNCHER` empty.

Do not store passwords, tokens, private keys, cloud credentials, or Hugging
Face access tokens in the repository.

## 2. Probe before changing the host

```bash
bash scripts/remote/probe.sh h100
bash scripts/remote/probe.sh b300
```

This checks SSH/scheduler access and reports the visible GPU, driver, compute
capability, and `nvcc` location. It does not alter the machine.

## 3. Administrator-controlled host prerequisites

The default reviewed stack is documented in `docs/environment.md`. Driver
changes are intentionally not automated: on a shared datacenter host they can
require a reboot and disrupt every user.

For an Ubuntu 24.04 host whose owner has approved system changes:

```bash
sudo ALLOW_SYSTEM_CHANGES=1 bash scripts/remote/install_host_prereqs_ubuntu.sh
sudo ALLOW_SYSTEM_CHANGES=1 bash scripts/remote/install_cuda_ubuntu.sh h100
# Use `b300` only on the B300 host.
```

The CUDA helper installs the **toolkit only**. For Rocky/RHEL/DGX OS or a
module-managed cluster, follow the matching NVIDIA/site procedure instead of
reusing the Ubuntu repository URL.

## 4. Sync and bootstrap user space

```bash
bash scripts/remote/sync.sh h100
bash scripts/remote/bootstrap.sh h100
bash scripts/remote/sync.sh b300
bash scripts/remote/bootstrap.sh b300
```

Bootstrap passes the profile name to the target-specific environment policy,
creates `.venv-h100` or `.venv-b300`, installs PyTorch from the reviewed official wheel,
checks out exact FlashAttention and Transformers SHAs under `.upstream/`,
applies the exact hash-locked H100 patch when that profile is selected,
installs FA4 with `[dev]` on H100 or `[dev,cu13]` on B300, runs the model
contract/oracle and CPU/static checks, then performs a strict GPU environment
check. Strict H100 validation accepts only the recorded patch diff and rejects
any additional upstream modification.

## 5. Run commands and collect artifacts

```bash
bash scripts/remote/check.sh h100
bash scripts/remote/run.sh h100 python scripts/verify_model_contract.py --transformers
bash scripts/remote/gpu-run.sh h100 pytest -q
bash scripts/remote/gpu-run.sh h100 python benchmarks/roofline.py \
  --out agent_space/h100-roofline.json
bash scripts/remote/collect.sh h100 agent_space/collected-h100
```

Use `b300` analogously. `run.sh` preserves argument boundaries for commands
that do not need exclusive GPU ownership. `gpu-run.sh` additionally takes an
exclusive `flock` lease, records an owner sidecar while the command runs, and
by default rejects an already-present compute process after acquiring the
lease. Exit 75 means another lease holder won; exit 76 means an uncooperative
process is already using the GPU. Set `REMOTE_GPU_REQUIRE_IDLE=0` only on a
scheduler allocation whose isolation is guaranteed externally.

For a fresh isolated checkout without a virtual environment, acquire the same
lease around bootstrap with:

```bash
bash scripts/remote/gpu-run.sh h100 --no-venv bash scripts/setup_env.sh h100
```

## 6. Measurement-session checklist

Before a performance run:

- use `scripts/remote/gpu-run.sh` so concurrent tasks cannot share the device;
- confirm the allocation contains the intended GPU and no competing process;
- save `scripts/check_env.py --strict --json ...` output;
- record driver, toolkit, PyTorch runtime, CuTe DSL, source SHAs, clocks,
  power, and temperature;
- lock a sustainable clock only when lab policy permits it;
- use architecture/revision-specific JIT caches;
- rerun `scripts/check_env.py --strict --require-profilers` before benchmarks;
- run a fixed canary before and after the ladder;
- report cold compile, warm cache, hot L2, and cold L2 separately;
- unlock clocks when the session ends.

The remote helpers never change clocks, power limits, drivers, or scheduler
policy automatically.
