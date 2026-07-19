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

Examples:

```bash
REMOTE_INIT_COMMAND='source /etc/profile && module purge && module load cuda/13.3'
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
sudo ALLOW_SYSTEM_CHANGES=1 bash scripts/remote/install_cuda_ubuntu.sh
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

Bootstrap creates `.venv`, installs PyTorch from the reviewed official wheel,
checks out exact FlashAttention and Transformers SHAs under `.upstream/`,
installs FA4 with its exact CuTe-DSL pin, runs the model contract/oracle and
CPU/static checks, then performs a strict GPU environment check.

## 5. Run commands and collect artifacts

```bash
bash scripts/remote/check.sh h100
bash scripts/remote/run.sh h100 python scripts/verify_model_contract.py --transformers
bash scripts/remote/run.sh h100 pytest -q
bash scripts/remote/run.sh h100 python benchmarks/roofline.py \
  --out agent_space/h100-roofline.json
bash scripts/remote/collect.sh h100 agent_space/collected-h100
```

Use `b300` analogously. `run.sh` preserves argument boundaries and executes in
the remote checkout with architecture and JIT-cache variables set.

## 6. Measurement-session checklist

Before a performance run:

- confirm the allocation contains the intended GPU and no competing process;
- save `scripts/check_env.py --strict --json ...` output;
- record driver, toolkit, PyTorch runtime, CuTe DSL, source SHAs, clocks,
  power, and temperature;
- lock a sustainable clock only when lab policy permits it;
- use architecture/revision-specific JIT caches;
- run a fixed canary before and after the ladder;
- report cold compile, warm cache, hot L2, and cold L2 separately;
- unlock clocks when the session ends.

The remote helpers never change clocks, power limits, drivers, or scheduler
policy automatically.
