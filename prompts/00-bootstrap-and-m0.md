# Starter prompt — harden and establish M0

> Historical/reusable bootstrap template. The checked-in repository has
> already completed this stage; consult `docs/status.md` and do not reuse its
> occupied experiment IDs.

You are working in the Gemma 4 FA4 kernel lab. In this session, establish a
reproducible M0 baseline only. Do not implement or tune a kernel.

Read in order:

1. `AGENTS.md`
2. `docs/model-contract.md`
3. `docs/status.md`
4. `docs/environment.md`
5. `docs/remote-gpus.md`
6. `skills/gemma4-kernel-project/SKILL.md`
7. `skills/writing-cute-dsl-kernels/SKILL.md`
8. the CuTe skill references for versioning, correctness/profiling, and
   agent-assisted development.

Critical facts you must preserve:

- full shape is 32Q/16 local KV and 32Q/4 global KV;
- scale is exactly 1.0;
- global K and V share a projection source but are distinct FMHA operands;
- local mask is `(k > q - 1024) AND (k <= q OR same nonnegative vision block)`;
- full layers are causal even for vision;
- window 1024 maps to FA left window 1023; the overlay has no separate right cap for future same-block vision keys;
- `num_kv_shared_layers=0`; no cross-layer KV reuse is active in this checkpoint.

Tasks:

1. Use the untracked remote profiles; verify `upstream.lock.json` checkouts and the exact environment without floating or weakening the locks.
2. Run offline and Transformers-oracle contract checks.
3. Run CPU tests and the upstream fake-tensor compile pass.
4. On each GPU, capture strict environment JSON, locked-clock state, roofline,
   cold compile time, warm cache time, and current supported/unsupported
   attention cases.
5. Run fwd, bwd, and fwd+bwd separately. Never substitute full-causal SDPA for
   sliding attention or a shared-KV MLA formula for distinct prepared K/V.
6. Allocate unused experiment IDs and append baseline records for only the
   explicitly authorized target host; never overwrite EXP-0001 through
   EXP-0008.
7. Update `docs/status.md` with exact commands, pass/fail counts, unsupported
   cases, and artifact paths.

No speedup claims. If a shape cannot run, record the exact rejection and keep
the full model geometry visible.
