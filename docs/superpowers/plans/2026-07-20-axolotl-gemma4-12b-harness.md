# Axolotl Gemma 4 12B FA4 Harness Implementation Plan

> **Execution note:** Implement inline with test-driven development. Preserve the
> existing dirty worktree and do not create commits unless the user asks.

**Goal:** Add a short, reproducible Axolotl LoRA training harness that compares the
project FA4 backend with Axolotl's Gemma 4 hybrid default and SDPA on the real
`google/gemma-4-12B-it` model, checking output-loss/gradient agreement, route
coverage, and steady-state step time.

**Architecture:** Register a separately named 12B compatibility backend that maps
the 12B prepared-head geometry into the already validated 31B kernels. Local
attention zero-pads 16Q/8KV to 32Q/16KV. Global attention zero-pads Q and maps the
single KV head to `[K, K, 0, 0]` / `[V, V, 0, 0]`, which converts GQA-16 into two
identical GQA-8 groups; the first 16 output heads are returned. Autograd naturally
sums the duplicated global KV gradients. The Axolotl plugin selects a backend,
records CUDA-event step timings and a bounded gradient sketch, and writes a JSON
report. A standalone comparator rejects workload drift, loss/gradient drift,
missing FA4 layer routes, and fallback paths before reporting speedup.

**Pinned boundary:** Axolotl `2f5cb9da62a0fe763a1ddeb7798fc9acb2f4a417`,
Gemma 4 12B-it `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, project
Transformers `7ea2320c76117e6742364808a666ef6f2fb40a67`, and project
FlashAttention `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`. This is an
experimental integration boundary, not a refresh of the accepted H100 environment.

---

## Task 1: Freeze the 12B compatibility contract

**Files:**
- Create: `configs/model/gemma4-12b-harness.lock.json`
- Create: `tests/test_gemma4_12b_compat.py`
- Create: `src/gemma4_fa4/gemma4_12b_compat.py`

- [ ] Add tests for exact 12B local/global geometry, padding/duplication, sliced
      forward equivalence, and q/k/v gradient equivalence against the reference.
- [ ] Run the new tests and retain the expected pre-implementation failure.
- [ ] Implement fail-closed input validation, head adaptation, route accounting,
      and idempotent collision-safe Transformers registration.
- [ ] Run the focused compatibility tests.

## Task 2: Add report and comparison contracts

**Files:**
- Create: `tests/test_axolotl_harness.py`
- Create: `src/gemma4_fa4/axolotl_harness.py`
- Create: `scripts/compare_axolotl_runs.py`

- [ ] Test percentile calculation, workload matching, loss and gradient tolerances,
      route coverage, fallback rejection, and speedup computation.
- [ ] Run the tests and retain the expected pre-implementation failure.
- [ ] Implement the pure report builder/comparator and CLI with nonzero exit on any
      semantic failure.
- [ ] Run focused tests and CLI fixture checks.

## Task 3: Integrate with Axolotl without changing training state

**Files:**
- Create: `src/gemma4_fa4/axolotl_plugin.py`
- Create: `configs/axolotl/gemma4-12b-smoke.yaml`
- Create: `scripts/axolotl/make_smoke_dataset.py`
- Create: `scripts/axolotl/check_env.py`
- Create: `scripts/axolotl/run_h100_matrix.sh`
- Create: `tests/test_axolotl_assets.py`

- [ ] Test static YAML invariants, deterministic dataset generation, backend
      mutation, and environment rejection paths.
- [ ] Implement an Axolotl `BasePlugin` and Trainer callback. Use learning rate 0,
      no dropout, no packing, fixed seeds, eight steps, three excluded warmups, and
      a one-time bounded LoRA-gradient sketch.
- [ ] Add a fail-closed preflight and a matrix runner for hybrid, SDPA, and project
      FA4 in isolated output directories.
- [ ] Run focused tests and Ruff.

## Task 4: Document and execute available validation

**Files:**
- Create: `docs/axolotl-gemma4-12b-harness.md`
- Update: `experiments/EXP-0036-axolotl-gemma4-12b-harness.md`

- [ ] Document access prerequisites, exact setup/run commands, report schema,
      acceptance criteria, compatibility-overcompute caveat, and how to interpret a
      slower result.
- [ ] Run the complete CPU test suite and `ruff check .`.
- [ ] Sync to the configured H100, run preflight, then run the eight-step matrix only
      if pins, GPU, HBM, Axolotl, and gated-model access all pass.
- [ ] Record actual remote evidence or the exact blocking preflight condition. Do not
      make an acceptance or performance claim without the completed matrix.
