# GitHub Publication Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:test-driven-development` for the oracle guard,
> `superpowers:verification-before-completion` before the completion claim, and
> the repository's `gemma4-kernel-project` router for correctness constraints.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a clean, verified initial GitHub repository without changing
Gemma 4 attention or kernel semantics.

**Architecture:** Preserve the starter bundle and its provenance files. Make
only evidence-backed hygiene corrections: make the optional Transformers
oracle skip incompatible installations, replace the abbreviated Apache notice
with the canonical Apache-2.0 license text, and enforce portable LF line
endings. Refresh the bundle checksum manifest after those changes, then
initialize and publish Git history.

**Tech Stack:** Python 3.12 target, pytest, Ruff, Bash static checks, Git, and
GitHub CLI.

## Global Constraints

- Prepared global K and V remain distinct operands.
- Softmax scale remains exactly `1.0`.
- Do not claim GPU compilation, sanitizer, correctness, or performance evidence.
- Keep `remote/*.env`, `.upstream/`, virtual environments, and profiler outputs untracked.
- Create `amazingvince/gemma4-fa4-kernels` as a private repository unless the user requests otherwise.

---

### Task 1: Guard the optional Transformers oracle

**Files:**
- Create: `tests/test_hf_oracle_guard.py`
- Modify: `tests/test_hf_oracle_optional.py`

**Interfaces:**
- Consumes: pytest module-level skip behavior and the optional oracle test module.
- Produces: clean collection when `transformers` exists but its pinned Gemma 4 APIs do not.

- [x] **Step 1: Add a subprocess regression test**

```python
def test_optional_oracle_skips_incompatible_transformers(tmp_path):
    package = tmp_path / "transformers"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "0.0"\n')
    # Run only the optional oracle with the incompatible package first on PYTHONPATH.
    # Assert exit code 0 and one skipped module.
```

- [x] **Step 2: Run the regression test and verify red**

Run: `python -m pytest -q -p no:cacheprovider tests/test_hf_oracle_guard.py`

Expected: FAIL because `tests/test_hf_oracle_optional.py` raises
`ModuleNotFoundError` while importing `transformers.masking_utils`.

- [x] **Step 3: Gate each required pinned module with `pytest.importorskip`**

```python
masking_utils = pytest.importorskip("transformers.masking_utils")
configuration_gemma4 = pytest.importorskip(
    "transformers.models.gemma4.configuration_gemma4"
)
modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
```

Bind the required symbols from these imported modules after the gates pass.

- [x] **Step 4: Verify green and run the full CPU suite**

Run: `python -m pytest -q -p no:cacheprovider`

Expected on an incompatible or absent Transformers install: 29 passed and 1
skipped. The pinned optional oracle remains runnable when its checkout is installed.

### Task 2: Normalize repository licensing

**Files:**
- Modify: `LICENSE`

**Interfaces:**
- Consumes: the existing `Apache-2.0` declaration in `pyproject.toml`.
- Produces: the canonical Apache License 2.0 terms from apache.org.

- [x] **Step 1: Replace the abbreviated notice**

Replace `LICENSE` with the canonical text at
`https://www.apache.org/licenses/LICENSE-2.0.txt`. Do not alter the declared
license identifier or invent a different copyright owner.

- [x] **Step 2: Compare the result with the authoritative source**

Fetch the source text read-only and assert that normalized line endings match.

### Task 3: Refresh provenance and verify the tree

**Files:**
- Create: `.gitattributes`
- Modify: `FILE_SHA256SUMS.txt`
- Modify: `docs/model-contract.md` (formatting only)
- Track: `docs/superpowers/plans/2026-07-18-github-publication.md`
- Track: `tests/test_hf_oracle_guard.py`

**Interfaces:**
- Consumes: the final publishable file set.
- Produces: a checksum manifest and clean verification evidence for the initial commit.

- [x] **Step 1: Update checksums for every changed or added tracked file**

Run `sha256sum` for `LICENSE`, both oracle test files, and this plan, then update
their sorted entries in `FILE_SHA256SUMS.txt`.

- [x] **Step 2: Run the offline verification matrix**

Run the model-contract verifier, full pytest suite, current Ruff check and
format check, shell syntax checks, JSON parsing, CuTe skill validation, and
both checksum manifests.

- [x] **Step 3: Review the exact initial Git file set**

Confirm ignored credentials and generated artifacts are absent, all files are
smaller than GitHub's limits, and no kernel or model-contract files changed.

### Task 4: Initialize and publish GitHub history

**Files:**
- Create: `.git/` metadata only; no source changes.

**Interfaces:**
- Consumes: the verified publishable tree and authenticated GitHub account `amazingvince`.
- Produces: private GitHub repository `amazingvince/gemma4-fa4-kernels` with `main` as default branch.

- [x] **Step 1: Initialize Git and inspect the staged snapshot**

Run: `git init -b main`, `git add .`, and `git diff --cached --check`.

- [x] **Step 2: Commit the verified snapshot**

Run: `git commit -m "chore: publish Gemma 4 FA4 kernel lab"`.

- [x] **Step 3: Create and push the private GitHub repository**

Run: `gh repo create amazingvince/gemma4-fa4-kernels --private --source . --remote origin --push`.

- [ ] **Step 4: Verify the remote and initial GitHub Actions run**

Confirm the repository URL, remote default branch, clean local status, and the
result of the `cpu-contract` workflow. If CI exposes an environment-only issue,
diagnose it without weakening the model contract.
