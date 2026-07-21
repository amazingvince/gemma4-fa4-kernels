# EXP-0040: H100 global owner-computes dK/dV

- Date / author: 2026-07-20 / Codex
- Kernel family: global-d512-backward
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Design brief: `docs/h100-global-owner-dkv-design.md`

## Invariant changed

Each `(batch, kv_head, N32)` tile has one CTA that accumulates all eight GQA
Q-head contributions in FP32 registers and directly writes final BF16 dK/dV.
The accepted forward and dQ paths are unchanged.

## Hypothesis

An initially opt-in owner dKV kernel eliminates all whole-sequence FP32 dK/dV buffers and
their postprocess launches while preserving the fixed/packed exact-BF16
reference policy, sanitizer cleanliness, bounded cache classes, and finite
S8K/S64K execution.

## Single change

Replace cross-CTA FP32 dK/dV bulk reduction with one K-major owner CTA that
visits the eight Q heads in fixed order and directly stores BF16 dK/dV.

## Gates

- [x] fake compile and generated resource inspection
- [x] fixed and packed reference matrices
- [x] five-repeat dK/dV identity and gradient-source isolation
- [x] GQA and mixed-empty ownership
- [x] fixed and packed memcheck/synccheck/racecheck
- [x] no FP32 dK/dV allocation or dK/dV postprocess launch
- [x] bounded fixed/packed compile-cache inventory
- [x] measured peak memory below the candidate estimate
- [x] S8K and S64K timing characterization
- [x] unchanged EXP-0038 rollback and EXP-0039 deterministic route
- [x] strict environment, full local/H100 suites, and pinned HF oracle

## Evidence

### Correctness and ownership

- Fixed S1/31/32/33/63/64/65/127/128/129 passed independent BF16/FP32
  references. dK/dV were bitwise identical across five repeats in every case.
- Structured V-slab superposition, isolated GQA Q-head ownership, nondefault
  streaming, and `lse`/`out_lse` gradient sources passed. LSE-only dV was
  exactly zero.
- Native packed tiny, mixed, mixed-empty, and fixed-versus-packed S65 parity
  passed. Empty-query segments produced exact-zero dK/dV and did not modify a
  neighboring segment.
- Default `deterministic=True` never selected owner dKV and retained bitwise
  dQ/dK/dV across five repeats. Setting
  `FLASH_ATTENTION_GEMMA4_EXPERIMENT_OWNER_DKV=0` passed the fixed S65
  reference and restored the accepted EXP-0038 accumulation route.

### Synchronization, resources, and launches

- Fixed S128 and packed mixed memcheck, synccheck, and racecheck were clean
  with `--report-api-errors no`: zero memory errors, synchronization errors,
  race hazards, and warnings. An earlier invocation without that repository
  compatibility option reported unsupported `cuGetProcAddress_v2` queries;
  the kernel run itself was not the source of that tool/API mismatch.
- Fresh cache
  `/workspace/.cache/fa4-sm90-exp0040/75b533422cb34e457716f0837a585efde3be76c734c40492a82e898763eab471`
  contained exactly four main objects: owner dKV and unchanged dQ for the
  fixed and packed ABIs. Runtime lengths added no object class.
- Fixed and packed owner dKV generated with 168 registers and zero stack/local
  memory. Nsight Systems attributed two backward main kernels and two dQ
  postprocess kernels, with no dK or dV postprocess launch.

### Memory

- The allocation contract removes exactly `16384 * padded_K` bytes. At
  Q1/K262144 that is 4,294,967,296 bytes of eliminated FP32 dK/dV padding;
  measured owner peak was 4,831,904,256 bytes versus the former conservative
  8,615,887,104-byte estimate.
- Square S32768 measured 7,948,248,064 bytes versus the former
  9,710,370,816-byte estimate. Fixed S33 measured 10,053,120 bytes versus the
  former 12,312,832-byte estimate.
- Default fixed S128 and packed mixed runs stayed below the updated project
  preflight estimates. Whole-sequence FP32 dQ remains and is the next
  bounded-memory target.

### Performance

The locked hot-L2 BF16 backward rows are in
`agent_space/remote-h100-exp0040/exp0040-benchmarks.jsonl`.

| Case | Owner median / IQR | Rollback median / IQR | Delta |
|---|---:|---:|---:|
| global S8K | 41.8628 / 0.0638 ms | 42.9963 / 0.0627 ms | **-2.64%** |
| global S64K | 2569.6875 / 0.2068 ms | 2579.5447 / 0.8586 ms | **-0.38%** |

Both comparisons use the same H100, scale 1.0, exact global-causal semantics,
PyTorch 2.8.0+cu128, and non-overlapping IQRs. These are scoped comparisons to
the accepted rollback path, not general performance claims.

### Final gates

- Strict H100 environment with profilers and pinned Transformers: PASS;
  cumulative FA4 patch SHA256
  `138eddb2d4bf7d08f91947700daa47be12b53871e5a1174ad69a267c04ceb2b1`,
  both upstream diffs exact, no warnings or errors.
- Local: compileall PASS, Ruff PASS, model contract PASS, pytest
  `438 passed, 106 skipped, 8 warnings`.
- Fresh-cache H100 pytest: `531 passed, 17 skipped, 1 xfailed, 116 warnings`.
- Pinned Transformers contract: PASS; optional oracle pytest
  `5 passed, 1 xfailed, 1 warning`.

## Decision

**ACCEPT.** Promote owner-computed dK/dV to the fast default. Retain the
environment rollback and the independent deterministic route. EXP-0040 is the
first bounded-memory slice only; owner-computed or otherwise bounded dQ is the
next production experiment.
