# EXP-0010: H100 packed local metadata at production lengths

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256 block-sparse integration
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar: pinned fixed `flash_attn_func`, compact block sparsity, and separate
  forward/backward sparse metadata
- Design brief: `docs/h100-m1-local-long-metadata-design.md`

## Invariant changed

Metadata-bearing packed local calls above S1025 change from explicit rejection
to an exact per-sequence fixed block-sparse composition. Native packed text and
the accepted S<=1025 dense custom path do not change.

## Hypothesis

A tile-exact Q128/K80 forward schedule and independently generated
Q64/K64 transposed backward schedule, with every candidate left partial for
the exact Gemma mask callable, preserve O/LSE/dQ/dK/dV for packed
vision/document metadata through the locked K262144 maximum when the request
fits the predeclared `2^40` padded-score, 2 GiB metadata, and 10%-free-HBM
safety envelope, without an upstream FA4 patch.

The pinned SM90 loader traces both sides of its dynamic empty-list branch, so
each direction carries an explicit head-broadcast, row-aligned zero
count/index sentinel for the
otherwise absent full-block list. This is compile plumbing only; every executed
candidate remains in the partial list.

Falsification is any missing true tile, semantic mismatch, cross-sequence or
cross-document gradient leakage, unbounded compile variants, sanitizer
finding, or rejection inside the declared resource envelope.

## Single change

Admit metadata-bearing nonempty packed local attention above S1025 through the
declared sparse per-sequence composition. Do not change the native path, dense
S<=1025 path, kernel tile/stage/barrier protocol, model geometry, scale, or
prepared K/V contract.

## Correctness evidence

- [x] exhaustive CPU schedule coverage
- [x] adapter admission/rejection and composition tests
- [x] exact forward O/FP32-LSE reference matrix above S1025
- [x] O-only, LSE-only, and combined separate dQ/dK/dV reference matrix
- [x] lower-right, strict-window, future-vision, and document sentinels
- [x] hostile packed-sequence isolation
- [x] Q1/K262144 far-offset strict-window/self-ownership sentinel
- [x] Q2049/K262144 far-future vision/document-isolation sentinel
- [x] repeats and nondefault stream
- [x] schedule allocation/work preflight

The local CPU suite passed 128 tests with 75 target/oracle skips. The 14
schedule tests include exhaustive small metadata, independently generated
forward/backward incidence, document filtering, strict W1024 behavior,
K262144 sentinels, the pinned 94,027,776-byte maximum rectangular bound, and
incremental work-budget exhaustion. Public adapter tests also prove the 2 GiB,
10%-free-HBM, and translated padded-work rejection paths.

On H100, the tractable Q=`[33,65]`, K=`[2049,4097]` adjacent-vision/split-doc
forward matched the dense O/LSE reference. Combined `out_lse` backward passed
the unchanged upstream-relative BF16 policy: candidate maxima were dQ `0.5`,
dK `0.3125`, and dV `0.03125`. O-only passed with maxima dQ `0.5`, dK
`0.31542969`, dV `0.03125`; true LSE-only passed with dQ/dK `0.0625` and
exact-zero dV. Three nondefault-stream repeats kept O/LSE/dK/dV bitwise equal;
dQ had maximum pairwise drift `0.015625`, with every repeat independently
inside the frozen numerical policy.

The hostile long-metadata packed-boundary probe passed with repeated vision
and document IDs across sequences. Mutating sequence 1 left sequence 0
O/LSE/dK/dV exact; independent reference dQ/dK/dV remained exact. The
Q1/K262144 sparse sentinel passed the strict far-offset W1024/self-ownership
check. The Q2049/K262144 sentinel included a far-future same-document vision
key, excluded the same vision ID in another document and a far-past key,
returned `log(1025)` LSE, and assigned dV only to the allowed KV head/key.

The complete real H100 suite passed `196 passed, 8 skipped, 1 xfailed` in
8.97 seconds. The skips are fake-only tests; the expected failure remains the
pinned Transformers generic 2D FA4 mask adapter. The dedicated explicit
zero-full-list sentinel fake-compiled forward and backward: `1 passed` with
only PyTorch's FakeTensor `data_ptr()` deprecation warning.

## Synchronization and generated code

- [x] memcheck
- [x] synccheck
- [x] racecheck
- [x] bounded cache variants across lengths, metadata, and compact widths
- [x] PTX/cubin/SASS and resource evidence

At Q=`[64,65]`, K=`[2048,2049]`, adjacent vision, split documents, and
combined gradients, memcheck and synccheck each reported `0 errors`;
racecheck reported `0 hazards displayed (0 errors, 0 warnings)`. API-probe
reporting was disabled consistently with the earlier H100 experiments.

The isolated long custom cache probe varied lengths, segment order, tensor and
metadata contents, and compact widths without adding or removing objects. It
retained one forward key
`43a8e5b603bc98d7737d8467421a9ff42b7d318bd614b0288c18f31aa3afdbd7`,
two bounded main-backward selector keys
`5b2ac6ccf74d9c5e3d06270b6f2f673f9080b195b63f5254a08844f18cdd4626`
and `6d911ac30fff981f25f2b7768b2eba7e5e321eef6e60e855fd0e5497db7e11ab`,
plus one preprocess and one postprocess object. The toolchain fingerprint was
`6167b82729b6e91949486c7a469474318b0906e396849651633c18d6f82aef92`.

Fresh retained code for Q65/K4097 targets PTX 8.8 / `sm_90a`. Canonical
forward PTX/cubin/SASS SHA256 values are respectively
`92778a7944c242608cd2edbaecde406221872dbb75b02297567bcbccea405c12`,
`545c16e106864dc7dc20b375d77ec21f3bd6565a871d34c90e9166570d1b89bf`,
and `47d62c80928f87053e3520d16198604445d45ec7696c2ce3e9c75278241f8336`.
Main-backward values are
`41d9466090a112506f9f37d9b22129c8f3a2e5498e4efc0dc5f54e2e39ec067d`,
`d9bcf3ac1193fb053e67bc45cf00118acd5d3d985afd1201f003f0bddebfe399`,
and `8e658b03be14ba798fdd691a23d7ff98f547bc48bdb61f338977ae059842780f`.

Forward SASS contains 100 HGMMA, 56 TMA-load, 4 TMA-store, and 18 warpgroups
arrive/dependency-barrier instructions. It uses 168 registers, 1 KiB static
shared memory, `LOCAL=0`, and a 144-byte stack with 51 LDL/38 STL
instructions; it is not claimed spill/stack-traffic-free. Main backward
contains 88 HGMMA, 32 TMA-load, and 20 warpgroup arrive/dependency-barrier
instructions, with 168 registers, 1 KiB static shared, and zero stack/local or
LDL/STL traffic. Preprocess uses 78 registers; postprocess uses 60 registers
and 1 KiB static shared; both have zero stack/local traffic. Artifacts remain
under `/workspace/agent_space/exp0010-codegen-20260719-7a41d2`.

## Iterations

1. Initial fixed sparse compile with `full_* = None` failed before launch in
   `block_sparse_utils.sparse_physical_n_block_forward`: the SM90 loader traced
   its dynamic zero-mask branch and indexed the compile-time `None` full list.
   Retain an explicit head-broadcast, row-aligned INT32 zero count/index
   sentinel in each
   direction and rerun the same discriminating forward case. No kernel source
   or synchronization protocol changed.
2. The explicit sentinels fake-compiled both directions and the real reference
   matrix passed. Exact physical-tile tests then replaced coverage-only checks
   so wholly masked document tiles could not hide as conservative candidates.
3. Independent tensor slices were replaced by one `torch.split` per packed
   differentiable input, preventing per-segment full-base scatter gradients.
   The hostile isolation and maximum-context ownership tests then passed.
4. The rectangular metadata, exact compact metadata, free-HBM, and cumulative
   padded-work guards were frozen before acceptance. Over-budget valid
   schedules remain explicit unsupported paths rather than approximations.

## Measurement

No performance measurement or speed claim is authorized for EXP-0010.

## Decision

**ACCEPT** at implementation revision
`12cfe711ad29139c7c78dcb355645ee5b9a70bb0` for nonempty packed H100 local
metadata attention within the declared resource envelope. This adds no
performance, deterministic-dQ, empty-segment, generic-dispatch, B300, or
over-budget-schedule claim.

## Record

`scripts/record_result.py` appended the accepted H100 record against
`12cfe711ad29139c7c78dcb355645ee5b9a70bb0` after the strict environment,
model-oracle, real/fake suite, numerical, sanitizer, cache, and generated-code
gates passed.
