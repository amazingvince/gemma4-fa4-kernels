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

- [ ] exhaustive CPU schedule coverage
- [ ] adapter admission/rejection and composition tests
- [ ] exact forward O/FP32-LSE reference matrix above S1025
- [ ] O-only, LSE-only, and combined separate dQ/dK/dV reference matrix
- [ ] lower-right, strict-window, future-vision, and document sentinels
- [ ] hostile packed-sequence isolation
- [ ] Q1/K262144 far-offset strict-window/self-ownership sentinel
- [ ] Q2049/K262144 far-future vision/document-isolation sentinel
- [ ] repeats and nondefault stream
- [ ] schedule allocation/work preflight

## Synchronization and generated code

- [ ] memcheck
- [ ] synccheck
- [ ] racecheck
- [ ] bounded cache variants across lengths, metadata, and compact widths
- [ ] PTX/cubin/SASS and resource evidence

## Iterations

1. Initial fixed sparse compile with `full_* = None` failed before launch in
   `block_sparse_utils.sparse_physical_n_block_forward`: the SM90 loader traced
   its dynamic zero-mask branch and indexed the compile-time `None` full list.
   Retain an explicit head-broadcast, row-aligned INT32 zero count/index
   sentinel in each
   direction and rerun the same discriminating forward case. No kernel source
   or synchronization protocol changed.

## Measurement

No performance measurement or speed claim is authorized for EXP-0010.

## Decision

**OPEN.** Record ACCEPT, REJECT, or REFINE only after the declared H100 gates.

## Record

Append through `scripts/record_result.py` only after an implementation commit
and strict H100 evidence exist.
