# H100 M1 packed-varlen local design brief

This brief governs EXP-0008. It extends the accepted local d256 paths to
packed variable-length batches. It does not add a block-sparse schedule,
framework dispatch, or a performance claim.

Historical-scope note: EXP-0015 later supersedes only this brief's
empty-segment exclusion. The current mixed packed contract permits
`0 <= Sq <= Sk <= 262144` with positive aggregate totals/exact maxima; the
remaining EXP-0008 evidence and limits stay historical and unchanged.

## 1. Environment and version

- Date: 2026-07-19.
- Target: NVIDIA H100 80GB HBM3, SM90a, CUDA 12.8, driver 580.126.09.
- PyTorch / CuTe DSL: `2.8.0+cu128` / `4.6.0.dev0`.
- FlashAttention: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6` plus the
  hash-locked H100 project patch.
- Production exemplars: pinned `FlashAttnVarlenFunc`, `_flash_attn_fwd`,
  `_flash_attn_bwd`, `test_mask_mod_varlen.py`, and the global packed
  auxiliary callables in `mask_mod_definitions.py`.
- Upstream evidence boundary: packed custom-mask forward is tested upstream;
  public autograd wires the same callable and auxiliaries into backward, but
  upstream's packed custom-mask test module is explicitly forward-only.

## 2. Operation contract

Packed inputs and outputs use:

```text
Q:   (Tq, 32, 256) BF16
K/V: (Tk, 16, 256) BF16, distinct storage
O:   (Tq, 32, 256) BF16
LSE: (32, Tq) FP32
cu_seqlens_q/cu_seqlens_k: (B + 1) CUDA INT32
```

For sequence-local query `q`, key `k`, `Sq <= Sk`, and
`q_abs = q + Sk - Sq`, the complete custom keep predicate is:

```text
same document
AND k > q_abs - 1024
AND (
  k <= q_abs
  OR same nonnegative vision block
)
```

Scale is exactly 1.0; dropout and softcap are zero. Q/K score, softmax/LSE,
and backward accumulation retain the pinned FP32 stage contracts. dQ, dK,
and dV are separate BF16 outputs. K/V may not alias.

## 3. Shape, metadata, and validation regime

- `B >= 1`; every segment is nonempty; empty segments are rejected.
- Per-sequence `1 <= Sq <= Sk <= 1025` for this correctness experiment.
- Cumulative arrays start at zero, end at Tq/Tk, are strictly increasing,
  have the same B, and their exact maxima match the public max arguments.
- Packed vision and document metadata follow the K stream with shape `(Tk,)`.
  Query metadata is read at its lower-right K position `q_abs`; this prevents
  inconsistent Q/K IDs and intentionally excludes generic cross-attention.
  Each public tensor shares Q's CUDA device and is normalized from INT32 or
  exactly representable INT64 to contiguous INT32.
- Omitting vision metadata means text IDs of `-1`. Omitting document metadata
  means one document per sequence.
- Runtime cumulative values, IDs, totals, and pointers do not enter a compile
  key. Callable hashes, auxiliary metadata, varlen presence, max-length class,
  and pinned one-block/multi-block selectors may create bounded variants.

## 4. Dispatch and kernel family

- Text-only packed calls preserve FA4's native lower-right causal/local path:
  `causal=True`, `window_size=(1023,0)`.
- Vision or document metadata selects one module-level hash-stable callable
  that owns the complete predicate. The adapter therefore passes
  `causal=False` and `window_size=(None,None)` in both forward and backward.
- The callable indexes packed K-stream INT32 auxiliaries using
  `seqlen_info.offset_k`, derives the query's K coordinate from `q_abs`, and
  clamps padded tile coordinates before every global load.
- The fixed B1 callable is not reused: it has no packed offsets or
  lower-right coordinate correction.
- `pack_gqa=False`, `num_splits=1`, and `deterministic=False` remain locked.

## 5. Tile, ownership, and pipeline

- SM90 d256 custom forward is expected to retain the dense M128 x N80 family.
- SM90 d256 backward is expected to retain M64 x N64 with Q/dO/PdS stages
  1/1/1 and transposed KV ownership.
- Q/K/V TMA, WGMMA, softmax, epilogue, dQ/dK/dV ownership, barriers, and
  drains remain pinned FA4 behavior.
- The only new data movement is clamped scalar INT32 auxiliary reads per
  score coordinate. Cumulative arrays are consumed by FA4's existing varlen
  sequence-info path.
- No fast-sampling or block-sparsity annotation is permitted: arbitrary
  vision/document boundaries are not proven by a five-point classifier.

## 6. Correctness matrix

CPU truth tables and packed reference tests precede GPU compilation. H100
equal-length batches cover:

```text
[1]
[31, 32, 33]
[63, 64, 65, 127, 128, 129]
[1023, 1024, 1025]
[129, 1, 65, 33]
[33, 65, 1, 129]
```

Lower-right text/custom cases cover Q lengths `[1,31,64,129]` against K
lengths `[33,64,128,1025]`. Packed offsets deliberately cross 64/128-byte
and tile boundaries.

Required evidence:

- O and FP32 LSE against an independent per-sequence FP32 reference;
- separate dQ/dK/dV under the frozen EXP-0004 upstream-relative rule;
- O-only, LSE-only, and combined gradients;
- repeated-ID isolation across cumulative boundaries;
- an internal document boundary that defeats an equal vision ID;
- extreme K/V mutation in another sequence/document cannot change this
  sequence's O, LSE, or gradients;
- lower-right future vision allowance and exact W1024 left edge;
- structured isolated dO ownership under GQA-2;
- three same-input repeats and a nondefault stream;
- fake compile, real full/partial tiles, memcheck, synccheck, racecheck, and
  retained PTX/cubin/SASS/resource observation.

## 7. Rejections and risks

Reject malformed cumulative arrays, mismatched totals/batch counts, wrong
maxima, Sq greater than Sk, empty segments, metadata length/device/dtype/range
errors, unsupported geometry, K/V aliasing, and any sequence above 1025.

Primary correctness risks are forgetting the lower-right `Sk-Sq` offset,
reading packed auxiliaries with local indices, or allowing native causal/local
flags to bypass the custom predicate in backward. Primary performance risk is
the dense custom schedule. Rollback is the accepted fixed path plus the native
packed text path; custom packed metadata remains rejected if any gate fails.

## 8. Deferred work

Empty segments, B300, production context through 262144, block-sparse tile
classification, generic Transformers dispatch, cross-attention semantics,
benchmarking, and performance tuning remain outside EXP-0008.
