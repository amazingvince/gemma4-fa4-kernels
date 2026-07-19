# H100 M1 packed local metadata: production-length sparse design brief

## 1. Environment and version

- Date: 2026-07-19.
- `nvidia-cutlass-dsl`: 4.6.0.dev0.
- CUDA toolkit / driver: CUDA 12.8 / 580.126.09.
- Python / framework: Python 3.12.3 / PyTorch 2.8.0+cu128.
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, target `sm_90a`.
- Upstream: FlashAttention commit
  `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6` plus the recorded project patch.
- Interfaces: fixed `flash_attn_func`, `BlockSparseTensorsTorch`, compact sparse
  indices, separate forward/backward sparse tensors, and the accepted SM90
  local d256 mask-mod kernels.
- Starting exemplars: pinned `tests/cute/test_mask_mod.py`,
  `flash_attn/cute/block_sparsity.py`, `interface.py`, and the unchanged SM90
  forward/backward implementations.

## 2. Operation contract

For packed sequence `b`, query coordinate `q`, and key coordinate `k`, define
`q_abs = q + Sk_b - Sq_b`. Keep exactly:

```text
document[k] == document[q_abs]
AND k > q_abs - 1024
AND (k <= q_abs OR (
      vision[k] == vision[q_abs]
      AND vision[q_abs] >= 0
    ))
```

Inputs are distinct contiguous BF16 packed THD tensors Q `(Tq,32,256)` and
K/V `(Tk,16,256)`, CUDA INT32 cumulative arrays, and optional K-stream INT32
vision/document metadata. O is BF16 `(Tq,32,256)` and LSE is FP32 `(32,Tq)`.
Backward returns separate BF16 dQ, dK, and dV after the pinned FP32 reduction
paths. Scale is exactly 1.0. Empty segments and aliased K/V remain rejected.

## 3. Shape and layout regime

- Dynamic: `B>=1`, packed totals, and every nonempty
  `1 <= Sq_b <= Sk_b <= 262144` whose exact schedule fits the declared
  metadata, free-HBM, and padded-score safety envelope.
- Static: BF16, 32Q/16KV, GQA-2, d256, W1024, scale 1.0, H100 SM90.
- Public layout: packed THD plus packed K-stream metadata.
- Internal layout: one fixed BSHD `B=1` view per packed sequence. Slices are
  disjoint, so concatenated outputs preserve autograd ownership.
- Forward sparse grid: Q blocks 128, K blocks 80.
- Backward sparse grid: Q blocks 64, K blocks 64, represented in transposed
  K-block rows. It is generated at backward granularity rather than by
  transposing the incompatible K80 forward lists.
- Exact runtime lengths, metadata values, sparse counts, indices, and pointers
  are excluded from compile keys. Backward still has the pinned bounded
  single-Q-block/single-K-block selector buckets. Callable hash, sparse
  broadcast pattern, tile sizes, dtype, geometry, architecture, and other
  constexpr choices remain in the key.

## 4. Target and kernel family

- Target: H100 SM90a only.
- Family: correctness-first per-sequence composition over pinned fixed
  `flash_attn_func` block-sparse forward/autograd backward.
- Native text remains on the accepted packed native path.
- Metadata calls through S1025 remain on the accepted EXP-0008 dense custom
  path. Only metadata calls above S1025 enter this experiment.
- No upstream kernel, MMA, TMA, stage, barrier, or epilogue change is planned.
- Fallback is an explicit unsupported-path or preflight error; never a causal
  approximation, dense silent substitute, shared-KV/MLA route, or CPU result.

## 5. Candidate schedule and exactness

For each Q block, include a K block if and only if at least one in-range token
pair in that tile satisfies the complete predicate. Build the causal/window
part by grouping query coordinates by document and intersecting that
document's indexed K positions with the exact union of `[q_abs-1023,q_abs]`
intervals. Add a future K block only when it contains a token above the
minimum query threshold for the same `(document_id, nonnegative vision_id)`.
Sort and deduplicate every row.

Generate this Q-to-K adjacency twice: once at Q128/K80 for forward and once at
Q64/K64 for backward. Transpose only the independently generated Q64/K64
adjacency into backward K rows. This gives exact physical-tile incidence in
both directions; it neither omits a true pair nor schedules a wholly masked
tile.

All candidates are supplied as masked/partial blocks. Full-block counts remain
zero, with one head-broadcast, row-aligned zero count/index sentinel per
direction because the
pinned SM90 loader traces both sides of its dynamic empty-list branch and
cannot compile a `None` full-index tensor. The exact callable evaluates
document, strict left-window, causal, and same-vision conditions for every
scheduled element. The builder emits neither wholly masked candidate tiles nor
false negatives.

The index tensors use compact rank-4 storage with broadcast batch/head
dimensions. Before host enumeration, the adapter checks the rectangular INT32
upper bound
`4 * [Mf*(3+Nf) + Nb*(3+Mb)]`, where `Mf=ceil(Sq/128)`,
`Nf=ceil(Sk/80)`, `Mb=ceil(Sq/64)`, and `Nb=ceil(Sk/64)`. The locked maximum
square case is 94,027,776 bytes. During enumeration, a cumulative `2^40`
padded score-slot ceiling counts every candidate times its physical tile area
and 32 Q heads, preventing a semantically dense vision span from exhausting
host memory before the final allocation check. The final compact CUDA storage
must also remain below 2 GiB and 10% of free HBM. Rejection is explicit; the
adapter never drops candidate tiles or weakens the predicate.

## 6. Ownership and composition

For each packed sequence:

1. Split Q, K, V, vision IDs, and document IDs once into noncopying segment
   views. One `SplitWithSizesBackward` per differentiable packed input
   reassembles its gradients without per-segment full-base scatter buffers.
2. Build exact forward and backward sparse metadata from host-visible IDs,
   stopping during construction if the declared work budget would be crossed.
3. Transfer compact INT32 counts/indices to the input CUDA device.
4. Call fixed `flash_attn_func` with `causal=False`, no native window,
   `mask_mod` owning the full predicate, both sparse tensor sets,
   `softmax_scale=1.0`, `num_splits=1`, `pack_gqa=False`, and `return_lse=True`.
5. Concatenate O along packed Q and LSE along its packed sequence dimension.

Each K/V segment participates in exactly one call, so autograd returns disjoint
segment gradients that concatenate into separate packed dK and dV. dQ follows
the corresponding Q segment. No gradient merge changes the attention contract.

## 7. Movement, synchronization, and resources

The pinned fixed sparse kernels retain their existing TMA/WGMMA pipelines and
barriers. This experiment changes scheduling metadata and launch composition,
not the device synchronization protocol. Sparse tensors are read-only INT32
GMEM. Candidate counts, compact widths, rectangular worst-case storage, exact
compact storage, and padded scheduled work receive explicit checks before GPU
tensor creation.

Generated-code evidence must record forward/backward PTX/cubin/SASS hashes,
registers, stack/local memory, static/dynamic shared memory, and cache objects.
New mask-callable or sparse-broadcast variants must be bounded and explained.

## 8. Correctness and debug plan

- Exhaustive small CPU schedule coverage against the exact token predicate.
- Reject duplicate/out-of-range indices and prove sorted deterministic rows.
- Cover lower-right Q<K, partial Q/K tails, document mismatch, repeated IDs,
  ID zero, text ID -1, far-future same-vision inclusion, and far-past
  same-vision exclusion.
- Compare O, FP32 LSE, and separate dQ/dK/dV with dense references at tractable
  lengths above 1025.
- Prove hostile packed-sequence and document isolation.
- Run a Q1/K262144 far-offset strict-window/self-ownership sentinel.
- Run a Q2049/K262144 far-future vision plus different-document isolation
  sentinel.
- Exercise O-only, true `dout=None` LSE-only, and combined gradients.
- Run repeats and a nondefault stream; characterize rather than hide any
  nondeterministic reduction.
- Run memcheck, synccheck, and racecheck for both multi-tile directions.
- Vary runtime lengths, metadata contents, segment order, and compact widths in
  cache probes.

## 9. Measurement and risks

EXP-0010 authorizes no timing or speed claim. Semantically valid schedules
above a declared safety ceiling are rejected explicitly and remain outside
this acceptance; they are never approximated. Primary risks are a sparse false
negative, wrong lower-right metadata coordinate, forward/backward granularity
mismatch, padded-tail metadata reads, unbounded rectangular compact width,
per-sequence launch overhead, and new nondeterministic dQ order. Rollback keeps
the existing S1025 metadata guard and accepted native-text path.

## 10. Outcome

EXP-0010 passed at implementation revision
`12cfe711ad29139c7c78dcb355645ee5b9a70bb0`. Exact CPU tile-incidence and
public preflight tests, tractable H100 O/LSE/dQ/dK/dV references, hostile
packed/document isolation, both K262144 sentinels, repeat/nondefault-stream
checks, bounded cache reuse, memcheck/synccheck/racecheck, and retained
PTX/cubin/SASS inspection all passed. The aggregate real H100 suite reported
`196 passed, 8 skipped, 1 xfailed`; the dedicated sparse forward/backward fake
compile also passed.

The generated forward and backward retain HGMMA/TMA paths. Main backward has
zero stack/local traffic. Custom sparse forward reports `LOCAL=0` but uses a
144-byte stack with explicit LDL/STL traffic, so it is not described as
spill-free. No benchmark or speed claim was made.
