# H100 M1 local multimodal design brief

This brief governs EXP-0007. It extends the accepted H100 local d256 text
path with Gemma 4's exact vision-block overlay. It does not change the
WGMMA/TMA pipeline or claim a performance-ready sparse schedule.

## 1. Environment and version

- Date: 2026-07-19.
- `nvidia-cutlass-dsl`: `4.6.0.dev0`.
- FlashAttention: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6` plus the
  hash-locked H100 project patch.
- CUDA / driver: 12.8 / 580.126.09.
- Python / framework: 3.12.3 / PyTorch `2.8.0+cu128`.
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, target `sm_90a`.
- Production exemplars: pinned `flash_attn/cute/mask.py`,
  `flash_fwd_sm90.py`, `flash_bwd_sm90.py`, `interface.py`, and
  `tests/cute/mask_mod_definitions.py`.
- Verified APIs: module-level `@cute.jit` mask callable with signature
  `(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors)`;
  `flash_attn_func(..., mask_mod=..., aux_tensors=..., return_lse=True)`;
  callable hashing and auxiliary-tensor metadata in the forward/backward
  compile keys.

## 2. Operation contract

For query position `q` and key position `k`, keep the score exactly when:

```text
k > q - 1024
AND
(k <= q OR (vision_id[q] == vision_id[k] AND vision_id[q] >= 0))
```

- Q/K/V: contiguous BSHD BF16, B=1, 32Q/16KV, GQA-2, Dqk=Dv=256.
- O: BSHD BF16; LSE: `(B,Hq,S)` FP32.
- Score, softmax, and gradient accumulation: FP32 at the pinned FA4 stage
  boundaries; final dQ/dK/dV are distinct BF16 tensors.
- Scale: exactly 1.0; dropout: zero.
- Empty sequence: rejected. A nonempty equal-length causal row always admits
  at least its diagonal element, so no all-masked row is expected.
- Q, K, and V may not alias. Vision IDs are read-only and have no gradient.
- The candidate uses `deterministic=False`; repeat behavior is measured rather
  than promised before H100 evidence exists.

## 3. Shape and layout regime

- Fixed-length rank-4 BSHD only for EXP-0007.
- B=1 and `1 <= S <= 1025`.
- Public vision IDs have shape `(B,S)`, share Q's CUDA device, and are
  normalized from INT32 or exactly representable INT64 input to a private
  contiguous INT32 `(S,)` auxiliary. The rank-1 B=1 layout keeps S=1 from
  presenting two ambiguous stride-1 dimensions to CuTe.
- Q/K/V retain the existing 16-byte base-alignment requirement.
- Runtime sequence length, vision-ID values, and auxiliary pointers do not
  enter the compile key.
- Static specialization additions are the mask callable hash,
  auxiliary-tensor presence, and auxiliary metadata. Expect a bounded,
  value-independent forward/backward variant set. In particular, the pinned
  backward key distinguishes single-block from multi-block shapes, so this
  brief does not promise exactly one backward binary.
- Packed varlen, unequal Sq/Sk, document IDs, padding, dropout, FP16, and B>1
  remain outside EXP-0007 and are the next compatibility experiment.

## 4. Target and kernel family

- Target: SM90a H100.
- Family: pinned FA4 SM90 attention with a custom element predicate.
- Text fallback: the accepted native causal/local path remains unchanged.
- General fallback: the repository PyTorch reference is correctness-only.
- Compute atoms, tiles, staging, producer/consumer roles, and epilogues remain
  selected by the pinned FA4 implementation.
- Because `mask_mod` encodes the whole predicate, the adapter passes
  `causal=False` and `window_size=(None,None)`. Passing native causal/window
  settings as well would make forward and backward resolver behavior differ.

## 5. Tile and ownership hierarchy

- Problem modes: B, Hq/Hkv, M=query, N=key, D=256.
- Expected forward tile: pinned dense/custom-mask d256 SM90 M128 x N80
  selection; record the realized tile at compile time.
- Expected backward tile: pinned d256 M64 x N64 configuration with Q/dO/PdS
  stages 1/1/1; record the realized variant.
- Grid, WGMMA ownership, TMA movement, FP32 accumulator ownership, and
  backward reduction ownership are unchanged.
- The mask callable reads one logical query and key block ID for each score
  coordinate and returns a keep predicate. No output writer ownership changes.

| Tensor | Logical modes | Shape/stride | Memspace | Owner | Consumer |
|---|---|---|---|---|---|
| Q | B,M,Hq,D | contiguous BSHD | GMEM -> SMEM | existing producer | QK WGMMA |
| K | B,N,Hkv,D | contiguous BSHD | GMEM -> SMEM | existing producer | QK WGMMA |
| V | B,N,Hkv,D | contiguous BSHD | GMEM -> SMEM | existing producer | PV WGMMA |
| vision IDs | S (private B=1 aux) | contiguous INT32 | GMEM -> scalar RMEM | score owner | mask predicate |
| S/P | M,N | pinned tile | RMEM | score/softmax owner | PV/backward |
| O/LSE | B,M,Hq,D / B,Hq,M | contiguous | GMEM | epilogue | caller/backward |

## 6. Data movement and pipeline

- Q/K/V and output movement use the unchanged pinned TMA/WGMMA pipeline.
- Vision IDs use ordinary global scalar reads through FA4 auxiliary tensors.
- Tail score coordinates use the pinned mask/seqlen predication before any
  wrapped auxiliary index can affect a valid output.
- No SMEM layout, stage count, barrier participant, named barrier, acquire,
  issue, commit, wait, consume, release, or drain transition changes.
- Sanitizers are nevertheless required because backward now exercises the
  custom mask in transposed ownership.

## 7. Predication and neutral values

- The mask receives the score fragment's logical M/N coordinates from FA4's
  existing identity-tensor partition.
- Invalid or rejected scores become negative infinity before softmax.
- The strict lower window boundary is `k > q - 1024`, not `>=`.
- A nonnegative equal vision ID relaxes only the causal upper bound. It never
  relaxes the lower window bound.
- Text ID `-1` never enables bidirectionality. Adjacent distinct nonnegative
  blocks never attend across their boundary.

## 8. Resource and cache budget

- Existing CTA: three warpgroups / 384 threads for the d256 SM90 path.
- No new SMEM allocation or pipeline descriptors.
- Additional live state is limited to auxiliary pointers, two INT32 loads,
  coordinate arithmetic, and predicate values.
- Record compiler-reported registers, static/dynamic SMEM, stack/local memory,
  PTX target, and relevant SASS for forward and backward.
- The mask callable hash and auxiliary metadata are already included in the
  pinned forward/backward cache keys. Tensor contents and pointers are not.

## 9. Correctness plan

- Trusted reference: `reference_attention` with FP32 score/softmax and the
  locked Gemma mask; independent upstream-style BF16 baseline for gradients.
- Forward policy: existing local O/LSE tolerances.
- Backward policy: the frozen EXP-0004 upstream-relative rule, unchanged.
- Lengths: 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025.
- Patterns: all text regression; a vision span crossing M/N tile boundaries;
  adjacent vision blocks; mixed text/vision; and an all-vision S1025 case that
  proves both far-future allowance and the strict far-past window edge.
- Values: seeded random plus coordinate-coded/zero-score V markers that expose
  future-token leakage or omission.
- Check O, FP32 LSE, separate dQ/dK/dV, shape/dtype/storage, nondefault stream,
  and at least three same-input repeats.
- Rejection tests: wrong ID rank/shape/device/dtype, unsupported B/S, non-BF16
  or misaligned Q/K/V, and K/V aliasing.
- Sanitizers: memcheck, synccheck, and racecheck at S128 and S129; include an
  S1025 window-edge case if runtime is practical.

## 10. Measurement and risks

- No performance measurement or speed claim in EXP-0007.
- Primary correctness risk: using native causal/local flags together with the
  custom predicate, causing backward to ignore the vision exception.
- Primary performance risk: the correctness path is scheduled as a dense
  custom mask and can inspect blocks outside the local/vision region.
- Rollback: keep `fa4_local_text_forward` on its unchanged native local path
  and reject multimodal inputs.
- Follow-on: packed varlen uses a separate globally indexed auxiliary layout,
  lower-right causal coordinates, and document-boundary matrix before any
  framework dispatch or block-sparse tile classifier is promoted.
