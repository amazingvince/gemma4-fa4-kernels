# H100 M1 packed local text: production-length design brief

## 1. Environment and version

- Date: 2026-07-19.
- `nvidia-cutlass-dsl`: 4.6.0.dev0.
- CUDA toolkit / driver: CUDA 12.8 / 580.126.09.
- Python / framework: Python 3.12.3 / PyTorch 2.8.0+cu128.
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, target `sm_90a`.
- Upstream: FlashAttention commit
  `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6` plus the recorded project patch.
- Verified interfaces: pinned `flash_attn_varlen_func`, `_flash_attn_fwd`,
  `_flash_attn_bwd`, `SeqlenInfoQK`, and SM90 `BlockInfo` local ranges.
- Starting exemplar: the unchanged pinned
  `flash_attn/cute/flash_fwd_sm90.py`, `flash_bwd_sm90.py`, and
  `block_info.py` native local path. No new MMA, copy, pipeline, or barrier
  protocol is introduced.

## 2. Operation contract

For packed sequence `b`, query-local coordinate `q`, and key-local coordinate
`k`, let `q_abs = q + Sk_b - Sq_b`. Text attention keeps exactly

```text
k > q_abs - 1024 AND k <= q_abs
```

with scale 1.0. Inputs are distinct contiguous BF16 packed THD tensors:
Q `(Tq,32,256)`, K/V `(Tk,16,256)`. Output O is BF16 `(Tq,32,256)`;
LSE is FP32 `(32,Tq)`. Score, LSE, and gradient reductions follow the pinned
FA4 FP32 accumulation contract; dQ, dK, and dV remain separate BF16 outputs.
Empty segments and aliased K/V remain rejected. The path is nondeterministic
only where the already accepted upstream backward reduction is nondeterministic.

## 3. Shape and layout regime

- Dynamic: packed totals, batch count, every nonempty `Sq_b` and `Sk_b`.
- Bounds: `1 <= Sq_b <= Sk_b <= 262144`; cumulative totals must fit INT32.
- Static: BF16, 32Q/16KV, GQA-2, d256, scale 1.0, W1024, H100 SM90.
- Layout: contiguous THD with 16-byte-aligned bases; CUDA INT32 cumulative arrays.
- Representative: equal S2048/S4096, ragged and lower-right long-K suffixes.
- Adversarial: Q1/K262144; strict excluded K=`q_abs-1024` and included
  K=`q_abs-1023`; 64/80/128 tile tails; multi-sequence isolation.
- Rejected: any length above the locked model maximum, empty segments,
  metadata-bearing vision/document calls above S1025 until the separate exact
  sparse schedule is accepted, non-SM90, non-BF16, or wrong model geometry.
- Specialization count: unchanged and bounded. Runtime sequence values and
  cumulative contents do not enter the compile key; only existing native
  forward and bounded backward structural variants may exist.

## 4. Target and kernel family

- Target: H100 SM90a only.
- Family: packed local FMHA through pinned FA4 native local scheduling.
- Fallback: explicit `UnsupportedH100Path`; no silent dense or semantic substitute.
- Compute atom / CTA hierarchy: unchanged from the accepted upstream kernels.
  EXP-0008 observed M128 x N80 forward and M64 x N64 backward main kernels.

## 5. Ownership and dataflow

| Tensor | Logical modes | Storage | Owner / consumer |
|---|---|---|---|
| Q | packed Tq,Hq,D | BF16 GMEM | native forward/backward Q tiles |
| K | packed Tk,Hkv,D | BF16 GMEM | native window-bounded K tiles |
| V | packed Tk,Hkv,D | BF16 GMEM | native window-bounded V tiles |
| cuQ/cuK | B+1 | INT32 GMEM | `SeqlenInfoQK` segment/offset resolution |
| O | packed Tq,Hq,D | BF16 GMEM | forward epilogue / backward input |
| LSE | Hq,Tq | FP32 GMEM | forward softmax statistic / backward input |
| dQ/dK/dV | corresponding input shape | BF16 GMEM | unchanged upstream backward epilogues |

The adapter changes only admission. Native `BlockInfo` continues to bound
forward K tiles and transposed backward Q tiles from runtime sequence-local
coordinates. No metadata tensor or block-sparse list is present in EXP-0009.

## 6. Movement, synchronization, and boundaries

All TMA/WGMMA movement, stage counts, barriers, drain, and epilogue behavior
remain inherited from the already accepted pinned kernels. The new correctness
risk is runtime coordinate width and far-offset lower-right arithmetic, not a
new synchronization state. Native predication must still zero/ignore padded
tile coordinates, and fully masked rows are impossible because every nonempty
query suffix includes its aligned self key.

## 7. Resource and memory policy

- Main-kernel resources must match the accepted native EXP-0008 objects unless
  retained generated code proves an intentional bounded variant.
- Long calls receive an explicit preflight in the probe before allocation.
- The maximum-context Q1/K262144 sentinel validates far offsets without
  pretending to validate a full 262144-query training allocation.
- A separate full-Q long smoke is required at a feasible recorded length.

## 8. Correctness and debug plan

- CPU adapter tests for native admission through 262144 and rejection at 262145.
- Preserve the custom metadata S1025 guard.
- Fake compile at long runtime maxima for forward, backward, and LSE-only paths.
- H100 reference comparisons at tractable >1025 equal and lower-right lengths.
- Exact Q1/K262144 O/LSE/dV ownership sentinel.
- Ragged multi-sequence isolation, nondefault stream, and repeat checks.
- Cache probe varying long totals/maxima without unbounded new objects.
- Memcheck, synccheck, and racecheck on representative >1025 tail/multiblock cases.
- Retain PTX/cubin/SASS hashes and resource observations.

## 9. Measurement and risks

EXP-0009 authorizes no performance measurement or speed claim. Primary risks
are 32-bit offset mistakes at the model maximum, accidental dense work despite
native local flags, memory pressure in full-Q backward, and sequence maxima
leaking into compilation. Rollback is the existing S1025 admission guard.

