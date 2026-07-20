# H100 global dQ FP8 feasibility brief

> Review gate: this is an approximate, opt-in research path. It changes operand
> dtypes, scale metadata, MMA atoms, and the gradient contract. Do not implement
> it until a human approves those choices. It must never replace the exact BF16
> default or be described as exact Gemma 4 attention.

## Question and short answer

FP8 can make a full-V512 dQ tile fit in H100 shared memory, but not in a single
32-thread warp. Hopper WGMMA is issued by a 128-thread warpgroup. The useful
candidate is therefore two MMA warpgroups at M64xN32, with only V and dO staged
as FP8. That preserves the accepted N16 ownership per warpgroup and recovers
49,152 shared-memory bytes compared with full-V512 BF16.

This is technically plausible on H100, but it is not supported by the pinned
FA4 backward interface. It requires a separate mixed-dtype backward kernel and
produces an approximate dQ. The preferred next exact experiment is BF16 V/dO
slot time-sharing; FP8 should remain a parallel feasibility track.

## Pinned-source evidence

- CUTLASS v4.6.0 Hopper `examples/python/CuTeDSL/cute/hopper/kernel/attention/fmha.py`
  accepts `Float8E4M3FN` and retains FP32 QK/PV accumulators. This proves the
  pinned H100 CuTe stack has FP8 WGMMA building blocks; it does not prove this
  backward design.
- Pinned FA4 `flash_attn/cute/interface.py` explicitly raises
  `NotImplementedError` for FP8 backward and currently restricts its public FP8
  path to SM100.
- Pinned FA4 `flash_attn/cute/flash_bwd_sm90.py` uses one `tiled_mma_SdP` for
  both BF16 QK and `dO @ V^T`, and its SM90 backward path assumes homogeneous
  Q/K/V/dO types. Mixed BF16/FP8 operands therefore require a structural
  refactor, not a flag flip.

## Candidate numerical contract

- Keep Q, K, scores, masking, softmax, LSE, dS, and the `dS @ K` accumulation
  in their accepted BF16-input/FP32-accumulator path.
- Quantize only V and dO to FP8 before the dQ kernel, with explicit recorded
  scales. Use FP8 WGMMA with FP32 accumulation for `dP = dO @ V^T`.
- Apply the product of the V and dO dequantization scales to the FP32 dP
  accumulator before `dS = P * (dP - D)`.
- Keep the accepted BF16 dKV kernels. Consequently dK and dV remain exact to
  that path while dQ is approximate; the three gradients no longer represent
  one numerically identical BF16 backward evaluation.
- Expose this only through a separately named opt-in API and compile key.
  Results must name the FP8 formats, scale granularity, saturation policy, and
  whether quantization time is included.

Per-tensor scaling is the smallest first experiment. E4M3FN is the pinned
official Hopper example's supported input type. Choosing E4M3FN versus E5M2
for dO remains an empirical decision based on measured range and error; no
format is approved by this brief.

## Shared-memory budget

The estimates include one Q, K, V, dO, dS, and dQ-accumulator slot plus stats
and approximately 512 bytes of existing overhead.

| M64xN tile and staged types | Estimated dynamic SMEM | H100 fit |
|---|---:|---|
| accepted N32, V256 BF16 | 218,112 B | yes |
| full V512 N32, all BF16 | 267,264 B | no, +34,816 B |
| full V512 N16, all BF16 | 232,448 B | exactly at limit |
| full V512 N32, V+dO FP8 only | 218,112 B | yes |
| full V512 N32, all FP8 | about 166,912 B | yes, but unnecessary loss |

The mixed candidate retains BF16 Q/K and therefore avoids quantizing logits.
It also retains two warpgroup ownership of N32, unlike the rejected EXP-0032
one-warpgroup kernel.

## Required implementation changes

1. Add a distinct FP8 dP tiled MMA and layouts alongside the BF16 QK/dS MMA;
   do not reuse the current homogeneous `tiled_mma_SdP` object.
2. Add FP8 V/dO descriptors, aligned storage, scale arguments, saturation and
   non-finite policy, and all of them to the compile/runtime contract as
   appropriate.
3. Accumulate dP in FP32, apply dequantization exactly once, then form BF16 dS
   for the accepted BF16 dQ WGMMA.
4. Keep dKV on the accepted BF16 path and make the mixed-gradient semantics
   explicit at the public boundary.
5. Include prequantization buffers in the memory preflight and quantization in
   end-to-end timing. A kernel-only number may also be reported, but not alone.

## Falsification plan

- First measure real V and dO range, FP8 saturation rate, quantization error,
  and per-tensor versus finer-grained scale cost on representative global
  attention backward inputs.
- Compile only the dP micro-path and inspect SASS to prove FP8 WGMMA engages;
  reject any fallback conversion that performs BF16 WGMMA instead.
- Compare O/LSE unchanged and separately report dQ error distributions,
  cosine similarity, non-finite behavior, and accumulation sensitivity through
  S128, S8K, and adversarial high-dynamic-range inputs.
- Run memcheck, synccheck, and racecheck before performance claims.
- Benchmark quantize+dQ and whole backward against the accepted BF16 kernel.
  Require a material end-to-end win, not merely lower shared memory.
- Before any training claim, run an explicitly approved convergence study.

## Decision boundary

No implementation is approved yet. Human review must choose the scale
granularity, FP8 format for each operand, saturation policy, exposed API, error
thresholds, and whether an approximate gradient is acceptable for the intended
training or inference-adjacent use. Failure cannot affect the accepted path:
the new API and cache key remain opt-in, and rollback is deletion of that
isolated specialization.
