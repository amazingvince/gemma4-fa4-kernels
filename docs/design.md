# Kernel architecture plan

This plan starts at the prepared-Q/K/V FMHA boundary defined in
`docs/model-contract.md`. It does not assume K=V.

## 1. Two independent families

### Local d=256

- 50 layers, Q/KV heads 32/16, GQA 2.
- Window 1024; short mainloop and boundary-heavy masking.
- Text fast path and vision-block tile classifier.
- The pinned H100 SM90 path is project-validated as a fixed-length text
  baseline for forward and autograd backward over the declared M1 envelope;
  this is not an upstream/general d256-backward support claim.
- B300 dedicated d=256 path must gain local semantics; compare one-CTA and
  two-CTA schedules instead of assuming two-CTA wins.

### Global d=512

- 10 layers, Q/KV heads 32/4, GQA 8.
- Full causal; dominates attention arithmetic above modest sequence lengths.
- K and V are distinct prepared operands, so both need storage/dataflow and
  backward gradients.
- No current dense SM90/SM103 FA4 path covers the complete contract.

## 2. Global H100 starting design

M1 correctness outcome: the first accepted forward path is a two-launch
composition, not the fused design below. A hash-locked patch enables the
pinned SM90 asymmetric `(Dqk,Dv)=(512,256)` M128 x N32 specialization. The
adapter runs it once per V256 slab over identical full-d512 Q/K, requires
identical FP32 LSE, and concatenates the outputs. This is algebraically exact
but duplicates QK/softmax work; see EXP-0002. The following candidates are the
future single-launch design space and have not been implemented or timed.

Initial candidates:

```text
M64 × N32 or M64 × N64
one TMA producer warpgroup role
at least two consumer warpgroup roles
output-D split 256 + 256
FP32 accumulators
one CTA per SM initially
```

An M64×D512 FP32 output contains 32768 values. One 128-thread warpgroup would
hold 256 accumulator values per thread before score, softmax, address, and
pipeline state, which is structurally too close to the register limit. Split
D across consumer groups.

Shared-memory plans to compare:

1. Q plus one-stage K and V buffers;
2. K/V time-sharing after QK consumes K (storage reuse, not operand identity);
3. N32 with deeper staging;
4. smaller logical M packed across GQA heads.

Smem byte budget (H100, 228 KB/SM, BF16, distinct K and V — the aliasing
saving from the old K=V assumption does NOT apply):

```text
Q  tile M64  × D512 = 64 KB   (resident once)
K  tile N64  × D512 = 64 KB   per stage
V  tile N64  × D512 = 64 KB   per stage
-> one KV stage:  Q + K + V           = 192 KB  (fits, zero pipelining)
-> N32 tiles:     Q + 2×(K32+V32)     = 192 KB  (2 stages, shallow)
-> time-shared:   Q + K/V slot ×2     = 192 KB  (2 stages if QK fully
                                                 drains K before V load)
```

Consequence: double-buffered N64 with distinct K and V does not fit alongside
a resident M64 Q tile. The viable candidates are N32 staging, K/V slot
time-sharing, or streaming Q in D-chunks to shrink the resident Q footprint.
Measure all three before committing; do not assume the N64 shapes from
hd128/hd256 kernels transfer.

## 3. Global B300 starting design

Compare:

- one CTA with M32/M64 and sequential 256- or 128-wide output slabs;
- a two-CTA cluster where each CTA owns half of output D;
- cooperative score production versus duplicated QK;
- static versus CLC scheduling by workload regime.

TMEM makes d=512 possible but does not remove S/O placement, cluster,
softmax, or epilogue constraints.

### SM103 (B300) hardware deltas — applies to BOTH local d256 and global d512 paths

Pinned-upstream facts (see `flash_attn/cute/flash_fwd_sm100.py` and
`softmax.py` at the locked FA revision):

- **Fast hardware `ex2`**: the B200-era software exp2 emulation
  (`enable_ex2_emu`, polynomial on FMA units) is *disabled* when
  `arch.is_family_of(Arch.sm_103f)`. Any softmax-path experiment must state
  which exp path it ran; flipping the flag is a one-knob ablation worth an
  early experiment on each kernel family.
- **`tcgen05.ld.red` row-max reduction**: SM103 fuses the row-max into the
  TMEM load (`use_ldred_rowmax`), removing the software fmax tree. Verify in
  SASS that it actually engages for new kernel variants; it changes where the
  softmax bottleneck sits relative to B200 profiles.
- **Tuning keys**: upstream `_tune_key` already includes `is_sm103`. Every new
  compile-time flag we add (local masking, d-slab width, CTA-pair mode) must
  be threaded into both the tune key and the compile-cache key, keyed
  separately for sm_103 — never inherit B200 (or B100-class) tuning constants
  without measurement, and never let sm_103 results overwrite sm_100 table
  entries.
- Practical consequence of the two features: SM103 forward kernels tend to
  move from SFU/softmax-limited toward smem/TMEM-bandwidth-limited earlier
  than B200 intuition suggests; budget profiling time accordingly.

## 4. Backward ownership

Use three logical stages:

1. preprocess `D = sum(O * dO)` and internal LSE representation;
2. Q-major owner-computes dQ;
3. K-major owner-computes dK and dV, initially as separate kernels if live
   accumulator pressure is excessive.

Each output tile has one owner and accumulates in FP32 on-chip, avoiding whole
layer FP32 dQ/dK/dV workspaces at long context. GQA group ownership processes
several Q heads per KV head, reuses K/V, and reduces dK/dV contributions
on-chip. Tune packing factors 2, 4, and 8.

A temporary global FP32 accumulator path is acceptable for bring-up but is not
the intended maximum-context design.

## 5. Local mask specialization

Classify tiles as:

1. outside the window: skip;
2. fully valid causal interior: no element predicate;
3. same-vision-block interior: full tile inside the lower window;
4. causal diagonal, window edge, document edge, or vision edge: predicate.

Backward uses the transposed ownership/range calculation. Do not route the
whole workload through a generic per-element mask if tile classification can
prove an interior case.

## 6. Optional model-specific fusion after base FMHA

A later global kernel may consume the shared projection source Z and form:

```text
K = partial_RoPE(KNorm(Z))
V = VNorm(Z)
```

Its backward may return dZ only after applying both distinct preparation
adjoints. This is a separate fusion milestone with new tests and source-level
integration; it is not a shortcut for the base d=512 attention kernels.

## 7. Architecture and integration boundaries

- SM90 and SM103 share interfaces and references, not tuning tables or
  pipeline assumptions.
- Runtime sequence lengths, offsets, and vision IDs stay out of compile keys.
- Support fixed BSHD and packed varlen layouts through explicit adapters.
- The base checkpoint has no cross-layer KV reuse (`num_kv_shared_layers=0`); keep
  support for future variants outside the initial fast-path contract.
- Tensor-parallel or KV-replicated per-rank shapes can expose GQA ratios
  1/2/4/8, especially in the global family; dispatch and tests cover all four
  while preserving full-model ratios 2 and 8 as first-class cases.

## 8. Current H100 implementation order

The active session is H100-only; B300 remains deferred rather than sharing an
experiment or tuning table with SM90.

1. Contract/oracle/benchmark rig (complete).
2. Local d256 fixed-length text forward and backward (complete for the scoped
   M1 envelope; see EXP-0001, EXP-0003, and EXP-0004).
3. Global d512 fixed-length text forward (complete as the exact two-launch
   correctness composition in EXP-0002).
4. Global d512 backward with separate dQ/dK/dV.
5. Local multimodal/varlen forward and backward.
6. Framework dispatch, KV-sharing integration, and context-parallel offsets.
7. H100 performance baselines and tuning only after the preceding correctness
   and sanitizer gates pass.
8. Resume B300 one-CTA/two-CTA work as its own target-host milestone.
9. Projection/norm/RoPE fusion and lower precision only after BF16 evidence.
