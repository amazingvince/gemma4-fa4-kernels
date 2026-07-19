# Kernel architecture plan

This plan starts at the prepared-Q/K/V FMHA boundary defined in
`docs/model-contract.md`. It does not assume K=V.

## 1. Two independent families

### Local d=256

- 50 layers, Q/KV heads 32/16, GQA 2.
- Window 1024; short mainloop and boundary-heavy masking.
- Native text and exact custom vision/document paths. EXP-0010 adds exact
  physical-tile classification for production-length custom masking; no
  efficiency claim follows from that correctness result.
- The pinned H100 SM90 path is project-validated for fixed B1 text and vision
  attention plus nonempty packed native/custom self-attention through
  per-sequence S1025. EXP-0009 extends native packed text through S262144;
  EXP-0010 extends exact metadata semantics through the same maximum inside
  its declared sparse resource envelope. These are not upstream/general
  d256-backward or performance claims.
- B300 dedicated d=256 path must gain local semantics; compare one-CTA and
  two-CTA schedules instead of assuming two-CTA wins.

### Global d=512

- 10 layers, Q/KV heads 32/4, GQA 8.
- Full causal; dominates attention arithmetic above modest sequence lengths.
- K and V are distinct prepared operands, so both need storage/dataflow and
  backward gradients.
- No current fused single-launch SM90/SM103 FA4 path covers the complete
  contract. H100 M1 uses exact composed forward and backward paths.

## 2. Global H100 starting design

M1 correctness outcome: the accepted forward path is a two-launch composition,
not the fused design below. A hash-locked patch enables the pinned SM90
asymmetric `(Dqk,Dv)=(512,256)` M128 x N32 specialization. The adapter runs it
once per V256 slab over identical full-d512 Q/K, requires identical FP32 LSE,
and concatenates the outputs. This is algebraically exact but duplicates
QK/softmax work; see EXP-0002.

EXP-0006 accepts the corresponding correctness-first backward composition over
B1, S=1..1024, BF16, 32Q/4KV, GQA-8, d512, causal, scale 1.0, and distinct K/V.
For each V256 slab it runs one M64 x N32 dKV-only main kernel plus two M64 x
N32 dQ-only kernels owning D256 dQ output slices after full-d512 score
recomputation. Persistent FP32 accumulators preserve
`dQ = dQ(V0) + dQ(V1)` and `dK = dK(V0) + dK(V1)` before one BF16 conversion;
dV slabs are converted separately and concatenated. This is six main launches,
uses FP32 bulk/atomic reductions, and makes no performance or deterministic-gradient
claim. The following candidates are the future fused/single-launch design
space and have not been implemented or timed.

EXP-0011 adds framework composition without changing those kernels. Global
training calls that are batched, padded, packed, document-split, or
lower-right are decomposed into exact per-segment fixed calls. At the EXP-0011
revision, every K segment was capped at 1024. Lower-right segments receive a
zero Q prefix solely to establish the correct causal coordinates, and only the
original Q rows are returned. With autograd disabled, separate
fixed/rectangular and packed-varlen
two-V256-slab forward adapters admit nonempty `1 <= Sq <= Sk <= 262144` and
preflight their output/LSE footprint against free HBM. These are compatibility
compositions, not fused d512 or performance results. EXP-0012 validates the
unchanged fixed/composed backward scheduler through K2048. EXP-0013 moves
accepted packed/lower-right training to native THD/cu-seqlens launches for
nonempty per-segment `1 <= Sq <= Sk <= 2048`, while retaining the exact Gemma
32Q/4KV/GQA-8/d512/causal/scale-1.0/distinct-K/V contract. Only a dedicated
native HBM-budget exception may select the exact EXP-0012 composer before
launch; validation, contract, assertion, and runtime failures propagate.
The native path is still the two-V256-slab forward plus split dQ/dKV backward,
not a fused d512 kernel. K>2048 training remains a distinct design problem.

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

The target fused/long-context design gives each output tile one owner and
accumulates in FP32 on-chip, avoiding whole-layer FP32 dQ/dK/dV workspaces.
GQA group ownership processes several Q heads per KV head, reuses K/V, and
reduces dK/dV contributions on-chip. Tune packing factors 2, 4, and 8.

EXP-0006 uses the allowed temporary global FP32 accumulator path for H100 M1.
Its dKV-only launch reduces all eight query heads for each KV head, while its
two dQ-only launches own D256 slices. Across the two V slabs the persistent FP32
accumulators are postprocessed once. This is not the intended maximum-context
or deterministic-owner design: FP32 bulk/atomic reductions make repeated gradients
non-bitwise, though every tested repeat passes the frozen numerical policy.

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
- Pinned Transformers prepared BHSD tensors become BSHD by a storage-sharing
  transpose when their unit-D, outer-stride, nonoverlap, and 16-byte alignment
  contract is legal. Packing/scattering copies are local to rows that actually
  require packing; fixed canonical views are not unconditionally materialized.
- Register `gemma4_fa4_h100` as a paired attention and mask backend. Never
  replace generic `flash_attention_4`: its mask adapter cannot represent the
  local vision future-token exception.
- Admit a native path only after structurally matching the exact pinned
  Transformers mask expression and captured vision/packed metadata. Preserve
  arbitrary masks and static-cache offsets for an exact inference-only
  fallback; reject gradient-capable arbitrary-mask/static-cache fallback because
  it is outside the accepted H100 d512 backward envelope.
- Native packed global training carries explicit INT32 Q/K cumulative lengths
  and exact host maxima. Its only composer fallback is the dedicated
  pre-launch HBM-budget exception; validation, contract, assertion, and runtime
  failures must propagate rather than silently changing routes.
- The hash-locked one-file Transformers patch forwards one authoritative
  vision-block tensor through both newly built and prebuilt generation masks.
  Explicit IDs take precedence over derivation from multimodal token types.
- Eager execution is the accepted integration boundary. Framework
  FakeTensor/`torch.compile` tracing fails closed until a compatible ABI,
  static-cache policy, and bounded compile-key design are validated.
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
4. Global d512 backward with separate dQ/dK/dV (complete for the scoped H100
   M1 envelope through EXP-0006's split dQ/dKV composition; EXP-0005's direct
   asymmetric-path rejection remains historical evidence).
5. Local multimodal forward and backward (complete for fixed B1 in EXP-0007).
6. Packed local native/custom forward and backward (complete for nonempty
   `B>=1`, `1 <= Sq <= Sk <= 1025` in EXP-0008).
7. Production-length native packed local text (complete through the locked
   `1 <= Sq <= Sk <= 262144` maximum in EXP-0009).
8. Production-length vision/document metadata through an exact block-sparse
   schedule (complete within the declared resource envelope in EXP-0010).
9. Eager pinned-Transformers attention/mask dispatch, authoritative vision
   metadata, padding/packed/lower-right offsets, and global no-grad forward
   through K262144 (functionally, sanitizer, and bounded-cache validated in
   EXP-0011 and recorded against implementation `e7f26bb`).
10. Fixed and exactly composed global backward through K2048 (complete in
    EXP-0012 with resource, numerical, sanitizer, and cache evidence).
11. Native THD/cu-seqlens global backward for nonempty per-segment
    `1 <= Sq <= Sk <= 2048` (complete in EXP-0013, with budget-only exact
    composer fallback and fail-closed propagation of all other failures).
12. K>2048 training, empty segments, deterministic gradients, and separately
    designed FakeTensor/`torch.compile` plus compiled/static-cache integration
    (active compatibility work).
13. H100 performance baselines and tuning only after the preceding correctness
    and sanitizer gates pass.
14. Resume B300 one-CTA/two-CTA work as its own target-host milestone.
15. Projection/norm/RoPE fusion and lower precision only after BF16 evidence.
