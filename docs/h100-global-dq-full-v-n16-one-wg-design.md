# H100 full-V N16 one-warpgroup dQ design brief

> Review gate: this design changes pipeline and named-barrier participant
> counts. Do not implement it until a human explicitly approves this brief.

## 1. Environment and version

- Date: 2026-07-20
- `nvidia-cutlass-dsl`: 4.6.0.dev0
- CUTLASS: v4.6.0, `e6233cbac5d7c7a865c19c91cd684ceece19513c`
- CUDA toolkit / driver: 12.8.93 / 580.126.09
- Python / framework: 3.12 / PyTorch 2.8.0+cu128
- GPU: NVIDIA H100 80GB HBM3, SM90, 132 SMs
- Official example: CUTLASS v4.6.0 Hopper `fmha.py`; production base is
  pinned FA4 `flash_bwd_sm90.py` at `77aacb68...` plus the exact project patch.
- Verified APIs: `FlashAttentionBackwardSm90`, SM90 WGMMA/TMA construction,
  `PipelineTmaAsync`, `CooperativeGroup`, `NamedBarrier`, register allocation,
  dQ R2S/bulk-reduce handoff, compile and launch APIs.

## 2. Operation contract

- Operation: exact causal Gemma 4 global d512 attention dQ backward.
- Formula: `P=softmax(QK^T)` at scale 1.0; `dP=dO V^T` over full V512;
  `D=sum_j(O_j*dO_j)`; `dS=P*(dP-D)`; `dQ=dS K`.
- Inputs: BF16 Q `[B,Sq,32,512]`, K/V `[B,Sk,4,512]`, dO
  `[B,Sq,32,512]`, FP32 LSE-log2/dPsum, plus packed THD/int32 cumulative
  equivalents.
- Outputs: one FP32 M64xD256 dQ accumulation tile per dQ specialization,
  followed by the existing one-time BF16 postprocess. dK and dV remain separate
  and are produced by the unchanged two N32/Dv256 dKV slab launches.
- FP32 accumulation, global causal mask, NaN/Inf behavior, empty packed segment
  policy, non-aliasing, and accepted nondeterminism remain unchanged.

## 3. Shape and layout regime

- Fixed rank-4 BSHD through S2048 and native packed rank-3 THD through guarded
  K262144.
- Dynamic: batch, totals, cumulative values, and exact maxima within existing
  scheduler classes.
- Static dQ candidate: SM90 BF16 GQA8, M64xN16, Q/K/V512, D256 output offset
  0 or 256, one Q/dO/PdS stage, one 128-thread MMA warpgroup, four producer
  warps, 256 threads total.
- Contiguous storage and accepted TMA alignment classes remain.
- Representative/tail cases: S128 first discriminator; fixed S1024/1025/2048;
  packed mixed Sq/Sk, empty segments, S8K/S64K performance.
- Reject/fallback on wrong geometry/arch/dtype, invalid cumulative lengths,
  memory-budget failure, compile/resource failure, or any numerical/sanitizer
  regression. Fallback is byte-locked EXP-0029.
- Bounded variants remain `dkv`, `dq_lo`, and `dq_hi`; the key explicitly adds
  per-variant N/Dv/thread/WG ownership without runtime lengths.

## 4. Target and kernel family

- Target: H100 SM90 only.
- Family: nonpersistent warp-specialized TMA + WGMMA backward.
- Portable fallback: accepted two-WG N32 slab composition.
- Atom: existing SM90 WGMMA. One consumer WG owns full N16, satisfying the
  K16 dS operand conversion that rejected EXP-0031's two independent N8 halves.
- The relaxed GQA guard applies only when `compute_dQ=True`,
  `compute_dKV=False`, and QK/V dimensions are both 512. All asymmetric or dKV
  GQA cases continue to require two MMA warpgroups.

## 5. Tile and ownership hierarchy

- CTA tile: M64xN16, full reduction width V512, output D256.
- Block: 256 threads. Warps 0--3 are the producer group; warps 4--7 are the
  sole MMA consumer warpgroup.
- Warp 0 owns TMA descriptor/load issue. Warp 1 drains dQ accumulation.
  Consumer WG0 owns QK, dP, dS, dQ, its R2S copy, and the only dQ barrier slot.
- Scheduler grid has twice the N tiles of N32. Two dQ launches replace four;
  total dQ CTA count is unchanged, while QK and dQ arithmetic are halved.

| Tensor | Tile | Memory | Owner / consumer |
|---|---|---|---|
| Q | M64xD512 | GMEM/SMEM | warp 0 / WG0 |
| K | N16xD512 | GMEM/SMEM | warp 0 / WG0 |
| V | N16xD512 | GMEM/SMEM | warp 0 / WG0 |
| dO | M64xD512 | GMEM/SMEM | warp 0 / WG0 |
| LSE,dPsum | M64 FP32 | GMEM/SMEM | warp 0 / WG0 |
| dS | M64xN16 BF16 | RMEM/SMEM | WG0 / dQ WGMMA |
| dQ accum | M64xD256 FP32 | RMEM/SMEM/GMEM | WG0 / warp 1 |

## 6. Data movement

| Edge | Primitive | Layout/alignment | Tail | Completion |
|---|---|---|---|---|
| GMEM to SMEM | existing TMA bulk loads | pinned swizzles, 1024-byte buffer alignment | existing M/N predicates | TMA transaction barrier |
| SMEM to WGMMA | existing descriptors | one WG owns N16 | exact D, predicated sequence | consumer wait |
| WGMMA to FP32 | existing atoms | full WG0 fragments | masked neutral scores | warpgroup wait |
| FP32 to SMEM | existing vector R2S | one M64xD256 slot | existing predicate | dQ full barrier |
| SMEM to GMEM | existing bulk reduce-add | FP32 accumulator | existing predicate | async read completion |

N16 makes V512 equal the old N32/V256 byte span. No second shared slot, TMEM,
new copy primitive, or descriptor lifetime is introduced.

## 7. Pipeline state machine

- Stages: Q=1, dO=1, PdS=1.
- Producer: four-warps register-reduced group; warp 0 issues TMA, warp 1 stores
  dQ. Consumer: one 128-thread MMA warpgroup.
- TMA pipeline consumer group changes from eight warps to four warps.
- `PdS` named barrier changes from 256 to 128 participants.
- The single `dQEmptyWG0`/`dQFullWG0` handoff has 160 participants: 128 MMA
  threads plus the 32-thread store warp. WG1 barrier IDs are not entered.
- Pipeline state `(index,phase)`, prologue, steady-state advance, drain, store
  completion, and scheduler reset remain the existing code paths.

| Transition | Agent | State | Event | Match |
|---|---|---|---|---|
| acquire | warp 0 | current Q/dO stage | producer acquire | prior WG0 release |
| issue | warp 0 | current stage | TMA Q/K/V/dO/stats | transaction bytes |
| arrive | TMA engine | current stage | transaction complete | WG0 wait |
| wait | WG0 | current stage | consumer wait | TMA arrive |
| publish dS | WG0 (128) | PdS stage 0 | fence + PdS arrive/wait | same 128 participants |
| consume | WG0 | stage 0 | dQ WGMMA | dS publish |
| dQ full | WG0 + warp 1 (160) | slot 0 | R2S/full barrier | bulk reduce-add |
| dQ empty | warp 1 + WG0 (160) | slot 0 | read completion/empty barrier | next R2S |
| release | WG0 | current stage | consumer release | next producer acquire |

Review must confirm that no code path still expects WG1 to arrive at `PdS`, Q,
dO, or dQ barriers and that the store warp's loop count is exactly one.

## 8. Predication and neutral values

Existing causal coordinate tensors, M/N predicates, invalid load fills,
suppressed stores, masked-score neutral values, and vector-tail behavior remain.
N16/D512/D256 are exact vector and WGMMA multiples. Empty packed segments use
the accepted positive-aggregate scheduler behavior.

## 9. Resource budget

- 256 threads / 8 warps / one CTA; no cluster.
- Predicted dynamic SMEM: 232,448 bytes, exactly the H100 opt-in per-block
  limit. Delta from baseline: K saves 16,384; dO adds 32,768; dS saves 2,048;
  V is unchanged; net +14,336 over measured 218,112.
- dQ shared accumulator stays 65,536 bytes. No TMEM.
- Proposed register request: 240 for WG0 and 32 for producer warps. Generated
  metadata must prove registers/spills; any main-kernel spill or SMEM overflow
  rejects the candidate.
- One CTA/SM remains expected. The primary utilization risk is only one active
  MMA warpgroup in that resident CTA.

## 10. Correctness plan

- Trusted project PyTorch reference and optional pinned HF oracle; locked
  tolerances unchanged.
- Fake compile, then S128 fixed O/LSE/dQ/dK/dV with three repeats and a
  nondefault stream; fixed boundaries; packed/asymmetric/tail/empty/isolation;
  analytic long sentinels; full integration/cache regressions.
- Validate invalid/fallback cases, bounded cache keys, additional dPsum memory
  preflight, and separate gradients.
- Run memcheck, synccheck, and racecheck on fixed S1025 and packed mixed cases.
- Retain IR/PTX/SASS, 256-thread launch, N16 grid, register/spill/SMEM metadata,
  and confirm exactly two dQ plus two unchanged dKV main launches.

## 11. Benchmark plan

- Baseline: EXP-0029 global S8K/S64K hot/cold bwd and fwd_bwd.
- Candidate: 10 warmups, 30 CUDA-event reps, median/p25/p75/IQR; unlocked
  clocks labeled and sampled. No clock mutation.
- Metrics: ms, exact FLOP/s, launch durations, registers, spills, SMEM,
  sanitizer outcomes. NCU counters remain unavailable unless pod permissions
  change.
- Search space: exactly baseline versus full-V512 N16 one-WG dQ.
- Accept only with at least 10% S8K backward improvement, non-overlapping IQRs,
  confirming S64K direction, and all correctness/safety/resource gates clean.

## 12. Assumptions and risks

- Verified: EXP-0031 failed specifically because each of two WGs owned N8; one
  WG owns valid N16. Existing FA4 code already supports `num_wg_dQ=1` store-loop
  cardinality and one dQ barrier slot in its `dQ_single_wg` path.
- Unverified: the full kernel supports one MMA WG after the scoped GQA guard
  relaxation; 240 registers suffice; exactly-limit SMEM launches; one WG can
  use enough tensor-core throughput to win.
- Version-sensitive: accumulator conversion, WGMMA N16, pipeline participant
  derivation, register allocation, and auto-SMEM launch.
- Correctness risk: a hidden WG1 arrival/dependency or changed FP32 association.
- Performance risk: half-idle MMA capacity at one CTA/SM and doubled N-grid
  scheduling offset the arithmetic reduction.
- Rollback: restore EXP-0029's byte-locked patch and strict environment hash.
