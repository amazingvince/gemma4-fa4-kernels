# H100 global dKV D256-streaming design brief

This brief was the pre-implementation review gate for EXP-0037. The candidate
subsequently passed its declared gates and is now the accepted default; the
EXP-0035 two-slab dKV route remains available as the flag-off rollback.

## 1. Environment and version

- Date: 2026-07-20
- `nvidia-cutlass-dsl`: 4.6.0.dev0
- CUTLASS: v4.6.0, `e6233cbac5d7c7a865c19c91cd684ceece19513c`
- FlashAttention: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6` plus accepted project patch
- CUDA / driver: 12.8.93 / 580.126.09
- Python / PyTorch: 3.12.3 / 2.8.0+cu128
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, target SM90a
- Official example: CUTLASS v4.6.0 Hopper `fmha.py`; production starting
  point is pinned FA4 `flash_attn/cute/flash_bwd_sm90.py`
- Verified APIs: existing TMA producer/consumer lifecycle, WGMMA
  `gemm_w_idx`, shared-view fence, named barriers, and FP32 bulk-reduction
  epilogue in the pinned implementation

## 2. Operation contract

- Operation: dK/dV portion of exact global Gemma 4 attention backward
- Formula: `S=QK^T`, `P=exp(S-LSE)`, `dP=dO V^T`,
  `dS=P*(dP-row_sum(dO*O))`, `dK=dS^T Q`, `dV=P^T dO`
- Inputs: BSHD or THD BF16 Q/K/V/dO; 32 Q heads, 4 KV heads, D=512
- Outputs: distinct FP32 dK/dV accumulators followed by separate BF16 dK/dV
- Accumulation: FP32 S, dP, dK, and dV
- Mask: global causal only; scale exactly 1.0
- Empty/tail behavior: unchanged accepted fixed/packed rules; all-empty
  physical workloads remain rejected
- Aliasing: prepared K and V remain distinct; no in-place operand writes
- Determinism: unchanged accepted nondeterministic FP32 GQA reduction policy

## 3. Shape and layout regime

- Dynamic: accepted sequence totals/lengths and legal outer strides
- Static candidate: M64xN32, Dqk=Dv=512, dO stream chunk D256, two MMA
  warpgroups, one producer warp, one Q stage and one three-generation dO slot
- Alignment: unchanged 16-byte public alignment; D256 offsets are 512-byte
  aligned for BF16
- Specializations: one `dkv_d256_stream`; the accepted `dkv` key and both
  EXP-0035 dQ keys remain available and unchanged
- Rejection: non-SM90, non-BF16, non-global, non-d512, non-GQA8, block-sparse,
  or deterministic requests remain on existing validation/routing
- Expected cache effect: one additional bounded candidate key for each of the
  existing fixed/packed scheduler classes; runtime lengths and tensors stay
  out of the key

## 4. Target and kernel family

- Architecture: Hopper SM90a WGMMA
- Family: accepted nonpersistent warp-specialized SM90 backward, dKV-only
  structural variant
- Compute: BF16 WGMMA with FP32 accumulation
- Portable fallback: accepted two V256 dKV slab launches
- FP8, SM103, and local-attention paths remain untouched

## 5. Tile and ownership

- CTA: one N32 K/V tile and successive M64 Q tiles
- Roles: one 128-thread non-MMA warpgroup contains the elected producer;
  two 128-thread MMA warpgroups consume
- Q512, K512, and V512 remain resident while one M block is processed
- dO uses one M64xD256 slot across low, high, and low-replay generations
- Both MMA warpgroups retain disjoint accumulator ownership. dK is produced
  once. dV low/high fragments remain distinct until sequential epilogue stores.
- After the final M block and complete WGMMA drain, Q is dead. Its 64 KiB SMEM
  allocation is then reinterpreted as the sequential FP32 dK/dV reduction
  scratch; producer writes to Q have ended before this alias becomes live.

| Tensor | Logical shape | SMEM tile | Owner | Consumer |
|---|---|---:|---|---|
| Q | M x 512 | 64 x 512 | producer, one Q generation | QK and dK WGMMA |
| K | N x 512 | 32 x 512 | producer once per N tile | QK WGMMA |
| V | N x 512 | 32 x 512 | producer once per N tile | two dP WGMMA views |
| dO | M x 512 | 64 x 256 | producer low/high/low replay | dP and dV WGMMA |
| LSE/dPsum | M | 64 FP32 each | producer; dPsum duplicated per dO generation | pointwise |
| P/dS | M x N | 64 x 32 BF16 each | both MMA WGs | dV/dK WGMMA |
| dK accum | N x 512 | FP32 registers | disjoint MMA WG fragments | epilogue |
| dV accum | N x 512 | two FP32 register fragments | disjoint D256 halves | epilogue |
| reduction scratch | max(N x 512) FP32 | released Q storage | MMA WGs then elected bulk reducer | GMEM dK/dV accum |

## 6. Data movement

| Edge | Primitive | Alignment/tail | Completion |
|---|---|---|---|
| Q/K/V GMEM to SMEM | existing TMA loads | accepted M/N predication | Q/dO pipeline barriers |
| dO GMEM to SMEM | TMA D256 low/high/low | accepted M predication | three dO barrier generations |
| SMEM to accumulators | SM90 WGMMA | exact D512 or D256 views | WGMMA group waits |
| P/dS registers to SMEM | accepted R2S copy | exact M64xN32 | PdS named barrier |
| dK/dV registers to released Q scratch | universal 128-bit R2S | exact owned fragments | epilogue named barrier |
| scratch to GMEM FP32 accum | accepted bulk reduce-add | accepted N tail | async bulk wait/commit |

- SMEM layouts/swizzles come from the accepted pinned SM90 helper.
- The candidate removes the dedicated 64 KiB dKV epilogue scratch and adds
  16 KiB of resident V versus one accepted slab. The compiled allocation is
  174,080 dynamic shared bytes rather than 222,208 bytes.
- No TMEM or inline PTX is introduced.

## 7. Pipeline state machine

Q and dO no longer advance in lockstep. Q advances once per M block; dO
advances three times. Each one-stage phase toggle is a release-before-reuse
proof. The producer must not issue high dO until both MMA warpgroups release
low, and must not replay low until both release high after dV-high consumes it.

| Transition | Agent | Generation | Event | Matching transition |
|---|---|---|---|---|
| acquire/issue | producer | Q | load Q/LSE; first M also loads K/V | prior Q release |
| acquire/issue | producer | dO-low | load dO[0:256] and full dPsum | prior replay release |
| wait/consume | MMA WGs | Q + dO-low | issue full QK and low dP | producer arrivals |
| release | MMA WGs | dO-low | WGMMA low-dP drain, then release | high acquire |
| acquire/issue | producer | dO-high | load dO[256:512] and duplicated dPsum | low release |
| wait/consume | MMA WGs | dO-high | accumulate high dP; form P/dS; issue dV-high and dK | producer arrival |
| release | MMA WGs | Q | after dK WGMMA no longer references Q | next Q acquire |
| release | MMA WGs | dO-high | after dV-high WGMMA no longer references dO | replay acquire |
| acquire/issue | producer | dO-low replay | reload dO[0:256]; no V/Q/K transaction | high release |
| wait/consume | MMA WGs | replay | accumulate dV-low | producer arrival |
| release | MMA WGs | replay | final dV-low drain then release | next block low acquire |
| epilogue | MMA WGs/elected warp | after all M blocks | drain all groups; reuse dead Q storage for dK, dV-low, dV-high in sequence | async bulk waits/barriers |

The P/dS named barrier remains a 256-thread, two-MMA-warpgroup rendezvous.
Every dO release follows `warpgroup.wait_group(0)` for the generation whose
descriptor is being released. The epilogue alias begins only after the final
Q consumer release and all WGMMA groups drain.

## 8. Predication and neutral values

- M/N coordinates, causal mask, and all-masked behavior are unchanged
- D is exactly two D256 chunks; no D tail exists
- Invalid M/N loads retain accepted TMA zero-fill/predication behavior
- Stores retain accepted dK/dV tail predicates and conversion policy

## 9. Resource budget

- Threads: 384 (one 128-thread non-MMA warpgroup plus two MMA warpgroups)
- Dynamic SMEM: 174,080 bytes compiled, comprising Q64 + K32 + V32 + dO32 +
  P4 + dS4 KiB plus barriers/statistics/alignment
- Registers: one dK D512 and two dV D256 FP32 fragment families remain live;
  compile gate rejects spills or a register allocation beyond the accepted
  launch/resource policy
- Expected residency: still one CTA per SM unless compiler-reported resources
  prove otherwise; no occupancy claim is made from the SMEM estimate alone
- No new TMEM or cluster resource

## 10. Correctness plan

- Fake compile the candidate before real launch
- Fixed S128 frozen O/LSE and separate dQ/dK/dV comparison, three repeats,
  nondefault stream, and structured GQA/slab ownership
- Fixed boundary ladder and packed tiny/mixed/reversed/empty matrices inherited
  from EXP-0035
- Candidate-off fallback must retain accepted object hashes and results
- Run memcheck, synccheck, and racecheck on fixed and packed tail cases
- Verify repeated-launch/cache-key reuse and guarded memory

## 11. Benchmark plan

- Baseline: accepted EXP-0035 exact FA4 path
- First gate: global S8K backward hot, 10 warmups, 30 CUDA-event repetitions
- Accept first gate only for at least 10% lower median with non-overlapping IQR
- If accepted: S8K combined hot/cold, then S64K reduced screen and 10/30
  confirmation
- Record unlocked clocks unless lock permission becomes available
- Use Nsight Systems launch durations and SASS/resource inspection. Nsight
  Compute counters are explicitly unavailable on this pod.

## 12. Assumptions and risks

- Verified: dP and dV are additive across the exact D256 halves before their
  respective FP32 accumulation/conversion boundaries; dK depends on full dS
  and must be formed only once after both dP halves
- Unverified: the two dV fragment families plus dK fit without spill and the
  extra dO replay costs less than the removed full QK/dK work
- Primary correctness risk: early dO release or Q-scratch alias before every
  WGMMA/TMA consumer is complete
- Primary performance risk: register pressure or the third dO generation
  offsets the arithmetic and launch reduction
- Rollback: candidate is opt-in; candidate-off uses the accepted two dKV slab
  launches and byte-identical EXP-0035 dQ stream kernels

## Review decision

ACCEPTED. The implementation stayed within this state machine and passed the
fixed/packed reference, memory, memcheck, synccheck, racecheck, SASS/resource,
S8K hot/cold, default/fallback, and full S64K gates recorded in EXP-0037.
Future changes to barrier participants, Q alias lifetime, or generation order
require a new predeclared experiment.
