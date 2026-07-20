# H100 global dQ full-D single-launch design brief

This is the pre-implementation ownership and synchronization gate for
EXP-0038. It changes only the two accepted EXP-0035 dQ output-half launches.
The EXP-0037 full-D dKV kernel and every model semantic remain unchanged.

## 1. Environment and version

- Date: 2026-07-20
- `nvidia-cutlass-dsl`: 4.6.0.dev0
- CUTLASS: v4.6.0, `e6233cbac5d7c7a865c19c91cd684ceece19513c`
- FlashAttention: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6` plus the accepted
  EXP-0037 project patch
- CUDA / driver: 12.8.93 / 580.126.09
- Python / PyTorch: 3.12.3 / 2.8.0+cu128
- Target: NVIDIA H100 80GB HBM3, SM90a
- Official exemplar: CUTLASS v4.6.0 Hopper `fmha.py`; production exemplar:
  pinned FA4 `flash_attn/cute/flash_bwd_sm90.py`
- Reused APIs: accepted TMA Q/dO lifecycle, WGMMA `gemm_w_idx`, R2S dQ
  epilogue, named barriers, and FP32 bulk-reduce-add

## 2. Operation contract

- Formula: `S=QK^T`, `P=exp(S-LSE)`, `dP=dO V^T`,
  `dS=P*(dP-sum(O*dO))`, `dQ=dS K`
- Inputs: fixed BSHD or packed THD BF16 Q/K/V/dO, 32 Q heads, 4 KV heads,
  D512, distinct prepared K/V
- Output: two contiguous FP32 D256 dQ accumulation slabs produced by one main
  launch, followed by the unchanged one-time BF16 conversion of each slab
- Accumulation: FP32 score, dP, and dQ; scale exactly 1.0; causal global mask
- Empty/tail behavior: unchanged accepted fixed/packed rules
- Determinism: fast nondeterministic accumulation only; deterministic ownership
  is a later milestone
- Finite BF16 inputs are the numerical contract. Non-finite behavior is not
  promoted by this experiment and no device-wide value scan is introduced.

## 3. Shape, layout, and specialization

- Dynamic: accepted sequence totals, segment lengths, and legal outer strides
- Static: M64xN32, Dqk=Dv=512, D256 Q/dO stream slot, two MMA warpgroups,
  one producer warp, one Q/dO stage
- FP32 dQ destination layout: `(slab=2, batch-or-1, q_head,
  padded_q*256)`. Each slab remains contiguous for bulk reduction and existing
  postprocess layouts.
- One `dq_d512_stream` key replaces `dq_lo_stream` and `dq_hi_stream` for each
  existing fixed/packed scheduler class. Runtime lengths and tensor values do
  not enter the key.
- Rollback dispatch restores both byte-accepted EXP-0035 dQ objects.

## 4. Target and kernel family

- Architecture: Hopper SM90a WGMMA
- Family: nonpersistent warp-specialized dQ-only backward
- Compute atom: the accepted BF16 WGMMA/FP32-accumulator atoms
- Portable fallback: EXP-0037's two dQ main launches
- No FP8, B300, local attention, mask, scale, or dKV change

## 5. Tile ownership and dataflow

- One N32 K/V tile is resident while successive M64 Q tiles run.
- Both MMA warpgroups retain their disjoint 32-row ownership.
- Q/dO low and high generations form one full-D score and dP, then one dS.
- After dS is resident, each MMA warpgroup computes its D256 dQ-low fragment,
  hands it to the store warp, drains that handoff, then reuses the same
  registers and 64 KiB shared epilogue arena for dQ-high.
- K512 and dS remain live until both output halves finish. No two full-D dQ
  accumulator families are live together.

| Tensor | SMEM tile | Owner | Consumer |
|---|---:|---|---|
| Q / dO | M64xD256 each | producer, low/high generations | score/dP WGMMA |
| K / V | N32xD512 each | producer once per N tile | score/dP/dQ WGMMA |
| dS | M64xN32 BF16 | score warpgroups | both dQ halves |
| dQ epilogue | M64xD256 FP32 | one row partition per MMA WG | store warp |
| dQ slabs | 2x padded-Q D256 FP32 | store warp bulk reduction | postprocess |

## 6. Pipeline and barrier state machine

The Q/dO two-generation lifecycle is byte-for-byte the accepted EXP-0035
protocol. The only new lifecycle repeats the existing dQ empty/full handoff.
Barrier IDs and participant counts do not change: each
`dQEmptyWG{0,1}`/`dQFullWG{0,1}` event has 160 participants (128 owning MMA
threads plus the 32-thread store warp).

| Transition | Agent | Output half | Event | Reuse proof |
|---|---|---|---|---|
| compute | MMA WG | low | `dS @ K[0:256]` | WGMMA accumulator is low-only |
| acquire | MMA WG + store warp | low | accepted dQ-empty barrier | prior store no longer reads SMEM |
| publish | MMA WG | low | R2S, shared fence, dQ-full arrive | store sees complete low fragment |
| consume | store warp | low | dQ-full wait, bulk reduce-add, commit | low destination is slab 0 |
| release | store warp | high | bulk wait, second dQ-empty arrive | low bulk engine no longer reads SMEM |
| compute | MMA WG | high | `dS @ K[256:512]` | low WGMMA group is drained |
| publish | MMA WG | high | R2S, shared fence, dQ-full arrive | same arena now contains only high |
| consume | store warp | high | dQ-full wait, bulk reduce-add, commit | high destination is slab 1 |
| drain | store warp | after high | bulk wait group 0 | safe scheduler advance/exit |

The store warp acquires both slabs under one scheduler work item. It may not
advance to the next N tile between low and high. The MMA warpgroups may not
overwrite the dQ arena until the store warp's second empty arrival. The
candidate is invalid if compilation merges the two FP32 dQ accumulator live
ranges and spills.

## 7. Predication and resources

- Existing M/N predicates, causal mask, TMA zero fill, and partial-row stores
  remain unchanged. D has exactly two D256 halves and no D tail.
- Threads remain 384. Modeled dynamic SMEM remains 201,728 bytes because the
  same D256 epilogue arena is reused.
- Expected register policy remains 240 requested / 168 reported per MMA path.
- Compile rejection gates: dynamic SMEM above 232,448 bytes, any local-memory
  allocation, spill growth, missing HGMMA/TMA, or more than one candidate key
  per scheduler class.

## 8. Correctness and performance gates

- Fake compile, then fixed S1/31/32/33/63/64/65/127/128/129 references.
- Packed tiny/mixed/reversed/empty-segment references and isolation through
  the accepted K2048 dense-reference envelope.
- O, FP32 LSE, separate dQ/dK/dV, half-zero structured ownership, three
  repeats, nondefault stream, and candidate-off parity.
- Fixed and packed memcheck, synccheck, and racecheck.
- SASS/resource/launch-count/cache inventory inspection.
- S8K backward hot gate: at least 10% lower median than EXP-0037 with
  non-overlapping IQRs. If it passes, run S8K combined hot/cold and S64K
  confirmation; reject any important-regime regression above 3%.

## 9. Assumptions, review, and rollback

- Verified: dS is identical for both output halves; `dS @ concat(K0,K1)` is
  the concatenation of the two D256 products; each bulk destination is
  contiguous and disjoint.
- Primary correctness risk: the second R2S write begins before the low-half
  bulk engine releases the shared arena.
- Primary performance risk: the repeated epilogue handshake and longer kernel
  reduce less time than the removed score/dP/dS launch.
- Rollback: set
  `FLASH_ATTENTION_GEMMA4_EXPERIMENT_DQ_D512_SINGLE_LAUNCH=0` to restore the
  accepted two-launch dQ path without changing dKV.
- Promotion: the H100 gates in EXP-0038 passed on 2026-07-20, so the
  environment-free default is `1`; the explicit `0` rollback remains tested.
- Human review: approved on 2026-07-20 when the user requested implementation
  of the proposed plan containing this sequential D256 accumulator/epilogue
  lifecycle. Any change to barrier IDs, participants, or reuse order requires
  a new review gate.
