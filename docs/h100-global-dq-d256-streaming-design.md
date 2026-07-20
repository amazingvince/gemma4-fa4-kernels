# H100 global dQ D256-streaming design brief

This brief records the mandatory human-review gate and the implementation that
subsequently passed EXP-0035's acceptance criteria.

## 1. Environment and version

- Date: 2026-07-20
- `nvidia-cutlass-dsl`: 4.6.0.dev0
- CUTLASS: v4.6.0, `e6233cbac5d7c7a865c19c91cd684ceece19513c`
- FlashAttention: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6` plus accepted patch
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CUDA / driver: 12.8.93 / 580.126.09
- Python / PyTorch: 3.12.3 / 2.8.0+cu128
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, target SM90a
- Official example: CUTLASS v4.6.0 Hopper `fmha.py`; production starting
  point is pinned FA4 `flash_attn/cute/flash_bwd_sm90.py`
- Verified APIs: TMA pipeline create/acquire/wait/release, SM90 WGMMA
  `gemm_w_idx` accumulation, shared-memory layout construction, and existing
  dQ chunk epilogue at the pinned revisions

## 2. Operation contract

- Operation: dQ-only portion of exact global Gemma 4 attention backward
- Formula: `S=QK^T`, `P=exp(S-LSE)`, `dP=dO V^T`,
  `dS=P*(dP-row_sum(dO*O))`, `dQ=dS K`
- Inputs: BSHD or THD BF16 Q/K/V/dO; Q has 32 heads, K/V 4 heads, D=512
- Output: one contiguous D256 FP32 dQ accumulation chunk per specialization,
  later converted to BF16; separate accepted dKV kernels remain unchanged
- Accumulation: FP32 S, dP, and dQ
- Mask: global causal only; scale exactly 1.0
- Empty/tail behavior: unchanged accepted fixed and packed rules; no all-empty
  physical workload
- Aliasing: K and V remain distinct; no in-place operand writes
- Determinism: unchanged accepted nondeterministic FP32 GQA accumulation policy

## 3. Shape and layout regime

- Dynamic: valid accepted sequence totals/lengths and outer strides
- Static candidate: M64xN32, Dqk=Dv=512, stream chunk D256, two MMA
  warpgroups, one producer warp, one pipeline stage reused for two generations
- Alignment: unchanged 16-byte public alignment; D256 offsets are 512-byte
  aligned for BF16
- Specializations: `dq_lo_stream` and `dq_hi_stream`; accepted `dkv` key and
  object are byte-for-byte unchanged
- Rejection: any non-SM90, non-BF16, non-global, non-d512, non-GQA8, or
  block-sparse request stays on existing validation/routing

## 4. Target and kernel family

- Architecture: Hopper SM90a
- Family: accepted nonpersistent warp-specialized SM90 backward, dQ-only
  structural variant
- Compute: BF16 WGMMA with FP32 accumulation
- Portable fallback: accepted dQ slab composition
- Unsupported FP8 and SM103 atom paths remain guarded and untouched

## 5. Tile and ownership

- CTA: one N32 K/V tile and successive M64 Q tiles
- Roles: one 128-thread non-MMA warpgroup contains the elected producer and
  store warps; two 128-thread MMA warpgroups consume
- Each MMA warpgroup retains its accepted row partition. Both consume both D256
  generations and own disjoint dQ rows.
- K512 and V512 remain resident for the whole N tile. Q256 and dO256 slots are
  reused only after both warpgroups drain the corresponding WGMMA generation.

| Tensor | Logical shape | SMEM tile | Owner | Consumer |
|---|---|---:|---|---|
| Q | M x 512 | 64 x 256 | producer generation 0/1 | QK WGMMA |
| K | N x 512 | 32 x 512 | first generation of N tile | QK and dQ WGMMA |
| V | N x 512 | 32 x 512 | first generation of N tile | dP WGMMA |
| dO | M x 512 | 64 x 256 | producer generation 0/1 | dP WGMMA |
| LSE/dPsum | M | 64 FP32 each | producer; duplicated per generation | pointwise |
| dS | M x N | 64 x 32 BF16 | both MMA WGs | dQ WGMMA |
| dQaccum | M x 256 | 64 x 256 FP32 | one WG row partition | store warp |

## 6. Data movement

| Edge | Primitive | Alignment/tail | Completion |
|---|---|---|---|
| Q/dO GMEM to SMEM | TMA D256 tile | unchanged public alignment; M predication | generation mbarrier |
| K/V GMEM to SMEM | TMA full D512 tile | N predication | extra bytes on first generation |
| SMEM to accumulators | SM90 WGMMA | exact D256 halves | WGMMA group wait |
| dS registers to SMEM | accepted R2S copy | exact M64xN32 | PdS named barrier |
| dQ accum to GMEM scratch | accepted R2S/TMA path | accepted M tail | dQ full/empty barriers |

The Q/dO TMA atoms use a D256 tiler over the original D512 tensor. K and V
keep full-D layouts. Each WGMMA partitions the matching K or V D256 view.

## 7. Pipeline state machine

Q and dO keep separate one-stage mbarrier storage but advance in lockstep.
For every processed M block, producer and consumers advance exactly twice.
The phase toggle is the reuse proof for the single Q/dO slots.

The accepted pipeline consumer group remains eight warps / 256 threads. The
Q and dO transaction counts become one D256 tile plus one 64-row FP32 statistic
tile per generation; the first generation of the first M block adds the full
K512 and V512 transaction bytes. The PdS named barrier remains 256 threads.
Each existing per-warpgroup dQ empty/full handoff remains 160 participants:
128 owning MMA threads plus the 32-thread store warp.

| Transition | Agent | Generation | Event | Matching transition |
|---|---|---|---|---|
| acquire | producer warp | g0 | acquire Q/dO stage; add K/V bytes only for first M block | prior g1 release |
| issue | producer warp | g0 | load Q0, dO0, stats; first M block also K512/V512 | barrier arrival |
| wait | both MMA WGs | g0 | wait Q and dO barriers | producer arrival |
| consume | both MMA WGs | g0 | zero-init S/dP, issue Q0K0 and dO0V0 | WGMMA drain |
| release | both MMA WGs | g0 | release Q/dO only after `wait_group(0)` | next g1 acquire |
| acquire/issue | producer warp | g1 | load Q1, dO1, duplicated stats; no K/V reload | g0 release |
| wait/consume | both MMA WGs | g1 | accumulate Q1K1 and dO1V1 into existing FP32 fragments | WGMMA drain |
| release | both MMA WGs | g1 | release Q/dO after `wait_group(0)` | next block g0 acquire |
| pointwise/dQ | both MMA WGs | after g1 | form P and full dS once; run accepted dQ/PdS handoff | existing barriers |

K/V are not overwritten for a new N tile until the final g1 of the prior tile
has drained and released. Zero-, one-, and multi-M-block paths must preserve
the two-advance invariant. The candidate is not wired through block-sparse
producer/consumer helpers.

At the adapter boundary, the two accepted D256 preprocessing results remain
available to the byte-unchanged dKV slab launches. Their FP32 dPsum rows are
added once to form the full-D dPsum supplied to both stream dQ variants. This
extra elementwise add is included in performance timing and memory preflight.

## 8. Predication and neutral values

- M/N coordinates, causal mask, and all-masked behavior are unchanged
- D has no tail: two exact D256 chunks cover D512
- Invalid M/N loads retain the accepted TMA zero-fill/predication behavior
- Stores retain the accepted dQ chunk predicate and conversion path

## 9. Resource budget

- Threads: 384 (one 128-thread non-MMA warpgroup plus two 128-thread MMA
  warpgroups)
- Dynamic SMEM estimate:
  Q32 + K32 + V32 + dO32 + dS4 + dQacc64 KiB, plus barriers/statistics/alignment
  = approximately 201,728 bytes
- H100 opt-in dynamic-SMEM ceiling: 228 KiB; estimated margin about 31 KiB
- Registers: retain accepted 240 registers per MMA WG pending compile evidence
- Expected residency: one CTA per SM, unchanged
- Compile gate: reject overflow, spill growth, missing WGMMA, or more than the
  two bounded dQ stream keys

## 10. Correctness plan

- Fake compile both stream variants before real launch
- S128 frozen O/LSE and separate dQ/dK/dV comparison, three repeats,
  nondefault stream
- Fixed and packed boundary/tail cases inherited from the accepted route
- Exact accepted dKV object hashes/resources must remain unchanged
- Run memcheck, synccheck, and racecheck on fixed S128 and packed tail cases
- Verify repeated-launch and cache-key reuse

## 11. Benchmark plan

- Baseline: accepted patch and EXP-0029/0034 FA4 rows
- First gate: global S8K backward hot, 10 warmups, 30 CUDA-event repetitions
- Accept first gate only for at least 10% lower median with non-overlapping IQR
- If accepted: S8K combined hot/cold, then S64K reduced screen and full
  confirmation
- Record unlocked clocks unless lock permission is available; no cross-GPU
  claim
- Inspect Nsight/NCU and SASS only after correctness/sanitizer gates pass

## 12. Assumptions and risks

- Verified: D512 is exactly two D256 chunks; K/V are distinct; dP and score
  partials are algebraically additive before the nonlinear pointwise step
- Unverified: pinned `gemm_w_idx` can retain/accumulate the score and dP
  fragments across a pipeline phase transition without register spill
- Primary correctness risk: releasing a Q/dO slot before all asynchronous
  WGMMA consumers finish
- Primary performance risk: the forced WGMMA drain between generations removes
  overlap and offsets the arithmetic/launch reduction
- Rollback: restore the accepted patch byte-for-byte; baseline adapter and dKV
  kernels are unchanged

## Human review decision

APPROVED 2026-07-20. The user approved checking the exact-BF16 d512 candidate
after reviewing the ownership, two-generation pipeline, barrier counts, and
early-release proof. This approval does not extend to EXP-0033's FP8 path.

## Acceptance outcome

ACCEPTED 2026-07-20 on the declared H100. Fixed and packed O/LSE/dQ/dK/dV,
ownership, isolation, memory, memcheck, synccheck, and racecheck gates pass.
The final consumer lifecycle adds one two-warpgroup rendezvous after the
generation-1 statistic loads and WGMMA drain, before the release arrivals that
permit the next producer overwrite. The two dQ kernels compile to 168
registers, zero local bytes, 66 HGMMA instructions, and 201,728 dynamic shared
bytes each. The retained dKV object is byte-identical.

Post-fix unlocked-clock medians are 57.683 ms for S8K backward and 3435.331 ms
for S64K backward, 40.5% and 43.1% below the accepted ruler. Combined medians
improve by 38.4% and 40.6%. The path is enabled by default; setting
`FLASH_ATTENTION_GEMMA4_EXPERIMENT_DQ_D256_STREAM=0` restores the retained
slab dQ fallback. No FP8, local-attention, B300, or universal speed claim is
made.
