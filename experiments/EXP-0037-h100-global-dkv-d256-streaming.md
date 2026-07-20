# EXP-0037: H100 global dKV D256 streaming

- Date / author: 2026-07-20 / Codex
- Kernel family: global-d512-dk/global-d512-dv
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Exemplar: pinned FA4 `flash_attn/cute/flash_bwd_sm90.py`
- Design brief: `docs/h100-global-dkv-d256-streaming-design.md`

## Invariant changed

Replace the two accepted V256-specific dKV main launches with one exact
full-D512 dKV specialization. Q512, K512, and V512 remain resident for each
CTA. dO streams through one D256 slot as low, high, then a low-half replay.
The two dP halves accumulate before the nonlinear dS step, dK is formed once,
and the replay supplies the low dV half after P is available. The released Q
tile is reused only in the epilogue as sequential FP32 dK/dV reduction scratch.

## Hypothesis

Removing one full QK pass and one full dK pass lowers exact global S8K
backward median by at least 10% relative to EXP-0035, with non-overlapping
IQRs and no correctness, sanitizer, spill, cache-key, or S64K regression.

## Single change

Add one `dkv_d256_stream` specialization and adapter route. The accepted two
dKV slab launches remain the rollback path. The two accepted
EXP-0035 dQ stream kernels, forward kernels, masks, scale, dtype, tolerance,
and public ABI do not change.

## Baseline profiling

Fresh unlocked-clock Nsight Systems profiling at S8K measured each accepted
dKV slab launch at about 11.2 ms and each accepted streamed dQ launch at about
16.1 ms. Across the measured backward call, main kernels account for 87.3%
of GPU kernel time; preprocessing, postprocessing, and the full-D dPsum add
are individually sub-millisecond. Nsight Compute counter collection is
unavailable on this pod (`ERR_NVGPUCTRPERM`), so no unavailable counter is
used to justify the candidate.

## Correctness evidence

- [x] locked contract retained; no tolerance change
- [x] fake compile and fixed S1/31/32/33/63/64/65/127/128/129 references
- [x] three repeats, nondefault stream, structured GQA/slab ownership
- [x] packed tiny/mixed/reversed/empty references through K2048 and isolation
- [x] O, FP32 LSE, and separate dQ/dK/dV
- [x] bounded fixed/packed peak memory and unchanged flag-off fallback

## Synchronization and generated code

- [x] fixed and packed memcheck: 0 errors
- [x] fixed and packed synccheck: 0 errors
- [x] fixed and packed racecheck: 0 hazards
- [x] SASS: 96 HGMMA and 63/66 UTMA instructions in fixed/packed objects
- [x] 168 registers, 0 stack/local bytes, 1,024 static and 174,080 dynamic shared bytes
- [x] accepted fixed and packed dQ object contents byte-identical
- [x] one bounded dKV candidate key; Nsight Systems shows three main launches per call

The fixed and packed dQ contents retain the accepted SHA-256 values
`480d1e4865cd05894c960cda5bbf383aca5d1eeb1bf3d541d4e6f24235d36e11`,
`3567a4752574a2bad8becc78341108761dd8eff615557f86b2b88dfe2fb1e627`,
`2c412e47303e50a1275614a612433123bbe8bb36f9f7c8134e2f16b5bcf54041`,
and `4b02829509d2349f9c6b30f9cc0635be6ec7075d7a1cabf592a9f769235ac7e1`.
The candidate dKV SASS contains no `LDL`, `STL`, or `CALL` instructions.
Fixed S128 peak delta was 33,652,736 bytes against a 37,847,040-byte bound;
packed Q33,65/K65,129 was 41,497,088 against 60,875,264 bytes.

## Measurement

- Clock/power: unlocked and labeled unless locking becomes available
- L2: S8K hot first gate, then hot/cold confirmation
- Warmup/repetitions: 10/30 CUDA events, median/p25/p75/IQR
- Baseline: accepted exact EXP-0035 FA4 composition

| case | baseline median/IQR | candidate median/IQR | delta |
|---|---:|---:|---:|
| global S8K bwd hot | 57.683 / 1.054 ms | 51.442 / 0.272 ms | -10.8%, 1.12x |
| global S8K fwd_bwd hot | 63.507 / 0.158 ms | 57.221 / 0.110 ms | -9.9%, 1.11x |
| global S8K bwd cold | 57.530 / 0.369 ms | 51.514 / 0.267 ms | -10.5%, 1.12x |
| global S8K fwd_bwd cold | 63.056 / 0.088 ms | 57.029 / 0.417 ms | -9.6%, 1.11x |
| global S64K bwd hot | 3435.331 / 2.474 ms | 3053.413 / 1.698 ms | -11.1%, 1.13x |
| global S64K fwd_bwd hot | 3803.218 / 2.149 ms | 3410.774 / 1.376 ms | -10.3%, 1.12x |

The predeclared first gate was also rerun candidate-on/candidate-off in the
same live session: 51.442 / 0.272 ms versus 57.748 / 0.105 ms, a 10.9%
reduction with disjoint IQRs. After promotion, the environment-free default
reran at 51.344 / 0.266 ms; setting
`FLASH_ATTENTION_GEMMA4_EXPERIMENT_DKV_D256_STREAM=0` restored the accepted
slab path at 57.058 ms in a reduced 2/5 fallback screen. All measurements use
unlocked H100 clocks.

Nsight Systems records one 174,080-byte fused dKV main launch followed by the
two unchanged 201,728-byte dQ main launches per backward call. The three
launches each took about 16.1 ms at S8K. Nsight Compute counter collection
remained unavailable with `ERR_NVGPUCTRPERM`; no counter claim is made.

## Decision

ACCEPT

The exact-BF16 fused dKV path passes fake/real compile, fixed and packed
correctness, peak-memory, sanitizer, generated-code, S8K hot/cold, default-on,
fallback, and full S64K gates. It reduces the H100 global d512 backward path
from four to three main launches and becomes the default. Set
`FLASH_ATTENTION_GEMMA4_EXPERIMENT_DKV_D256_STREAM=0` to retain the accepted
two-slab dKV rollback. This is not a single-launch, FP8, B300, local-attention,
or deterministic-gradient result.

## Record

```bash
python scripts/record_result.py EXP-0037 \
  --kernel global-d512-dkv-d256-streaming --arch sm_90 \
  --decision accept --hypothesis '<measured result>' --bench <jsonl>
```

Raw benchmark rows are under `agent_space/remote-h100-exp0037/`. The remote
profiling directory retained its provisional `remote-h100-exp0036` name
because the experiment-number collision was discovered after GPU collection.
