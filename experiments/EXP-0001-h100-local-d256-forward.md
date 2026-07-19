# EXP-0001: H100 local d256 forward contract adapter

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256-fwd
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / 2.8.0+cu128
- QuACK runtime helpers: `quack-kernels==0.5.3`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `1e1767ac7a87ba5503a97aa38bbefcccef5fa88e9793f330bdd9b3658fcceb75`
- Strict environment artifact: `agent_space/h100-check-precommit.json`
  (SHA256
  `9fe9febc9333e481c810caa8736cfb5ca7c28fe1439805060caf5a5c935497a4`)
- Exemplar path and revision: pinned
  `flash_attn/cute/flash_fwd_sm90.py` through `interface.py`

## Invariant changed

No math invariant changes. Add one contract adapter that fixes the prepared-Q/K/V
layout, exact scale `1.0`, inclusive FA window `(1023, 0)`, distinct K/V
storage, SM90 target, and `(O, FP32 LSE)` return contract.

## Hypothesis

The pinned SM90 forward path at BF16 d256 matches the FP32-accumulating locked
reference for O within `atol=0.03125, rtol=0.02` and LSE within
`atol=0.125, rtol=0` across exact 32Q/16KV geometry, GQA ratios 1/2/4/8, tile
tails, and positions around the W1024 boundary.

## Single change

Introduce only the local text forward adapter and its exact GPU correctness
matrix. Do not add backward, multimodal masking, d512, or tuning in this
experiment.

## Correctness evidence

- [x] locked contract and pinned Transformers oracle
- [x] targeted O/LSE reference matrix including boundary/tail cases
- [x] GQA ratios 1, 2, 4, and 8
- [x] invalid layout/dtype/alias/fallback cases
- [x] repeated-run and nondefault-stream check
- [ ] coordinate-coded and adversarial-value matrix (deferred hardening)

The H100 commands and complete results were:

```bash
env FLASH_ATTENTION_FAKE_TENSOR=1 \
  pytest -q tests/test_h100_fa4.py -k local_d256_forward_fake_compile
# 1 passed, 36 deselected in 2.95s

pytest -q tests/test_h100_fa4.py -k 'local and not fake_compile'
# 18 passed, 19 deselected in 3.31s
```

The real run contains three CPU adapter/validation cases plus 15 H100
correctness cases. It covers sequence lengths `1, 63, 64, 65, 127, 128, 129,
1023, 1024, 1025`, GQA ratios `1, 2, 4, 8` at sequence length 33, and an
exact-repeat comparison on a nondefault CUDA stream. The declared tolerances
were not changed: O uses `atol=0.03125, rtol=0.02`; FP32 LSE uses
`atol=0.125, rtol=0`.

## Synchronization and generated code

- [ ] memcheck (not run; the adapter introduces no memory protocol)
- [ ] synccheck (not run; the adapter introduces no barrier protocol)
- [ ] racecheck (not run; the adapter introduces no protocol)
- [ ] IR/PTX/SASS observation
- [ ] registers/spills/SMEM recorded

This adapter exercises the unchanged pinned upstream SM90 protocol. The JIT
cache fingerprint is
`9cfb85b3e5e5cffdb934bf020adfba12751162bb310928a3ccda326ab89ea6d7`.
Sequence length is runtime data and did not change the specialization. The
four GQA compile keys, reproduced in an empty experiment cache, are:

| GQA ratio | compile key | cached object SHA256 |
|---:|---|---|
| 1 | `d624b90a0b0cfb11646c4bdc02ebcc3a91a3b49816cbbbfdd132edf06eae6b5f` | `12becb65efced3cac1e1da867f43436f7dd24c8487281093daadbdbbd96d5a81` |
| 2 | `2af8062164bb2f588565c69f86618b21fabb1347d52dbd40ede2b1e1a3082cfe` | `8058e7f34e1b803d7e525cf7cdd7adddaad5f1777d0c761681d7b42381b6ea89` |
| 4 | `10d2d8963d35acffcf6ab8f4b0360cb8ef987ff21f2619f021e253eb7eab684c` | `0e22e6c13baf576494a5610bcf25c602c98d29383405eeb4ed27db1cfd7150a3` |
| 8 | `3c7ed6370da787d7dbc3b37bb90ba50ef504ffc9b504bbcce0db6bcb72876efe` | `a4715da80cab44e30813df2d539c80919d2829db3bc88a0f825f5bde51a1fde4` |

The cache artifacts are x86-64 relocatable host wrappers. `cuobjdump` reports
that they contain no device code, so they do **not** establish PTX/SASS,
register, spill, or shared-memory evidence. Those items stay open for a device
artifact capture rather than being inferred from a passing launch.

## Measurement

No performance measurement is authorized in this experiment.

## Decision

**ACCEPT**, scoped only to fixed-length, contiguous-BSHD, BF16 local text
forward on SM90 with d256 and the exact adapter contract above. The complete
matrix passed without a tolerance change. This does not accept varlen,
multimodal masking, backward, d512, sanitizer cleanliness, generated-code
resources, coordinate-coded/adversarial-value coverage, or performance.

## Record

The H100 environment record is appended to `experiments/results.jsonl`
against immutable source revision
`5b9bfab072e8cc28a7e92c9e956608db591b246c`:

```bash
python scripts/record_result.py EXP-0001 \
  --kernel h100-local-d256-forward --arch sm_90 --decision accept \
  --git-sha 5b9bfab072e8cc28a7e92c9e956608db591b246c \
  --hypothesis 'pinned SM90 d256 forward satisfies the locked O/LSE envelope'
```
