# EXP-0003: H100 local d256 backward compile gate

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256-bwd
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / 2.8.0+cu128
- QuACK runtime helpers: `quack-kernels==0.5.3`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `1e1767ac7a87ba5503a97aa38bbefcccef5fa88e9793f330bdd9b3658fcceb75`
- Exemplar path and revision: pinned
  `flash_attn/cute/flash_bwd_sm90.py`, `interface.py`, and upstream tests

## Invariant changed

No forward invariant changes. Exercise autograd through the accepted local
d256 forward adapter and require three distinct attention-boundary outputs:
dQ, dK, and dV. Never merge dK/dV and never treat K as V.

## Hypothesis

The pinned FA4 SM90 backward dispatcher can compile the exact BF16 local d256,
32Q/16KV, GQA-2, W1024, scale-1.0 specialization and return separate dQ/dK/dV
matching the locked FP32-accumulating reference with `atol=0.125, rtol=0.05`.

That fixed pair is a predeclared smoke-gate envelope, not a fully calibrated
training-accuracy boundary. This experiment can falsify the narrow hypothesis
at its first S=128 seed; it cannot characterize all local-d256 backward
numerics. Any future acceptance experiment must justify its comparison policy
before execution and retain per-gradient max and mean errors.

## Single change

Add only a reproducible backward probe around the accepted local forward call.
Do not change tile sizes, stages, ownership, masking, or tolerances on the first
attempt. The first fake-tensor/compiler error is the gate result.

## Correctness evidence

- [x] locked forward contract and distinct K/V inputs
- [x] exact fake-tensor backward compile
- [x] separate dQ, dK, and dV shapes/dtypes/storage
- [x] first S=128 dQ/dK/dV versus locked reference (dQ/dK rejected)
- [ ] tail/window-boundary matrix
- [ ] repeated-run and nondefault-stream check

If compilation passes, the frozen real matrix is S=`1, 63, 64, 65, 127, 128,
129, 1023, 1024, 1025`. No real launch is authorized after a fake compile
failure.

The fake compile unexpectedly passed despite the upstream source/test warning:

```bash
env FLASH_ATTENTION_FAKE_TENSOR=1 \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0003-local-bwd \
  python scripts/probe_h100_local_backward.py --seqlen 128
# compiled dQ=(1,128,32,256) dK=(1,128,16,256) dV=(1,128,16,256)
# exit 0 in 17.9s
```

The first authorized real comparison failed and stopped the matrix:

```bash
env FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0003-local-bwd \
  python scripts/probe_h100_local_backward.py --seqlen 128 --reference
# exit 1
```

First failure: dQ had 552 mismatches among 1,048,576 elements (0.1%), maximum
absolute difference `0.2890625` versus frozen `atol=0.125`, and maximum
relative difference `1112.0` versus `rtol=0.05`. A diagnostic rerun under the
unchanged envelope collected all three outputs: dQ failed with max/mean
absolute error `0.28857422` / `0.0097916815`; dK failed with `0.375` /
`0.013713409`; dV passed with `0.0625` / `0.00076462259`. No tail,
window-boundary, repeat, stream, global-backward, or multimodal case ran after
this gate failure.

## Synchronization and generated code

- [ ] memcheck
- [ ] synccheck
- [ ] racecheck
- [ ] IR/PTX/SASS observation
- [ ] registers/spills/SMEM recorded

Upstream source already warns against assuming support: its test gate excludes
SM90 backward above d192, and the pinned static search finds no feasible d256
configuration in the existing configuration space. This experiment records an
actual exact-path compiler result rather than converting that source evidence
into a runtime claim.

The JIT fingerprint was
`4b2d00f9c628b1f1c6c10e891a241118d95e8d5f77fb3d40863788e1fb39fbc9`.
Backward compile keys and host-wrapper SHA256 values were:

| phase | compile key | object SHA256 |
|---|---|---|
| preprocess | `0172da051c34b40ceed26db161801c1e67e7bf911f6ece50f526d7fdb52d50c6` | `b225c3c5d1243980039954b2ff40c9af8b07000f0e343112239d980c8de60105` |
| main backward | `821e5a218bdafafc71ff9536e0c13d7697fd833d05f3d990b62316d2068114f3` | `f5cfc1c5759409340c2dc9a4fe5339f3c454390426cd84f496482ed48c7e27bd` |
| postprocess | `442819acf45e096690d2e01020103d40a05dc9ac8cfe581674abec28360bc25e` | `cf5148f524365e0eedc6324a7d4456f528e892bf665f9c24bd6a6bd9412ced7e` |

These `.o` files are JIT host wrappers and are not substituted for device
SASS/resource evidence. Sanitizers and artifact capture were not run after the
numerical gate failed.

## Measurement

No performance measurement is authorized.

## Decision

**REJECT** for the first unchanged-path S=128 smoke gate. Compilation and
execution succeeded, but dQ and dK violated the tolerance fixed before the
run. This is not a claim that every local-d256 backward input is numerically
invalid. Do not loosen the envelope based on this result. Per the ordered M1
gate, stop before global backward, multimodal forward/backward, or benchmarks.
The next session needs one new falsifiable hypothesis about the d256 backward
configuration or accumulation path, not a performance change.

## Record

```bash
python scripts/record_result.py EXP-0003 \
  --kernel h100-local-d256-backward --arch sm_90 --decision reject \
  --git-sha <40-char-source-commit> \
  --hypothesis 'pinned SM90 local d256 backward compiles and matches separate gradients'
```
