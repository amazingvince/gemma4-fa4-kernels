# EXP-0027: H100 local compiled StaticSlidingWindow decode

- Date / author: 2026-07-20 / Codex
- Status: **PREDECLARED — no candidate result yet**
- Kernel family: pinned Transformers local-d256 one-token decode over the
  retained native packed-varlen local text forward
- Architecture: sm_90
- Starting revision: `eec1f51baf447c1e1ef89eafca0eee0307824c6b`
- Model-contract lock SHA256:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Transformers patch SHA256:
  `ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79`
- H100 FA4 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`

## Prior boundary

EXP-0016 accepts pinned eager layer-0 StaticSlidingWindow active-prefix
semantics through the K1024 boundary and first rollover. EXP-0026 accepts the
explicit full-storage-view compiled-cache ABI only for pinned global layer 5,
whose `StaticLayer` is a monotonic prefix and whose tensor counter remains its
absolute length.

Neither result proves compiled local cache mutation. A pinned
`StaticSlidingWindowLayer` has a physical 1024-token ring represented in
oldest-to-newest order. Its Python `cumulative_length_int` remains the
absolute token count, while its CUDA scalar `cumulative_length` advances only
through 1024 and then remains saturated. At and after absolute position 1024,
one-token update rolls K/V left, writes the new token at slot 1023, preserves
the CUDA counter at 1024, and increments only the Python absolute count.

## Invariant changed

Extend the separately named cache-decode facade only to the exact pinned
local layer 0, B1/Q1 BF16 inference/no-grad, text-only
`StaticSlidingWindowLayer` with physical capacity 1024 after eager prefill.
The compiled tensor-only frame receives non-static full-storage views of the
upstream-marked K/V/saturated-counter roots plus explicit absolute
`position_ids`; no cache object, Python counter, mask object, or module state
may enter FX.

The opaque local custom op must reproduce the pinned update protocol exactly:

- absolute positions below 1024 update that one physical slot and increment
  the tensor counter;
- absolute position 1023 is the no-roll boundary update;
- positions 1024 and later roll K/V left, write slot 1023, and leave the
  tensor counter at 1024;
- only after a successful compiled call may the eager facade advance the
  pinned layer's `cumulative_length_int` by one.

The attention operand is the exact active prefix before saturation and the
whole ordered 1024-token window after saturation. It must use the retained
native packed lower-right local text route with CUDA INT32 cumulative arrays,
scale 1.0, 32Q/16KV GQA-2, d256, distinct K/V, and no metadata.

No vision/document cache, B>1, Q>1, chunked or compiled prefill, training,
kernel source, tile/stage, dependency, raw/full-model compile, performance,
or B300 behavior changes.

## Hypothesis

An explicit-view local cache custom op keyed by the runtime absolute position
can mirror the pinned saturated-counter and roll-left protocol while keeping
K, V, and the CUDA counter as explicit FX placeholders, producing one graph
per facade and bitwise eager cache/output equality across underfill, K1024
boundary, first K1025 rollover, and repeated rollover under stock eager and
Inductor backends.

Falsification is a second graph for one facade; graph break; cache `get_attr`;
Python absolute-count source in FX; more than one cache op; wrong tensor or
Python counter transition; wrong slot/roll order; eager numerical or cache
byte disagreement; hostile-tail influence; changed address or K/V aliasing;
prepared O/LSE policy failure; unexpected FA4 application class; sanitizer
finding; changed retained native local-varlen object/code/resource/launch
signature; or any widened local-metadata/performance/B300 claim.

## Single change

1. Add one local cache-aware opaque forward op with explicit K/V/counter
   mutation and a shape-only fake registration.
2. Add a pinned local binding/facade that reuses EXP-0025's non-static
   full-storage-view transport, validates the upstream Python and tensor
   counters outside Dynamo, and advances `cumulative_length_int` only after
   successful post-call cache validation.
3. Dispatch the existing public cache-decode constructor by locked layer
   family without changing the accepted global path.
4. Add CPU schema/FakeTensor guards and an H100 discriminator/matrix; do not
   change a CuTe kernel or numerical policy.

## First discriminator

On one pinned layer-0 Inductor facade after eager K1023 prefill, run absolute
positions 1023, 1024, and 1025. Require one backend attempt, zero breaks, one
local cache op, three explicit cache placeholders, zero cache `get_attr`, and
no forbidden Python/module/cache source. Each whole-layer output and complete
K/V/counter state must be bitwise equal to a weight-identical pinned eager
StaticCache twin. Boundary mutation writes only slot 1023; each rollover is
the exact byte-for-byte left shift plus new slot 1023. Root/view addresses
stay stable, `cumulative_length` transitions 1023 to 1024 and remains 1024,
and `cumulative_length_int` advances 1023 through 1026.

Stop and record a rejection before the wider matrix if this discriminator
fails.

## Correctness gates

- [ ] local custom-op schema declares only K/V/tensor-counter mutation and
      FakeTensor returns fresh BF16 output plus FP32 `(1,32,1)` LSE
- [ ] first discriminator passes one graph, zero breaks, explicit cache
      placeholders, no cache `get_attr`, and no Python/cache source in FX
- [ ] K1024 boundary, first K1025 roll, and two repeated rolls are bitwise
      equal to pinned eager whole-layer output and complete cache state
- [ ] independent underfill K33/K34 cases pass eager and Inductor, with a
      hostile unwritten tail unable to affect prepared O/LSE or projected O
- [ ] prepared Q/K/V are exact; local packed O and FP32 LSE pass the frozen
      reference policies at underfill, boundary, and rollover
- [ ] tensor counter saturates at 1024, Python absolute count advances only
      after success, and absolute position/kv-offset ownership matches eager
- [ ] root/view addresses remain stable, K/V remain distinct, and mutation
      versions/bytes follow the exact underfill versus roll protocol
- [ ] different seeds, reverse case order, and one nondefault CUDA stream add
      no graph, Inductor-cache, scheduler, or application class
- [ ] malformed position/counter combinations, B2, Q2, metadata, active grad,
      lazy/offloaded/foreign cache, reset, root rebind, forged view, wrong
      layer class/capacity, and unsupported kwargs reject before compiled entry
      with byte-identical state
- [ ] existing EXP-0016 eager cache, EXP-0023 no-cache, EXP-0025 view ABI,
      EXP-0026 global cache, raw-compile negative, fixed, packed-varlen, mask,
      and backward matrices remain unchanged

## Synchronization and generated code

- [ ] unfiltered memcheck passes one repeated local rollover call
- [ ] project-kernel-filtered synccheck passes the same call
- [ ] project-kernel-filtered racecheck passes the same call
- [ ] native local-varlen application object remains bitwise equal to the
      EXP-0016 cache replay baseline; retained PTX/cubin/SASS/resources and
      launch geometry remain equal to the accepted local-varlen predecessor
- [ ] Dynamo graph, Inductor files, root/view mutation state, Python/tensor
      counters, and FA4 application keys are separately inventoried

## Measurement

Correctness-only. No timing, speedup, compiled-prefill, whole-model,
raw-`torch.compile(layer)`, cached vision/document metadata, training, B300,
or cross-architecture claim is authorized.

## Decision

Pending. Accept only the exact local layer-0 envelope above and stop before
other layers, multimodal cache, compiled prefill, or full-model work if any
graph, counter, mutation, reference, sanitizer, or codegen gate fails.

## Record

```bash
python scripts/record_result.py EXP-0027 \
  --kernel h100-local-static-sliding-cache \
  --arch sm_90 --decision <accept|reject|refine> \
  --hypothesis '<exact hypothesis above>' --profile <strict-h100-json>
```
