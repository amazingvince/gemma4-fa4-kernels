# H100 pinned-Transformers integration

This document defines the eager framework boundary accepted by EXP-0011,
extended through K2048 fixed/composed training by EXP-0012, and promoted to a
native THD/cu-seqlens packed-global backward by EXP-0013. EXP-0014 extends only
that native packed backward route through K262144 under signed-INT32 and
guarded-HBM admission; fixed BSHD and the exact composer remain capped at
S/K2048. EXP-0015 admits mixed packed plateaus on the accepted local/global
routes when aggregate Q/K totals and exact maxima remain positive. EXP-0016
adds eager B1 text-only StaticCache active-prefix prefill/decode with no active
backward. It does not widen any model invariant in `docs/model-contract.md`,
claim a fused global d512 kernel, or cover B300. EXP-0018 tested a no-cache
compiler refinement whose patch provenance is pinned below. Its 16-case
cache-rejection matrix passed, but its positive fullgraph matrix was rejected
at local default-Inductor S1023 by the frozen BF16 numerical gate. No
framework compiler acceptance is claimed. EXP-0019 predeclares a diagnostic
localization and a tensor-explicit whole-layer opaque refinement; it is not an
accepted path unless its bitwise pinned-eager and retained-reference gates
pass. The S1023 diagnostic confirms non-bitwise Inductor Q/K/V preparation
while identical prepared FA4 O/LSE and the isolated output projection remain
bitwise; whole-layer implementation is still pending.

## Pinned boundary

- Transformers base revision:
  `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Reviewed two-file patch:
  `patches/transformers/0001-gemma4-forward-vision-block-ids.patch`
- Patch SHA256:
  `c812937e5a554c1887c2c16a0808f24437cb8b60b561e9fd5eacaa13fb277780`
- Registered attention and mask name: `gemma4_fa4_h100`

Register the pair before constructing or loading the model:

```python
from gemma4_fa4 import BACKEND_NAME, register_gemma4_fa4_h100

assert register_gemma4_fa4_h100() == BACKEND_NAME
config._attn_implementation = BACKEND_NAME
```

Registration is idempotent for the project callables and rejects a name
collision. It does not overwrite Transformers' generic `flash_attention_4`
entries.

The revised pinned patch also gives the exact plain causal and sliding mask
builders private identity capabilities. Only the selected project callback
receives the exact cache object, the matching family capability, and its own
recipient identity. During compilation a non-null cache is rejected at this
mask boundary, before `Gemma4TextAttention` can call `Cache.update`; eager
StaticCache handling remains on the accepted EXP-0016 path. These are
retained fail-closed semantics from rejected EXP-0018; they do not promote the
no-cache positive compiler route, compiled caches, or compiled training to an
accepted framework boundary.

## Prepared tensors and outputs

Pinned `Gemma4TextAttention` supplies distinct prepared Q/K/V tensors in BHSD.
The adapter validates BF16, the locked layer-index geometry, scale 1.0, zero
dropout, distinct K/V storage, positive nonoverlapping strides, unit D stride,
outer strides divisible by eight BF16 elements, and 16-byte base alignment.
Its canonical BHSD-to-BSHD transpose shares the original allocation; fixed
paths do not make an unconditional contiguous copy.

Packing copies are localized to valid rows when padding or packed positions
require a gather. The accepted global training route uses native packed THD
coordinates. A zero-Q prefix is created only if its guarded HBM preflight
raises `GlobalBackwardBudgetExceeded` while every active-query K segment is at
most 2048 and selects the retained exact composer. For an active-query
K>2048, that exception propagates before forward. V512 is materialized as two
V256 slabs for both routes. Padded rows scatter back as exact zero output and
`-inf` FP32 LSE.

EXP-0015 preserves every batch-row boundary, including zero-length plateaus.
Paired-empty and query-empty/key-nonempty rows own no packed O/LSE entries and
receive exact-zero gradients over their K/V slices. Decreasing cumulative
arrays, `Sq>Sk`, wrong maxima, and all-empty physical workloads are rejected
before backend dispatch.

The Transformers attention interface returns `(output, None)`: its second
return is attention weights, not LSE. The project-level
`gemma4_fa4_prepared` result exposes `output`, FP32 `lse` when the selected FA4
route provides it, and a route string. The interface also stores that route on
the attention module as `_gemma4_fa4_last_path` for diagnostics.

## Eager StaticCache active-prefix envelope

EXP-0016 accepts an underfilled pinned `StaticCache` only for eager B1
text-only calls with no active backward. The mask adapter snapshots the query
offset before the cache update mutates its counter, and dispatch exposes the
offset-proven contiguous active K prefix through zero-copy K/V views. Full and
rolled local caches retain their prior routes. Framework tracing and compiled
StaticCache remain outside this envelope.

## Mask and metadata contract

`gemma4_fa4_mask` preserves the pinned mask callable, 2D/4D mask tensor, and
Q/K offsets in a `Gemma4MaskPlan`. Before choosing a native FA4 route, the
adapter parses only the boolean combinators from the pinned
`transformers.masking_utils` revision, normalizes the expression, and compares
it with the exact locked layer predicate. Captured vision and packed-sequence
metadata must also equal the authoritative runtime tensors. This structural
fingerprint prevents a merely similar or stricter callable from entering a
native route.

The Transformers patch makes vision IDs authoritative at the source:

1. an explicit `vision_block_ids` tensor wins;
2. otherwise IDs may be derived once from multimodal token types;
3. the same tensor constructs the local mask and reaches every attention
   layer;
4. prebuilt generation-mask mappings retain the tensor instead of dropping
   it;
5. global masks remain causal and never gain vision bidirectionality.

Batch-1 `position_ids`, vision/document IDs, and 2D padding/static masks may be
broadcast across the prepared batch. Explicit cumulative arrays must preserve
every batch-row boundary. Shorter 2D static masks are extended with masked
positions to the requested K extent. Holey/static layouts, arbitrary callables,
and floating additive 4D score masks keep their exact plan for fallback; they
are never silently reinterpreted as the native Gemma predicate. Boolean or
integer 4D masks are ambiguous at the additive-score interface and fail
closed. A document ID that reappears after another document run also fails
closed instead of being split into semantically incorrect independent runs.

## Eager route matrix

| Contract | Route | Gradient support |
|---|---|---|
| Local B1, equal S<=1025 | `fa4_local_fixed` | forward and backward |
| Local padded/packed/reset-position/document/lower-right | `fa4_local_varlen` | forward and backward inside EXP-0009/0010 bounds |
| Global B1, equal S<=2048 | `fa4_global_fixed` | backward-capable; no-grad fixed route through S1024 |
| Global training outside the fixed row, including equal S>2048, batched/padded/packed/lower-right; every K segment <=262144 | `fa4_global_varlen_native` | native THD/cu-seqlens backward; guarded HBM admission may select `fa4_global_varlen_composed_budget_fallback` only when every active-query K<=2048 |
| Global fixed or lower-right, K>1024 through K262144 | `fa4_global_forward_only` | no grad only |
| Global packed/varlen with any K>1024 through K262144 | `fa4_global_varlen_forward_only` | no grad only |
| Underfilled eager B1 text-only StaticCache with one offset-proven active prefix | retained local/global route after zero-copy K/V prefix exposure | no active backward |
| Exact non-native mask/layout | `flex_attention` | inference only |

The long global routes require per-segment `0 <= Sq <= Sk <= 262144`, positive
aggregate Q/K totals and exact maxima, and preserve lower-right causality.
No-grad calls preflight their conservative composed output/LSE estimate.
Training calls through K262144 preflight the native packed workspace plus
retained forward state. Only the dedicated budget exception with every
active-query K segment at most 2048 may select the exact per-segment composer.
An active-query K>2048 budget rejection propagates before forward and cannot
select the composer or FlexAttention. Validation, contract, assertion, and
backend runtime failures also propagate.

FlexAttention is a correctness fallback for eager inference only. A diagnostic
D512 B2/S5 backward candidate produced a non-finite dQ after a larger default
tile had already exceeded H100 shared memory. The production adapter therefore
rejects every gradient-capable fallback before launch. Disabling fallback turns
every unsupported request into an explicit `UnsupportedH100Path`.

Framework FakeTensor and `torch.compile` tracing also fail closed. Lower-level
local/global kernel-wrapper FakeTensor compilation passes mixed plateaus in
EXP-0015, but the no-cache framework path still needs a separately designed
opaque ABI and compile-key audit. Compiled StaticCache follows only after that
boundary; eager or wrapper-level success is not evidence for compiled model
execution.

## Recorded H100 evidence

```bash
python scripts/probe_h100_transformers_integration.py --case all
python scripts/probe_h100_transformers_integration.py \
  --case global-forward-only-max-context
python scripts/probe_h100_transformers_integration.py \
  --case global-packed-empty-row --seed 11011
```

The first command passes eighteen cases covering zero-copy fixed views, local
padding and lower-right packing, native global B2/S5 training, contiguous
document splitting with rebuilt cumulative arrays, Q33/K2049 lower-right and
mixed packed K=[2049,4097] backward, odd-padded noncontiguous dO/dLSE views,
global fixed/varlen forward-only K2048, authoritative mask transport, the
registered backend, and actual pinned `Gemma4TextAttention` local and global
forward/backward execution. The EXP-0015 case adds a fully padded row beside a
nonempty row, selects native THD, restores zero O / `-inf` LSE, and proves
exact-zero empty-row gradients plus hostile-row isolation. The long global
module case selects native THD at S2049. EXP-0016 adds five actual pinned-layer
StaticCache cases: local S32, the S1023 boundary, first rollover, global
S32/q1-K33, and global S1024/q1-K1025.
The second is an exact Q1/K262144 zero-score sentinel:
output is `64/262144` and LSE is `log(262144)`.

EXP-0013 memcheck, synccheck, and racecheck pass for the native mixed packed
Q=[33,65], K=[1025,2048] case and the 33 one-token-segment scheduler case;
memcheck also passes for the framework document-split route and the direct
native-THD odd-stride dO/dLSE case housed in the integration probe. Earlier
composed/fixed sanitizer evidence remains retained.
No performance result, full-checkpoint run, compiled-model result, or B300
result is claimed.

The fresh EXP-0013 cache contains 28 objects, 16 unique contents, and 1,956,400
bytes. Fixed BSHD retains its nine application keys and byte-identical
EXP-0012 main objects. Native THD adds exactly nine application keys: three
scheduler classes times dKV, dQ-low, and dQ-high. Runtime totals, segment order,
cumulative values, logical batch, legal strides, and K1025/K2048 replays add no
specialization. Exact keys and hashes are recorded in EXP-0013. The
schema-valid result names implementation revision
`87ff75b1b40b55149ec5beea7480ed9ac14c9146`.

EXP-0014 replays K2049, K4097, S32768, and K262144 without adding a scheduler
or application-key class and retains the EXP-0013 generated main-object
contents. Its changed patch source fingerprint intentionally uses a fresh
cold-cache namespace. Fixed BSHD and composer evidence remains bounded by
S/K2048; the long acceptance applies only to resource-admissible native THD.
EXP-0015 replays leading, middle, and trailing plateaus in all SS/SM/MM native
scheduler classes without adding an object or application key. The global
inventory remains 28 objects, 16 unique contents, 1,956,400 bytes, and 18
application keys; native main-object contents remain byte-identical to
EXP-0014. EXP-0016 replays logical Q1/K33 under physical K65/K129 global and
local StaticCache layouts without adding an application class or changing a
retained main object. Its eager scope remains B1 text-only with no active
backward. No performance or framework-compiled claim is made.
