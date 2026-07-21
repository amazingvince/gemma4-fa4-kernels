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
framework compiler acceptance is claimed. EXP-0019 confirmed the outer-layer
localization and its local/eager whole-layer path was bitwise, but the
experiment is rejected: Inductor restored live weight metadata at the opaque
consumer, while the explicit detached-weight snapshot workaround produced two
backend captures for the single S1 class. No graph bound or requires-grad
condition was weakened after observation.
EXP-0020 removed the snapshot and admitted only exact live source Parameters
under global no-grad inference. Its local/eager S1 and focused ABI gates pass,
but it is also rejected: stock Inductor first raises
`TensorifyScalarRestartAnalysis` on the retained float-valued config/module
proof and then compiles the identical S1 graph. The frozen gate counts both
backend attempts. No scalar compiler setting, graph bound, or mutation guard
was changed after observation.
EXP-0021 moved all eleven float fields through an explicit CPU-FP64 tensor;
EXP-0022 instead forced them static through PyTorch 2.8's documented comptime
API. Both were rejected at local/Inductor/S1 because the first identical graph
raised `TensorifyScalarRestartAnalysis` and the backend was invoked twice.
EXP-0022's FX graph contained no symbolic-float or scalar-tensor node, which
localizes the remaining restart to Dynamo scalar-source bookkeeping. Both
candidates were removed from maintained code. EXP-0023 accepts a separately
named guarded facade for actual pinned layers 0/local and 5/global, B1 BF16
no-cache text inference/no-grad, zero-based positions, and S1 through S1024.
All live Python/module/config/weight validation remains outside Dynamo; only
tensor-explicit family functions enter the compiled graph. The facade is
bitwise to eager across eager/Inductor, retains bounded public S1/S>1 graphs
and one FA4 key per family, rejects later mutation and unsupported requests
before compiled entry, and passes focused sanitizers with unchanged retained
codegen. EXP-0026 separately accepts only the pinned global layer-5 compiled
StaticCache one-token facade through sequential K34 and independent K1025
after eager prefill. EXP-0027 rejects a local mutable-counter candidate;
EXP-0028 separately accepts only the pinned local layer-0 compiled
`StaticSlidingWindowLayer` one-token facade through K33/K34 underfill, K1024
boundary fill, and repeated saturated rollover. Raw `torch.compile(layer)`,
compiled prefill, cached vision/document metadata, full-model compilation, and
varlen facade inputs remain unsupported. EXP-0042 separately widens the
guarded no-cache facade to all 60 locked layer indices. EXP-0043 separately
widens the guarded one-token compiled-cache facade to those same 60 indices;
compiled prefill and full-model execution remain outside both APIs.

## Pinned boundary

- Transformers base revision:
  `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Reviewed two-file patch:
  `patches/transformers/0001-gemma4-forward-vision-block-ids.patch`
- Patch SHA256:
  `ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79`
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
K>2048, that exception propagates before forward. EXP-0041's cooperative V512
forward is the default; the exact two-V256 composition remains the explicit
rollback. Padded rows scatter back as exact zero output and `-inf` FP32 LSE.

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

Framework FakeTensor and raw `torch.compile(layer)` tracing still fail closed.
Lower-level local/global kernel-wrapper FakeTensor compilation passes mixed
plateaus in EXP-0015, and EXP-0023's explicit guarded no-cache facade passes
its original pinned layer-0/layer-5 envelope. EXP-0042 extends that same API
to all 60 locked indices and captures the exact construction-time layer index.
Its API validates live state before every call and admits no
cache/mask/fallback/offset/gradient request. Public defaults retain separate
S1/S>1 graph classes; the one-graph size-oblivious result is an explicitly
nondefault diagnostic. EXP-0026's separate global layer-5 cache
facade passes one-token K33/K34 and K1025 decode after eager prefill.
EXP-0028's separate local layer-0 cache facade passes one-token K33/K34,
K1024 boundary fill, and two saturated rolls through absolute position 1025
after eager prefill. Neither cache facade is evidence for compiled prefill,
cached vision/document metadata, or full-model execution. EXP-0043 later
widens only their exact cache-layer index selection to all 60 locked indices.

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

EXP-0042 runs every actual pinned layer index 0..59 at S1 through both eager
and Inductor guarded facades. All 120 comparisons are bitwise to the matching
eager layer, with zero graph breaks, exactly two family graphs, and exactly two
FA4 forward application classes. The representative local/global S33/S1024
matrix also remains bitwise with its replay, nondefault-stream, mutation, and
negative-input guards. Layers 58/local and 59/global retain the pinned
`store_full_length_kv=True` marker; because `num_kv_shared_layers=0`, no later
layer consumes those diagnostic stores.

EXP-0043 runs eager K32 prefill followed by guarded Q1/K33 decode for every
actual pinned cache layer under both eager and Inductor facade backends. All
120 decode outputs and target cache states are bitwise to independent
weight-identical eager layers, prepared FP32 LSE replay is bitwise, reference
tolerances pass, and same-family layer-index mutations reject before compiled
entry or cache mutation. Each backend retains exactly two family cache graphs
with zero graph breaks. The EXP-0026 global K33/K34/K1025 and EXP-0028 local
underfill/boundary/saturated-rollover matrices also pass unchanged.
