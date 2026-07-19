# H100 pinned-Transformers integration

This document defines the eager framework boundary accepted by EXP-0011 and
extended through K2048 training by EXP-0012. It does not widen any model
invariant in `docs/model-contract.md`, claim a fused global d512 kernel, or
cover B300.

## Pinned boundary

- Transformers base revision:
  `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Reviewed one-file patch:
  `patches/transformers/0001-gemma4-forward-vision-block-ids.patch`
- Patch SHA256:
  `773950a1f1feb04f5f2e6a1d66f8953ff8905e8ca9391f804089f169da59b671`
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

## Prepared tensors and outputs

Pinned `Gemma4TextAttention` supplies distinct prepared Q/K/V tensors in BHSD.
The adapter validates BF16, the locked layer-index geometry, scale 1.0, zero
dropout, distinct K/V storage, positive nonoverlapping strides, unit D stride,
outer strides divisible by eight BF16 elements, and 16-byte base alignment.
Its canonical BHSD-to-BSHD transpose shares the original allocation; fixed
paths do not make an unconditional contiguous copy.

Packing copies are localized to valid rows when padding or packed positions
require a gather. Global lower-right training composition may create a zero Q
prefix for the causal coordinate system, and V512 is materialized as two V256
slabs for the accepted global compositions. Padded rows scatter back as exact
zero output and `-inf` FP32 LSE.

The Transformers attention interface returns `(output, None)`: its second
return is attention weights, not LSE. The project-level
`gemma4_fa4_prepared` result exposes `output`, FP32 `lse` when the selected FA4
route provides it, and a route string. The interface also stores that route on
the attention module as `_gemma4_fa4_last_path` for diagnostics.

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
| Global batched/padded/packed/lower-right, every K segment <=2048 | `fa4_global_varlen` | exact per-segment backward-capable composition; no-grad composed route when every K<=1024 |
| Global fixed or lower-right, K>1024 through K262144 | `fa4_global_forward_only` | no grad only |
| Global packed/varlen with any K>1024 through K262144 | `fa4_global_varlen_forward_only` | no grad only |
| Exact non-native mask/layout | `flex_attention` | inference only |

The long global routes require nonempty `1 <= Sq <= Sk <= 262144`, preserve
lower-right causality, and preflight their conservative composed output/LSE
allocation estimate against 80% of currently free HBM. Training-capable
global composition remains limited to K2048 per segment; native packed
backward and larger K remain separate work.

FlexAttention is a correctness fallback for eager inference only. A diagnostic
D512 B2/S5 backward candidate produced a non-finite dQ after a larger default
tile had already exceeded H100 shared memory. The production adapter therefore
rejects every gradient-capable fallback before launch. Disabling fallback turns
every unsupported request into an explicit `UnsupportedH100Path`.

Framework FakeTensor and `torch.compile` tracing also fail closed. They need a
separately designed ABI, static-cache contract, and compile-key audit; eager
success is not evidence for compiled execution.

## Recorded H100 evidence

```bash
python scripts/probe_h100_transformers_integration.py --case all
python scripts/probe_h100_transformers_integration.py \
  --case global-forward-only-max-context
```

The first command passes ten cases covering zero-copy fixed views, local
padding and lower-right packing, global fixed and B2/S5 per-segment training,
global fixed/varlen forward-only K2048, EXP-0012 Q33/K1025 lower-right and
mixed packed K2048 backward, authoritative mask transport, the registered
backend, and an actual pinned `Gemma4TextAttention` local forward/backward.
The second is an exact Q1/K262144 zero-score sentinel:
output is `64/262144` and LSE is `log(262144)`.

Memcheck, synccheck, and racecheck pass for the global composed B2/S5 training
case, packed global Q33/K2048 forward-only case, fixed S1025 backward, and
mixed packed K2048 backward. No native-varlen backward
generated-code count, performance result, full-checkpoint run, compiled-model
result, or B300 result is claimed.

The isolated cache probe retains 15 object paths with nine unique contents
and 976,336 total bytes. The three global-backward block classes add 8, 3, and
3 paths respectively; composed segment order, runtime values, batch size,
legal strides, and long fixed-forward lengths add none. Native packed forward
adds one distinct object. Exact keys and hashes are recorded in EXP-0011. The
checksum-locked repository verifier and schema-valid result record pass against
implementation revision `e7f26bba9b6795e3022c733cff39e060075daf57`.
