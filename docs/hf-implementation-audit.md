# Pinned Hugging Face Gemma 4 implementation audit

This document is the semantic source map for the kernel project. It separates
three ideas that are easy to conflate:

1. the exact `google/gemma-4-31B` checkpoint configuration;
2. tensor preparation performed by `Gemma4TextAttention` before FMHA;
3. masks constructed by the model and what the generic Transformers attention
   backend adapters can actually carry.

## Pinned sources

- Model config: `google/gemma-4-31B` revision
  `2d418d1b7ed8c04d732c3359e19a11fbc85b6842`.
- Transformers: `huggingface/transformers` revision
  `7ea2320c76117e6742364808a666ef6f2fb40a67`.
- Relevant files:
  - `src/transformers/models/gemma4/configuration_gemma4.py`
  - `src/transformers/models/gemma4/modeling_gemma4.py`
  - `src/transformers/models/gemma4/modular_gemma4.py`
  - `src/transformers/models/gemma3n/modeling_gemma3n.py`
  - `src/transformers/masking_utils.py`

The machine-readable checkpoint subset is
`configs/model/gemma4-31b.lock.json`. Run
`python scripts/verify_model_contract.py --online --transformers` to compare
both the Hub config and installed pinned Transformers classes.

## Exact text-attention geometry

| Field | Checkpoint value |
|---|---:|
| decoder layers | 60 |
| layer pattern | five sliding, then one full, repeated ten times |
| query heads | 32 |
| sliding KV heads | 16 |
| full/global KV heads | 4 |
| sliding head dimension | 256 |
| full/global head dimension | 512 |
| sliding window | 1024 tokens |
| maximum positions | 262144 |
| attention dropout | 0.0 |
| `attention_k_eq_v` | true |
| `num_kv_shared_layers` | 0 |
| multimodal bidirectionality | `vision` |
| full RoPE | proportional, theta 1e6, partial factor 0.25 |
| sliding RoPE | default, theta 1e4 |

The active checkpoint does **not** use cross-layer KV reuse. The implementation
contains generic support for it, but `num_kv_shared_layers=0` means no decoder
layer consumes another layer's prepared K/V states.

## Tensor preparation before FMHA

For a sliding layer, the module creates independent Q, K, and V projections.
It applies learned RMS normalization to Q and K, unscaled RMS normalization to
V, and RoPE to Q and K.

For a full layer with `attention_k_eq_v=true`, only the projection source is
shared. In index notation, the implementation is conceptually:

```text
Q0 = Wq X
Z  = Wk X                    # one projection source
Q  = RoPE(QNorm(Q0))
K  = pRoPE(KNorm(Z))
V  = VNorm(Z)                # distinct normalization; no RoPE
O  = Attention(Q, K, V, scale=1.0, causal=True)
```

Therefore K and V are distinct at the FMHA boundary. A normal attention kernel
must accept separate K and V tensors and return separate dK and dV. A later
model-specific fusion may load `Z` once, produce both operands on-chip, apply
the two preparation adjoints in backward, and then combine their gradients
into dZ. That fusion is not algebraically equivalent to passing `k is v` into
ordinary FMHA.

## Attention scale

`Gemma4TextAttention` sets `self.scaling = 1.0` and passes it to the selected
attention interface. The kernel input Q/K are already normalized; the generic
FlashAttention default `1/sqrt(head_dim)` is wrong for this checkpoint. All
wrappers, references, baselines, and cache-specialized kernels in this project
must pass or specialize `softmax_scale=1.0`.

## RoPE boundary

The MVP FMHA contract receives prepared Q/K. Proportional RoPE therefore does
not live in the initial kernel. The full-layer rotary subspace is 25% of 512,
i.e. 128 channels. Any later Q/K preparation fusion must be tested against the
pinned `ROPE_INIT_FUNCTIONS["proportional"]` and the pinned
`apply_rotary_pos_emb` behavior instead of reimplementing it from memory.

## Mask construction

The model constructs two masks:

```text
full_attention    = causal
sliding_attention = sliding_window AND (causal OR same_nonnegative_vision_block)
```

`sliding_window_overlay(W)` keeps a pair when `kv_position > q_position - W`.
For W=1024 and causal attention this is exactly the inclusive range
`q-1023 <= k <= q`, so the FA runtime argument is `window_size_left=1023`.
When the blockwise vision exception admits a future key, the overlay adds no
independent right bound: future keys in the same vision block can remain valid,
while past keys older than `q-1023` are excluded.

`blockwise_overlay` keeps a pair only when query and key have the same
nonnegative block ID. Text tokens use `-1`; contiguous image regions receive
nonnegative IDs. Packed-document and padding masks are additional intersections,
not replacements for the vision rule. Full/global layers remain causal even
for image tokens.

## Current generic Transformers FA4 adapter limitation

At the pinned Transformers revision, `ALL_MASK_ATTENTION_FUNCTIONS` maps
`flash_attention_4` to the generic flash mask adapter. That adapter forwards a
2D padding/length mask and does not materialize or transmit the composed
`mask_function`. The FA4 call can still receive causal and sliding-window
arguments, but the future-token exception inside a vision block is not encoded
by that generic adapter.

The project records this as a strict expected failure in
`tests/test_hf_oracle_optional.py`. The first local kernel may target text-only
semantics. Multimodal support needs an explicit integration contract, such as
compact vision-block metadata plus a specialized tile classifier or a supported
FA4 `mask_mod` path. Do not mark multimodal Gemma 4 training correct merely
because the text-only local path matches.

## Tensor layouts at the integration boundary

Inside the Transformers module, prepared states are `(B, H, S, D)`. FA4's
public CuTe interface accepts `(B, S, H, D)`, so the integration wrapper must
preserve legal strides or create the required view deliberately. Last-dimension
contiguity and alignment must be validated; an unconditional `.contiguous()`
can hide stride bugs and add multi-gigabyte copies at long context.

## Backward contract

For attention-only backward:

```text
dQ = dS K
dK = dS^T Q
dV = P^T dO
```

The GQA reduction combines contributions from all Q heads mapped to one KV
head, but dK and dV remain separate. An optimized owner-computes design should
keep FP32 accumulation on-chip and write each gradient tile once rather than
allocating whole-layer FP32 accumulators at 256K context.

## Tests that enforce this audit

- `tests/test_model_contract.py`: exact geometry, pattern, scale, and disabled
  cross-layer reuse.
- `tests/test_masks.py`: text, window, vision-block, packed-document, padding,
  and offset behavior.
- `tests/test_reference.py`: O/LSE and separate dQ/dK/dV behavior, plus distinct
  prepared K/V from one source.
- `tests/test_hf_oracle_optional.py`: module and mask primitive parity, plus the
  strict expected failure for the pinned generic FA4 vision-mask adapter.
