# Gemma 4 31B attention contract — Transformers audit

**Audit date:** 2026-07-19<br>
**Model:** `google/gemma-4-31B` at revision
`2d418d1b7ed8c04d732c3359e19a11fbc85b6842`<br>
**Transformers:** `huggingface/transformers` at
`7ea2320c76117e6742364808a666ef6f2fb40a67`

This document defines the semantic boundary that every reference, wrapper,
kernel, and benchmark must obey. The machine-readable form is
`configs/model/gemma4-31b.lock.json`.

## 1. Static model shape

| Property | Value |
|---|---:|
| Text layers | 60 |
| Sliding/full pattern | five sliding, then one full, repeated |
| Sliding layers | 50 |
| Full layers | 10 |
| Hidden size | 5376 |
| Q heads | 32 in both families |
| Local KV heads | 16 |
| Global KV heads | 4 |
| Local QK/V dimension | 256 |
| Global QK/V dimension | 512 |
| Local window | 1024 |
| Maximum positions | 262144 |
| Attention dropout | 0.0 |
| Attention scale | 1.0 |
| Cross-layer KV-shared tail layers | 0 (feature disabled in this checkpoint) |

The full layer's GQA ratio is 8, not a placeholder ratio inferred from another
Gemma size. The local ratio is 2.

## 2. What enters the FMHA kernel

Transformers builds query states as:

```text
Q = RoPE(QNorm(q_proj(hidden)))
```

Local layers build independent K and V projection sources:

```text
K = RoPE(KNorm(k_proj(hidden)))
V = VNorm(v_proj(hidden))
```

Global layers set `attention_k_eq_v=true`, so V begins from the K projection
source, but the preparation immediately diverges:

```text
Z = k_proj(hidden)
K = partial_proportional_RoPE(KNorm(Z))
V = VNorm(Z)
```

`KNorm` has its learned scale; `VNorm` is constructed without a learned scale.
RoPE is applied to K, not V. Therefore the attention call receives separate K
and V tensors. Pointer aliasing is neither required nor generally present.

### Consequences

- The base FMHA API is `attention(q, k, v, scale=1.0, ...)`.
- A plain attention backward returns distinct dQ, dK, and dV.
- `dK + dV` is not the gradient of the shared projection source. A future
  fused preparation backward must first apply the separate K-normalization +
  RoPE and V-normalization adjoints, then sum their source gradients.
- Existing MLA/shared-KV kernels are implementation references only until an
  algebraic equivalence test proves they compute this exact operation.

## 3. Attention scale

`Gemma4TextAttention` sets `self.scaling = 1.0` and passes it through the
selected attention interface. This intentionally bypasses the generic
`1/sqrt(head_dim)` default used by many attention APIs.

Every FA4 call must pass `softmax_scale=1.0`. A missing argument is a silent
model-correctness failure.

## 4. Local and global masks

Transformers creates two masks for the text stack:

```text
full_attention    = causal
sliding_attention = sliding_window AND (causal OR blockwise)
```

`blockwise` means query and key have the same **nonnegative** vision-block ID.
Ordinary text positions carry ID `-1`. Vision block IDs are generated per
contiguous image region.

For a local query at absolute position `q` and key `k`, the exact predicate is:

```text
k > q - 1024
AND
(k <= q OR (vision_id[q] == vision_id[k] AND vision_id[q] >= 0))
```

The lower inequality is strict. Thus the FlashAttention argument is
`window_size_left=1023`, `window_size_right=0`.

The overlay supplies only this left lower bound. Therefore, for a query inside
a vision block, same-block future keys are not capped by a separate right
window; far-past keys are still excluded by `k > q - 1024`.

Global layers remain causal even for vision tokens. Padding, packed-document,
and cache offsets must be composed without changing these predicates.

During cached generation, Transformers drops multimodal token-type IDs after
the first iteration so bidirectional vision masking is not reapplied on top of
already cached states.

## 5. Rotary position encoding

| Layer | Rope type | Theta | Partial factor |
|---|---|---:|---:|
| Sliding | default | 10000 | full Q/K head |
| Full | proportional | 1000000 | 0.25 |

For full attention, 0.25 of 512 means a 128-channel rotary subspace. Q and K
are already rotated before the attention backend is called. The first kernel
milestone should therefore consume prepared Q/K/V; projection/norm/RoPE fusion
is a later optimization with its own gradient contract.

## 6. Cross-layer KV sharing is disabled in this checkpoint

The current Transformers implementation supports a separate
`num_kv_shared_layers` feature, but the pinned 31B config sets it to `0`.
Every text layer therefore owns its normal K/V projection/preparation path;
there is no last-layer KV reuse to include in the initial kernel integration.

Keep this configuration feature distinct from `attention_k_eq_v`. A future
checkpoint could enable cross-layer reuse, in which case the backend would
still receive separate prepared K and V tensors and must treat them as
read-only. That future case is outside the initial fast-path contract.

## 7. Tensor and numerical contract

Initial target:

```text
Q: (B, Sq, Hq, Dqk) BF16
K: (B, Sk, Hkv, Dqk) BF16
V: (B, Sk, Hkv, Dv) BF16
O: (B, Sq, Hq, Dv) BF16
LSE: (B, Hq, Sq) FP32
score / softmax / gradient accumulation: FP32
scale: 1.0
```

Fixed and packed-varlen forms must preserve lower-right causal alignment when
`Sq != Sk`. Input last dimensions must satisfy the selected FA4 vectorization
and alignment rules; unsupported strides reject or fall back rather than being
silently reinterpreted.

## 8. Golden checks

The repository tests lock:

- the 60-layer pattern and full shape;
- scale 1.0 versus `1/sqrt(d)`;
- the strict local-window boundary;
- vision bidirectionality only in local layers;
- GQA expansion;
- distinct K/V preparation from a shared source;
- independent dQ/dK/dV;
- FP32 LSE;
- explicit verification that cross-layer KV sharing is disabled.

With the pinned Transformers checkout installed, run:

```bash
python scripts/verify_model_contract.py --transformers
pytest -q tests/test_hf_oracle_optional.py
```

Use `--online` to compare the locked model fields with the pinned Hub revision.

## 9. Source anchors

- Model config: `google/gemma-4-31B/config.json`, pinned above.
- `Gemma4TextAttention` and `create_masks_for_vision_model`:
  `src/transformers/models/gemma4/modular_gemma4.py` at the pinned Transformers
  revision.
- `sliding_window_overlay`, `blockwise_overlay`, and mask composition:
  `src/transformers/masking_utils.py` at the same revision.
- Vision block IDs originate in Gemma 3's
  `get_block_sequence_ids_for_mask`: text is `-1`; contiguous image regions
  receive nonnegative group IDs.
