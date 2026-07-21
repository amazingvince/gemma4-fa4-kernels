# H100 all-layer compiled StaticCache design brief

EXP-0043 widens only the accepted guarded one-token compiled-cache facade
from layer 0/local and layer 5/global to every locked Gemma 4 text layer. It
does not compile prefill or the model, change a GPU kernel, or add a cache
layout.

## Scope and invariants

- H100/SM90, BF16, B1/Q1, inference/no-grad, text-only decode after eager
  nonempty prefill remains the complete public boundary.
- The facade binds the exact construction-time layer index and that exact
  `StaticCache.layers[layer_idx]` object, tensor identities, storage addresses,
  geometry, counters, and versions.
- All 60 indices must match the locked family map: 50 local sliding-window
  cache layers and 10 global cache layers.
- A later module-index mutation, including another index in the same family,
  must fail before compiled entry or cache mutation.
- Layers 58/local and 59/global retain the pinned
  `store_full_length_kv=True` marker. Its eager dictionary write has no consumer
  in the locked `num_kv_shared_layers=0` checkpoint; the explicit facade does
  not expose shared-KV dictionary state.
- Global logical-length and local absolute/saturated-counter transactions stay
  exactly as accepted in EXP-0026 and EXP-0028.

## Single structural change

Store `layer_idx` in each cache binding and facade. Select and revalidate
`cache.layers[layer_idx]` rather than the historical indices 5/global and
0/local. The tensor-only compiled functions remain family-specific, so layer
index never enters their graph or FA4 compile keys.

## Frozen gates

- CPU/FakeTensor construction and exact-index mutation guards across 0..59.
- All 60 actual pinned layers: eager prefill to K32 followed by one compiled
  Q1/K33 decode under both eager and Inductor facade backends.
- Bitwise equality to an independently instantiated, weight-identical eager
  layer; exact active cache bytes, mutation slots, logical counters, and FP32
  LSE/reference checks.
- Exactly two family cache graphs and at most one added FA4 application class
  per family across the all-layer sweep.
- Existing EXP-0026 global K33/K34/K1025 and EXP-0028 local
  underfill/boundary/saturated-rollover probes remain green, including hostile
  tails, nondefault streams, negative matrices, and cache-key bounds.
- The exact EXP-0041 FlashAttention patch, strict environment, full suites,
  pinned Transformers oracle, and bundle verifier remain unchanged and green.

## Rollback

Reverting the project integration change restores the layer-0/layer-5 cache
facades. No upstream patch or kernel rollback is required.
