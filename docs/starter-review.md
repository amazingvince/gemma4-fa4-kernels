# Starter scaffold review and corrections

The original scaffold had a good experimental loop but encoded several unsafe
model assumptions. This bundle makes the following corrections executable.

## Corrected model assumptions

1. Global attention uses 32 Q heads and 4 KV heads, not a provisional 8-Q
   geometry.
2. `attention_k_eq_v=true` shares the K-projection source only. Prepared K and
   V differ because K uses learned K RMSNorm plus partial/proportional RoPE,
   while V uses unscaled RMSNorm and no RoPE.
3. Attention scale is exactly 1.0; the generic `1/sqrt(d)` default is wrong.
4. Local masking is exactly:

   ```text
   key > query - 1024
   AND
   (key <= query OR same nonnegative vision block)
   ```

   The overlay has no independent right cap, so future same-block vision keys
   remain valid; far-past keys remain window-limited.
5. Global layers stay causal for vision.
6. `num_kv_shared_layers=0`; the checkpoint does not reuse prepared KV across
   decoder layers.
7. Attention-only backward produces separate dK and dV. A future shared-source
   fusion combines source gradients only after applying distinct preparation
   adjoints.

## Corrected workflow assumptions

- fwd, bwd, and fwd+bwd timing regions are distinct;
- a full-causal SDPA run is not labeled as a sliding-window baseline;
- full model geometry remains visible even when memory preflight skips it;
- upstream revisions and the CuTe DSL dependency are exact, detached locks;
- CUDA/driver administration is separate from the Python environment;
- remote profiles support direct SSH, module initialization, proxy jumps, and
  scheduler launchers without committing secrets;
- the full user-supplied CuTe skill is preserved and validated, rather than
  reducing it to a short summary.

## Known integration gap retained deliberately

At the pinned Transformers revision, the generic FA4 mask adapter does not
carry the blockwise future-vision exception. The bundle records this as a
strict expected failure. Text-only local support may land first, but
multimodal correctness requires an explicit mask metadata or `mask_mod`
integration path.
