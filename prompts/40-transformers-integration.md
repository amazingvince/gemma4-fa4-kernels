# Transformers integration prompt — per-layer Gemma 4 FA4 routing

Integrate verified kernel entry points with the pinned Transformers Gemma 4
implementation. This is an integration/correctness task, not a kernel-tuning
task.

Read `docs/hf-implementation-audit.md` and run the optional HF oracle first.
Preserve these boundaries:

- Transformers prepares `(B,H,S,D)` Q/K/V; FA4 receives `(B,S,H,D)` views with
  legal real strides and last-dimension alignment;
- `softmax_scale=1.0` is always explicit;
- global prepared K and V are distinct;
- local text routing uses causal left window 1023;
- multimodal local routing must carry
  `window AND (causal OR same nonnegative vision block)` explicitly;
- the generic pinned Transformers `flash_attention_4` mask adapter is a known
  strict expected failure for the future-within-vision-block exception;
- full/global layers remain causal;
- unsupported output-attention, cache, varlen, stride, or mask cases reject or
  use a semantically exact fallback.

Deliver a small adapter plus tests that compare O, LSE, dQ, dK, and dV against
the pinned eager/FlexAttention oracle on text, vision-boundary, padding,
packed-document, equal-length, and lower-right-offset cases. Do not add an
unconditional `.contiguous()` copy. Record compile/cache variants and state
which multimodal path is implemented.
