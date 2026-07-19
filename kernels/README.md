# Kernel workspaces

Do not copy large upstream kernel files here and let them drift. The pinned
FlashAttention checkout lives under `.upstream/flash-attention`; implementation
branches should normally be made in a fork/worktree of that repository.

This directory holds focused patches, design briefs, adapters, or small new
modules that are not yet placed upstream.

Current H100 M1 code lives in `src/gemma4_fa4/h100.py`; the reviewed upstream
patch is in `patches/flash-attention/`. Local d256 text forward/autograd
backward and exact composed global d512 text forward/backward pass their
scoped gates. EXP-0003's fixed elementwise envelope and EXP-0005's unchanged
asymmetric GQA-8 backward remain recorded rejections. EXP-0006 accepts a split
global path with one dKV-only and two D256 dQ-only main launches per V256 slab,
FP32 cross-slab dQ/dK accumulation, and separate dV-slab conversion. Its
14-length H100 matrix, S128/S129 sanitizers, and generated-code resource gates
pass. It is a six-main-launch correctness path with nondeterministic FP32
bulk/atomic reduction, not a performance result. EXP-0007 accepts fixed B1
local multimodal forward/backward. EXP-0008 accepts nonempty packed local
native/custom forward/backward through per-sequence S1025, including
lower-right alignment and vision/document isolation. Production context above
1025 is next. See `docs/status.md` and EXP-0001 through EXP-0008.

Planned families:

| Family | Scope | Closest upstream exemplars |
|---|---|---|
| local-d256-sm103 | exact local text/vision fwd+bwd; 1CTA vs 2CTA | dedicated d256 SM100 files + generic local SM100 path |
| global-d512-sm90 | distinct K/V causal fwd + owner backward | SM90 fwd/bwd, large-head FlashInfer ideas as secondary evidence |
| global-d512-sm103 | distinct K/V causal fwd + owner backward | generic SM100, dedicated d256, MLA mechanics only where algebra matches |
| hf-integration | per-layer dispatch, varlen, KV-shared tail | FA4 interface + Transformers attention integration |

Before code, complete
`skills/writing-cute-dsl-kernels/templates/kernel-design-brief.md`. Every
constexpr/codegen flag belongs in the compile key. Tuning values live in
`tuning/`, not hidden in kernel source.
