# Kernel workspaces

Do not copy large upstream kernel files here and let them drift. The pinned
FlashAttention checkout lives under `.upstream/flash-attention`; implementation
branches should normally be made in a fork/worktree of that repository.

This directory holds focused patches, design briefs, adapters, or small new
modules that are not yet placed upstream.

Current H100 M1 code lives in `src/gemma4_fa4/h100.py`; the reviewed upstream
patch is in `patches/flash-attention/`. Local d256 text forward and the exact
two-launch global d512 text-forward composition pass. The local adapter's
autograd backward also passes the predeclared EXP-0004 upstream-relative
numerical, boundary, stream, sanitizer, and generated-code gates. EXP-0003's
fixed elementwise envelope remains a recorded rejection. Global backward and
multimodal kernels are not yet promoted. See `docs/status.md` and EXP-0001
through EXP-0004.

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
