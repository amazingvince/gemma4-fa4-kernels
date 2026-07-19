# Kernel workspaces

Do not copy large upstream kernel files here and let them drift. The pinned
FlashAttention checkout lives under `.upstream/flash-attention`; implementation
branches should normally be made in a fork/worktree of that repository.

This directory holds focused patches, design briefs, adapters, or small new
modules that are not yet placed upstream.

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
