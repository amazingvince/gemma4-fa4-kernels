# Kernel workspaces

Do not copy large upstream kernel files here and let them drift. The pinned
FlashAttention checkout lives under `.upstream/flash-attention`; implementation
branches should normally be made in a fork/worktree of that repository.

This directory holds focused patches, design briefs, adapters, or small new
modules that are not yet placed upstream.

Current H100 M1 code lives in `src/gemma4_fa4/h100.py`; the reviewed upstream
patch is in `patches/flash-attention/`. Local d256 text forward/autograd
backward and exact composed plus native packed global d512 text
forward/backward pass their scoped gates. EXP-0003's fixed elementwise
envelope and EXP-0005's unchanged
asymmetric GQA-8 backward remain recorded rejections. EXP-0006 accepts a split
global path with one dKV-only and two D256 dQ-only main launches per V256 slab,
FP32 cross-slab dQ/dK accumulation, and separate dV-slab conversion. Its
14-length H100 matrix, S128/S129 sanitizers, and generated-code resource gates
pass. It is a six-main-launch correctness path with nondeterministic FP32
bulk/atomic reduction, not a performance result. EXP-0007 accepts fixed B1
local multimodal forward/backward. EXP-0008 accepts nonempty packed local
native/custom forward/backward through per-sequence S1025, including
lower-right alignment and vision/document isolation. EXP-0009 accepts native
packed local text through the locked per-sequence S262144 maximum. EXP-0010
accepts exact sparse-scheduled vision/document metadata through that maximum
inside its declared resource envelope; the dense custom path remains the
S<=1025 route. EXP-0011 accepts the eager framework boundary, and EXP-0012
extends fixed and exactly composed global backward through K2048. EXP-0013
accepts native THD/cu-seqlens global backward for nonempty per-segment
`1 <= Sq <= Sk <= 2048` under exact BF16 32Q/4KV/GQA-8/d512/lower-right-causal/
scale-1.0/distinct-K/V geometry. Only the dedicated native HBM-budget exception
may select the EXP-0012 composer; validation, contract, assertion, and runtime
failures propagate. EXP-0014 extends only the native packed route to nonempty
`1 <= Sq <= Sk <= 262144` segments under signed-INT32 and guarded-HBM
admission. Fixed BSHD and the exact composer remain capped at S/K2048; for
K>2048, a native budget rejection propagates before forward and cannot select
the composer or FlexAttention. It remains a slabbed/split correctness path,
not a fused d512 or performance result. EXP-0015 accepts mixed packed local and
global segments satisfying `0 <= Sq <= Sk <= 262144` with positive aggregate
totals and positive exact maxima. Empty-query segments launch no owned query
work, retain exact-zero K/V gradients, and add no cache class or changed main
object; all-empty physical workloads still reject before backend launch.
EXP-0016 accepts eager B1 text-only StaticCache active-prefix prefill/decode
with no active backward. Deterministic gradients, framework FakeTensor/fullgraph
`torch.compile`, compiled StaticCache, performance, and B300 remain unverified.
No-cache framework compilation is the next compatibility gate; compiled
StaticCache follows only after that boundary is proven. See `docs/status.md`
and EXP-0001 through EXP-0016.

Planned families:

| Family | Scope | Closest upstream exemplars |
|---|---|---|
| local-d256-sm103 | exact local text/vision fwd+bwd; 1CTA vs 2CTA | dedicated d256 SM100 files + generic local SM100 path |
| global-d512-sm90 | distinct K/V causal fwd + owner backward | SM90 fwd/bwd, large-head FlashInfer ideas as secondary evidence |
| global-d512-sm103 | distinct K/V causal fwd + owner backward | generic SM100, dedicated d256, MLA mechanics only where algebra matches |
| hf-integration | per-layer framework dispatch, context offsets, and explicit no-cross-layer-KV-reuse integration | FA4 interface + Transformers attention integration |

Before code, complete
`skills/writing-cute-dsl-kernels/templates/kernel-design-brief.md`. Every
constexpr/codegen flag belongs in the compile key. Tuning values live in
`tuning/`, not hidden in kernel source.
