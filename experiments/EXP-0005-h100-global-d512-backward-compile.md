# EXP-0005: H100 composed global d512 backward compile

- Date / author: 2026-07-19 / Codex
- Kernel family: global-d512-bwd
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- H100 patch:
  `patches/flash-attention/0001-sm90-d512-v256-forward.patch`, SHA256
  `8d3404ccc8bdb2b3fd8e6de09e1f100827d071de283885bf737932bcb41aca4f`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- QuACK runtime helpers: `quack-kernels==0.5.3`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `1e1767ac7a87ba5503a97aa38bbefcccef5fa88e9793f330bdd9b3658fcceb75`
- Strict environment artifact: `agent_space/h100-check-precommit.json`
- Exemplar path and revision: pinned `flash_attn/cute/interface.py`,
  `flash_bwd_sm90.py`, backward preprocess/postprocess files, and
  `tests/cute/test_flash_attn.py`

## Invariant changed

No model invariant changes. Keep B=1, fixed-length contiguous BF16 BSHD,
32 query heads, 4 KV heads, GQA-8, Q/K/V dimensions 512, causal masking,
scale 1.0, distinct K/V, FP32 reference accumulation, and separate dQ/dK/dV.

The accepted forward remains the exact two-launch V256 composition from
EXP-0002. For V slabs `V0,V1`, autograd must implement:

```text
dQ = dQ0 + dQ1
dK = dK0 + dK1
dV = concat(dV0, dV1)
```

It must never merge dK with dV or assume prepared K equals prepared V.

## Hypothesis

The pinned generic SM90 backward selected behind each accepted
`(Dqk,Dv)=(512,256)` forward launch can fake-compile unchanged at exact
B1/S128/32Q/4KV/GQA-8/causal/scale-1 geometry, and autograd through both slab
launches returns separate BF16 gradients with shapes
`dQ=(1,128,32,512)`, `dK=(1,128,4,512)`, and `dV=(1,128,4,512)`.

This is a compile hypothesis, not a correctness or support claim. The current
dispatcher falls through its `head_dim > 192` branch, whose comment says
"hdim 256"; exact d512 legality has not been established.

## Single change

Add only a project-side compile/correctness probe for autograd through
`fa4_global_text_forward`. Do not change the pinned patch, dispatcher, tile,
stages, atom layouts, kernel source, adapter contract, or numerical policy in
this experiment.

First run the exact S128 fake compile. If it fails, reject this hypothesis and
record the first compiler/validation error, realized dispatch values, and
cache key if emitted. Do not combine the failure with a speculative kernel
patch. A follow-up experiment must isolate one configuration or structural
change.

## Conditional correctness matrix

Only if fake compile passes, run seeds `5000 + S` at
S=`1,31,32,33,63,64,65,127,128,129,511,512,513,1024`. Each case must verify:

- exact B1/32Q/4KV/GQA-8/d512 shapes, causal mask, and scale 1.0;
- distinct K/V inputs and distinct full dQ/dK/dV output allocations;
- finite BF16 gradients;
- high-reference candidate max/mean error for dQ, dK, and dV;
- the same independently implemented upstream-style BF16-relative rule frozen
  in EXP-0004, using full d512 V in one reference operation;
- exact composition: autograd sums both slab dQ/dK contributions and maps the
  two dV gradients back to the corresponding full-d512 halves.

Then run S128 three times on identical inputs and once on a nondefault CUDA
stream. Report exact equality without requiring it for acceptance.

- [x] exact fake-tensor composed backward compile (**rejected at constructor**)
- [ ] declared S matrix (conditional on compile)
- [ ] separate finite dQ, dK, and dV (conditional on compile)
- [ ] repeated-run check (conditional on compile)
- [ ] nondefault-stream check (conditional on compile)

## Synchronization and generated code

Only after compile and correctness pass:

- [ ] compute-sanitizer memcheck at S128 and S129
- [ ] compute-sanitizer synccheck at S128 and S129
- [ ] compute-sanitizer racecheck at S128 and S129
- [ ] PTX/SASS instruction path recorded
- [ ] registers, spills, static SMEM, and modeled dynamic storage recorded

Because composed backward launches the slab kernel twice, evidence must cover
the main d512-QK/d256-V backward specialization and its preprocess/postprocess,
not only the outer `cat` autograd nodes.

## Measurement

No performance measurement is authorized. The two-launch path duplicates QK,
softmax, and backward score work and is a correctness composition only.

## Decision

**REJECT the unchanged generic composed-backward configuration.** The exact
command was:

```bash
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_FAKE_TENSOR=1 \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0005-global-bwd-true-fake \
  python scripts/probe_h100_global_backward.py --seqlen 128
```

Autograd reached the pinned SM90 backward constructor and failed before main
backward compilation:

```text
flash_attn/cute/flash_bwd_sm90.py:116
assert self.same_hdim_kv, "GQA backward requires head_dim == head_dim_v"
AssertionError: GQA backward requires head_dim == head_dim_v
```

For each accepted forward slab, `head_dim=512`, `head_dim_v=256`, and the
model's 32Q/4KV geometry gives GQA ratio 8. No direct-GQA main-backward cache
key or cubin was emitted, so no real direct-GQA execution, numerical matrix,
sanitizer, generated-code inspection, or performance measurement ran. The
conditional checklist remains unchecked by design.

This rejects only direct GQA-8 use of the pinned unequal-dimension backward.
It does not reject global d512 backward in general. Expanding K and each V
slab from 4 to 32 heads would bypass this assertion exactly and let autograd
reduce repeated-head dK/dV contributions back to four model KV heads, but it
is not by itself a viable configuration: the realized M64 x N64 monolithic
path models 320 accumulator registers and 336 KiB core shared storage, above
the search budgets of 216 registers and 224 KiB. Even N32 remains 280 KiB
while retaining a full d512 shared dQ accumulator.

The next experiment must therefore combine exact internal head expansion with
a structural resource change, such as D256-chunked dQ accumulation or separate
Q-major dQ and K-major dK/dV ownership. A head-expansion-only compile may be
used as a diagnostic to expose the next error, but must not be presented as a
durable correctness candidate.

That diagnostic was run through the same project probe after adding explicit
`FakeTensorMode` activation rather than treating the environment variable as a
fake-mode context:

```bash
# Exact internal 4-to-32 expansion: true fake compile.
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_FAKE_TENSOR=1 \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0005-expanded-fake \
  python scripts/probe_h100_global_backward.py --seqlen 128 --expand-kv-heads

# Same specialization: real launch validation.
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0005-expanded-real \
  python scripts/probe_h100_global_backward.py --seqlen 128 --expand-kv-heads
```

The explicit fake compile passed and returned the exact full gradient shapes.
The real launch then rejected the generated main kernel before execution:

```text
launch shared memory exceeds current GPU arch sm_90a allowed.
Allocated: 345088 bytes. Max: 232448 bytes.
```

The 345,088-byte aligned allocation is 337 KiB: 336 KiB of modeled core
storage plus 1 KiB of metadata/alignment. This confirms that bypassing the GQA
assertion is semantically exact but physically insufficient. It produced no
global-backward correctness or sanitizer evidence.
