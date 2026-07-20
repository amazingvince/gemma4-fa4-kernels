# EXP-0025: H100 explicit StaticCache view ABI

- Date / author: 2026-07-20 / Codex
- Status: **ACCEPTED — global/Inductor Q1/K33 discriminator only**
- Kernel family: pinned Transformers global-d512 attention decode over the
  retained FA4 forward path
- Architecture: sm_90
- Starting revision: `4722b1d1a58da7985aa1bd33d46ebf87c40b90af`
- Model-contract lock SHA256:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Transformers patch SHA256:
  `ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79`
- H100 FA4 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`

## Prior boundary

EXP-0024's global Q1/K33 candidate passed its numerical, prepared-reference,
cache-byte, address, graph-count, application-key, and fail-closed checks. It
was rejected because the pinned `StaticLayer` had marked K, V, and its tensor
counter as static addresses; PyTorch 2.8 lifted all three nominal arguments
into FX `get_attr` buffers. That closed cache identity over the graph instead
of retaining the declared explicit runtime tensor ABI.

No local-cache, wider-global, sanitizer, generated-code, performance,
raw-compile, compiled-prefill, or B300 result was accepted from EXP-0024.

## Invariant changed

Keep the actual pinned StaticCache backings and counter as the sole mutable
state, with their original objects and static addresses frozen and revalidated
outside Dynamo. At facade construction, create one full-shape, identical-stride
view of each backing/counter. The views must share exactly the corresponding
root storage and cover the complete root tensor, but must not themselves carry
the upstream static-address marker. Pass only these frozen views as explicit
arguments to the compiled tensor function.

The cache-aware custom op retains the same mutation schema and mutates the
views. Because they cover the complete roots, the actual pinned cache bytes and
counter must change exactly as before. Root and view object identities,
shape/stride/storage offset, shared storage pointer, root/view mutation
versions when available, and the absence of cross-K/V aliasing are validated
before every compiled entry. No copied or shadow cache is permitted.

No projection, norm, RoPE, attention, output-projection, cache-update order,
kernel, tolerance, compiler option, or public support envelope changes.

## Hypothesis

Passing frozen full-storage views of the three upstream-marked StaticCache
tensors will keep K, V, and the counter as explicit FX placeholders rather
than `get_attr` buffers, while one Inductor backend attempt still returns the
bitwise eager Q1/K33 layer output and mutates only the actual root cache slot 32
and counter byte-for-byte like the pinned eager twin.

Falsification is any cache `get_attr`; missing cache tensor placeholder;
second backend attempt; graph break; copied/shadow state; view/root pointer,
shape, stride, offset, or mutation disagreement; cache or module mutation on a
rejected call; changed numerical/reference result; different application-key
bound; or any widening beyond the declared first discriminator.

## Single change

1. Extend the EXP-0024 binding with one retained full-storage view for K, V,
   and the scalar counter.
2. Validate that each view is a complete, contiguous, zero-offset alias of
   exactly one frozen root and that K/V remain distinct.
3. Pass the views, not the marked root tensor objects, into the compiled
   tensor-only function and cache-aware custom op.
4. Require the captured graph to expose all three cache tensors as explicit
   placeholders and to contain no cache-related `get_attr` node.
5. Change nothing else and stop at global/Inductor Q1/K33.

## First discriminator

Repeat EXP-0024 candidate 6 on the actual pinned layer 5 with capacity 65:
eagerly prefill two weight-identical caches to K32, then run one compiled and
one eager Q1/K33 decode. The candidate must retain every previously passing
gate and additionally prove:

- exactly three cache tensor inputs reach the captured graph/AOT boundary;
- K, V, and counter are placeholders or backend tensor inputs, never
  graph-module attributes;
- the custom op mutates the three retained views and the corresponding actual
  root bytes/counter exactly once;
- altered position, a foreign cache argument, a rebound root, and a forged
  non-full view reject before compiled entry with byte-identical root state.

Stop and reject before eager-compiler, Q1/K1025, local rollover, sanitizer, or
codegen work if any discriminator item fails.

## Correctness gates

- [x] view/root identity, storage, shape, stride, offset, version, device,
      dtype, and complete-coverage checks fail closed
- [x] global/Inductor Q1/K33 has one backend attempt and zero graph breaks
- [x] graph contains one cache custom op, three explicit cache tensor inputs,
      and zero cache `get_attr` sources
- [x] whole-layer output is bitwise eager-equal; prepared O and FP32 LSE retain
      their frozen reference policy
- [x] only root K/V slot 32 and the root counter change, byte-for-byte equal to
      the eager twin, with all root/view addresses stable and K/V distinct
- [x] malformed position, foreign cache, rebound root, and forged view reject
      before compiled entry without state, graph, Inductor-cache, or FA4-key
      change
- [x] no more than one global decode FA4 application class is added
- [x] existing eager EXP-0016, no-cache EXP-0023, raw-compile negative, fixed,
      packed-varlen, mask, and backward behavior remains unchanged

## Synchronization and generated code

Not authorized until the first discriminator passes. No kernel source changes
in this experiment, so any later retained host/PTX/cubin/SASS signature must
match the accepted predecessor exactly.

## Measurement

Correctness-only. No timing, speedup, compiled-prefill, whole-model,
raw-`torch.compile(layer)`, local-cache, varlen-facade, training, B300, or
cross-architecture claim is authorized.

## Decision

**ACCEPT for the declared discriminator only.** On the pinned H100, candidate
2 exposed `L_cache_k_`, `L_cache_v_`, and `L_cache_length_` as explicit FX
placeholders and exposed no cache `get_attr`. Inductor invoked the user backend
once with zero graph breaks and one cache-aware custom op. The whole-layer
output was bitwise equal to the independently initialized eager twin; prepared
O differed from the FP32 reference by at most `0.015625`, FP32 LSE by at most
`3.814697265625e-05`, and compiled LSE was bitwise equal to the prepared FA4
result. Only root K/V slot 32 and the scalar counter changed, cache bytes
matched eager, every root/view address remained stable, and K/V storage stayed
distinct.

Altered position, foreign cache metadata, a rebound root, and a forged
transport view all rejected before compiled entry without changing cache
bytes, backend count, Inductor files, or FA4 keys. The K32 prefill application
class already covered K33, so the compiled call added zero FA4 keys. Pinned
inference tensors do not expose `_base` for views created outside inference
mode; complete alias coverage is therefore proven by frozen view/root object
identities plus equal shape, stride, storage offset, device, dtype, and storage
pointer rather than by `_base` identity.

Evidence is retained under `agent_space/remote-h100-exp0025/`. This decision
does not accept K1025, repeated decode, eager-compiler, local cache/rollover,
sanitizers, raw layer/full-model compilation, compiled prefill, performance,
or B300. Those require a separately predeclared widening.

Regression verification on the candidate source completed with `compileall`,
`ruff check .`, the offline model-contract verifier, and the complete local
suite (`415 passed, 105 skipped`). The complete H100 suite reported
`507 passed, 17 skipped, 1 xfailed`; the expected xfail remains the pinned
generic FA4 mask adapter's inability to encode the vision future-token
exception. The complete checksum manifest passed locally. The remote manifest
pre-check was not used as evidence because the intentional sync allowlist omits
seven older retained evidence files; the H100 source/test suite itself passed.

## Record

```bash
python scripts/record_result.py EXP-0025 \
  --kernel h100-explicit-static-cache-views \
  --arch sm_90 --decision accept \
  --hypothesis '<exact hypothesis above>' --bench <jsonl>
```
