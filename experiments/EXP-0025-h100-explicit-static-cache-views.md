# EXP-0025: H100 explicit StaticCache view ABI

- Date / author: 2026-07-20 / Codex
- Status: **PREDECLARED — no candidate result yet**
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

- [ ] view/root identity, storage, shape, stride, offset, version, device,
      dtype, and complete-coverage checks fail closed
- [ ] global/Inductor Q1/K33 has one backend attempt and zero graph breaks
- [ ] graph contains one cache custom op, three explicit cache tensor inputs,
      and zero cache `get_attr` sources
- [ ] whole-layer output is bitwise eager-equal; prepared O and FP32 LSE retain
      their frozen reference policy
- [ ] only root K/V slot 32 and the root counter change, byte-for-byte equal to
      the eager twin, with all root/view addresses stable and K/V distinct
- [ ] malformed position, foreign cache, rebound root, and forged view reject
      before compiled entry without state, graph, Inductor-cache, or FA4-key
      change
- [ ] no more than one global decode FA4 application class is added
- [ ] existing eager EXP-0016, no-cache EXP-0023, raw-compile negative, fixed,
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

Pending. Implement only the full-storage-view ABI and rerun the global
Inductor Q1/K33 discriminator with the strengthened graph audit.

## Record

```bash
python scripts/record_result.py EXP-0025 \
  --kernel h100-explicit-static-cache-views \
  --arch sm_90 --decision <accept|reject|refine> \
  --hypothesis '<exact hypothesis above>' --bench <jsonl>
```
