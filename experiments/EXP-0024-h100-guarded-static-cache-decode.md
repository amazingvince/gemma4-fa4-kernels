# EXP-0024: H100 guarded compiled StaticCache decode

- Date / author: 2026-07-20 / Codex
- Status: **PREDECLARED — no candidate result yet**
- Kernel family: pinned Transformers local-d256 and global-d512 attention
  decode over the retained FA4 forward paths
- Architecture: sm_90
- Starting revision: `0781ceca2ee409ed23310a28423b4c1e95009d42`
- Model-contract lock SHA256:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Upstream Transformers revision: `7ea2320c76117e6742364808a666ef6f2fb40a67`
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- Transformers patch SHA256:
  `ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79`
- H100 FA4 patch SHA256:
  `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- Exemplars:
  - pinned Transformers `cache_utils.py::StaticLayer.update` and
    `StaticSlidingWindowLayer.update`;
  - pinned Transformers `Gemma4TextAttention` preparation/cache-update order;
  - accepted EXP-0016 eager active-prefix/rollover evidence;
  - accepted EXP-0023 guarded tensor-only compile facade.

## Prior boundary

EXP-0016 accepts eager B1 text-only `StaticCache` active-prefix prefill and
decode, including local rollover, stable storage, hostile-tail isolation, and
global K1025. It explicitly does not claim compiled cache execution.

EXP-0023 accepts a no-cache project-owned compile facade for actual pinned
layers 0/local and 5/global. It validates all live Python/module/config/weight
state outside Dynamo and compiles only tensor-explicit family functions. It
does not admit any cache object or cache tensor.

The pinned upstream cache implementation identifies decode as the intended
compiled region. Static storage must already be initialized: global
`StaticLayer` mutates fixed K/V backings and a tensor counter; local
`StaticSlidingWindowLayer` additionally owns a Python cumulative-length field
and rolls its fixed W1024 storage after it becomes full. Eager prefill remains
outside this experiment.

## Invariant changed

Add a separately named project-owned **decode-only** compiled-cache facade for
an actual early-initialized pinned `StaticCache`. The accepted eager prefill
produces the initial cache state. Each later call accepts exactly one new token
and must:

1. validate the live module/config/weights, cache/container/layer classes,
   initialization state, capacity, lengths, device/dtype/layout, stable K/V and
   counter addresses, distinct K/V storage, exact position, and unsupported
   arguments outside Dynamo and before mutation;
2. pass only explicit tensors plus a bounded local state class into the
   compiled frame;
3. reproduce the pinned projection, Q/K/V normalization, partial/full RoPE,
   cache write or local roll, lower-right local/global attention, and output
   projection order;
4. mutate only the intended K/V slot(s) and pinned counter state exactly once;
5. update the local Python cumulative length only after the compiled call
   succeeds, without using it as a compiler-visible source.

The compiled tensor ABI may use family-specific cache-aware whole-layer custom
ops with exact `mutates_args` declarations. Every module weight and cache
backing/counter is an explicit tensor argument; no module, cache object,
registry, mask callable, Python float, capacity, or payload identity is closed
over by the graph. The local underfilled/boundary and full/rolled states may be
separate bounded graph classes because they change cache mutation structure.
Exact sequence length, absolute position, capacity, payload, tensor address,
and cumulative value must remain runtime state rather than unbounded compile
keys.

No CuTe kernel, attention predicate, arithmetic tolerance, dependency, raw
`torch.compile(layer)`, no-cache EXP-0023 API, or eager EXP-0016 route changes.

## Declared envelope

- H100/SM90, pinned PyTorch 2.8, BF16, global inference/no-grad;
- actual pinned exemplar layers 0/local and 5/global;
- B1 text-only single-token decode (`Sq=1`) after an accepted eager prefill;
- exact contiguous absolute `position_ids` and no reset/document/vision/padding
  metadata;
- early-initialized, non-offloaded, non-quantized pinned `StaticCache` with
  stable CUDA storage and no crop/reorder/rewrite;
- local `StaticSlidingWindowLayer` underfilled, W1024 boundary, first rollover,
  and post-roll decode;
- global `StaticLayer` Q1/K33 and Q1/K1025 with capacity strictly larger than
  the active prefix;
- no active autograd, shared prepared KV, fallback, beam search, batched cache,
  chunked decode, compiled prefill, full-model compilation, or varlen facade.

Capacity overflow, malformed/nonmonotonic positions, empty or lazily
initialized caches, foreign cache/container/layer classes, altered storage,
aliased K/V, active vision/document metadata, padding, cache offload, and every
unsupported API request reject before compiled entry and before any cache or
module mutation.

## Hypothesis

An explicit decode-only facade that validates the actual pinned `StaticCache`
outside Dynamo and invokes a tensor-only cache-aware whole-layer custom op will
give stock PyTorch 2.8 Inductor exactly one backend attempt for global
Q1/K33, mutate exactly the same cache bytes/counters as the pinned eager layer,
and return a bitwise-identical whole-layer output while preserving all cache
addresses and adding at most one global decode FA4 application class.

Falsification is any second backend attempt; graph break; Python/module/cache
source in FX; cache mutation before validation or after a rejected request;
different written slot, counter, rollover order, or untouched byte versus the
pinned eager twin; changed storage address; K/V aliasing; non-bitwise whole
layer output; failed prepared O/FP32-LSE reference; more than the declared
bounded local/global graph or FA4 key classes; sanitizer finding; changed
retained FA4 object; tolerance increase; compiled prefill/full-model claim; or
wording that promotes the facade to raw `torch.compile(layer)` compatibility.

## Single change

1. Add an explicit `compile_gemma4_fa4_h100_static_cache_decode(...)` facade
   rather than widening raw `gemma4_fa4_compile_layer`.
2. Require an actual early-initialized pinned `StaticCache` and freeze its
   container/layer identities plus K/V/counter storage identities at facade
   construction; revalidate every live invariant before each call.
3. Add family-specific tensor-only cache-aware whole-layer custom ops whose
   mutation schemas name only the pinned cache K/V/counter tensors. Reproduce
   pinned global index-copy and local underfilled/roll behavior without reading
   a cache object inside the graph.
4. Keep eager prefill and all object/Python state transitions outside Dynamo.
   Advance the local Python cumulative length only after a successful compiled
   invocation and prove failure leaves it unchanged.
5. Extend a dedicated probe with backend/graph/compiled-entry counters, exact
   pre/post cache snapshots including content hashes and addresses, eager-twin
   comparison, hostile-tail isolation, cache-key inventory, and sanitizer mode.
6. Do not alter any retained FA4 kernel or the accepted EXP-0016/EXP-0023
   behavior.

## First discriminator

Run actual pinned layer 5 with two independently initialized weight-identical
global caches. Fill both eagerly to K32, then run one Q1/K33 decode through the
pinned eager layer and one through the Inductor facade. The candidate must:

- invoke the user backend exactly once with zero graph breaks;
- expose exactly one cache-aware whole-layer op node and tensor inputs only;
- be bitwise equal to the eager whole-layer output;
- match captured prepared Q/K/V, prepared O, and FP32 LSE under the frozen
  references;
- mutate only cache slot 32 and the cumulative counter, byte-for-byte equal to
  the eager twin;
- retain the cache K/V/counter addresses and distinct K/V storage;
- reject one altered position and one foreign cache before compiled entry with
  a byte-identical cache snapshot;
- add no more than one global decode application key.

Stop and reject at this discriminator before running local rollover or the
wider matrix if any item fails.

## Correctness gates

- [ ] construction rejects wrong module/config/cache/container/layer class,
      lazy initialization, offload/quantization, family/layer mismatch,
      training, altered weights, device/dtype/layout, K/V aliasing, or unstable
      storage before compilation
- [ ] global/Inductor Q1/K33 passes the complete first discriminator
- [ ] global/eager-compiler backend Q1/K33 passes the same bitwise and mutation
      contract with exactly one backend attempt
- [ ] global Q1/K1025 with physical capacity 1026 matches pinned eager and
      prepared O/LSE references while hostile unwritten tails remain invisible
- [ ] local underfilled Q1/K33, boundary Q1/K1024, first rolled Q1/K1024, and
      post-roll Q1/K1024 match pinned eager output, prepared operands/O/LSE,
      cache bytes, counters, and roll order
- [ ] all outputs are bitwise equal to the pinned eager twin; frozen FA4 versus
      FP32 prepared O/LSE policies remain unchanged
- [ ] repeated and reordered local/global decode sequences and a nondefault
      CUDA stream add no graph, Inductor, or FA4 application class
- [ ] cache K/V/counter addresses remain stable and K/V storage remains
      distinct across every accepted update
- [ ] hostile unwritten global capacity and local overwritten/old slots cannot
      affect output or LSE
- [ ] altered position, capacity overflow, B2, Q>1, cache reset/crop/reorder,
      metadata, padding, shared KV, fallback, active grad, foreign/lazy cache,
      aliased/rebound storage, and unsupported kwargs reject before compiled
      entry with byte-identical cache/module state
- [ ] graph classes are bounded exactly by family and the declared local
      mutation state; runtime position/capacity/length/payload/address values do
      not specialize a new class
- [ ] existing eager EXP-0016, no-cache EXP-0023, raw-compile negative, fixed,
      packed-varlen, mask, and backward matrices remain unchanged

## Synchronization and generated code

- [ ] unfiltered memcheck passes global Q1/K1025 and local first rollover
- [ ] project-kernel-filtered synccheck passes both cases, with any
      vendor-library output-projection finding retained separately
- [ ] project-kernel-filtered racecheck passes both cases
- [ ] retained local/global FA4 host object, PTX/cubin/SASS, resource signature,
      and launch geometry are unchanged
- [ ] compiled graph/cache entries, mutation-state classes, and FA4 application
      keys are inventoried separately and remain within the declared bound

## Measurement

Correctness-only. No timing, speedup, compiled-prefill, whole-model,
raw-`torch.compile(layer)`, varlen-facade, training, B300, or
cross-architecture claim is authorized.

## Decision

Pending. Implement and run only the global/Inductor Q1/K33 discriminator first.
Reject immediately on any graph, mutation, bitwise, reference, address, cache
key, or fail-closed violation. Run the local rollover and wider matrices only
after that discriminator passes without changing these gates.

## Record

```bash
python scripts/record_result.py EXP-0024 \
  --kernel h100-guarded-static-cache-decode \
  --arch sm_90 --decision <accept|reject|refine> \
  --hypothesis '<exact hypothesis above>' --bench <jsonl>
```
