# EXP-0026: H100 global StaticCache decode envelope

- Date / author: 2026-07-20 / Codex
- Status: **PREDECLARED — no candidate result yet**
- Kernel family: pinned Transformers global-d512 attention decode over the
  retained fixed/rectangular FA4 forward paths
- Architecture: sm_90
- Starting revision: `cd94bca5665f1fe54a0d1d191de690ca6ed1a9e7`
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

EXP-0025 accepts only pinned global layer 5, B1/Q1/K33 BF16
inference/no-grad Inductor decode after eager K32 prefill. It proves the
full-storage view ABI: K, V, and counter are explicit FX placeholders rather
than cache `get_attr` sources, and the actual root cache mutation is bitwise
equal to eager.

It does not accept a second decode, K1025, the eager compiler backend,
nondefault streams, hostile long-cache tails, sanitizers, or generated-code
stability. Local sliding-window cache mutation remains explicitly out of scope.

## Invariant changed

Widen only the accepted global full-storage-view runtime envelope. Keep the
same module/cache classes, custom-op schema, projection/norm/RoPE/update order,
root/view validation, one-token ABI, and compiler options. Admit:

- sequential K33 then K34 decode on the same facade;
- independent Q1/K1025 decode after eager K1024 prefill with physical capacity
  1026 and one hostile unwritten tail slot;
- stock `eager` and `inductor` compiler backends;
- repeated seed/payload order and one nondefault CUDA stream replay.

For active K through 1024, the opaque body must reproduce the accepted eager
composed fixed route (left-zero-padded Q followed by the final row). At K1025
it must reproduce the accepted direct lower-right forward-only route. Exact
active length, capacity, position, payload, addresses, and stream remain
runtime state and must not become Python/cache sources in FX.

No local cache, kernel source, tolerance, dependency, raw compile hook,
compiled prefill, full model, training, performance, or B300 behavior changes.

## Hypothesis

The accepted explicit-view global custom op will preserve one graph per facade
and exact eager cache/output semantics across K33/K34 and K1025 under both
stock compiler backends and a nondefault stream, while hostile spare capacity
remains invisible and the retained FA4 host/code objects remain unchanged.

Falsification is a second graph for one facade; graph break; cache `get_attr`;
more than one cache op; eager/Inductor numerical or mutation disagreement;
hostile-tail influence; changed root/view address; non-bitwise whole-layer
output; prepared O/LSE policy failure; unexpected FA4 application class;
sanitizer finding; changed retained kernel object/code/resource signature; or
any local/performance/B300 claim.

## Single change

1. Generalize the EXP-0025 probe over backend, prompt length, capacity, replay
   count/order, stream, and hostile tail without changing the product ABI.
2. Run K33 then K34 on one facade for each backend and prove one graph, stable
   compiled entry, exact per-step mutation, and bounded FA4 keys.
3. Run independent K1025/capacity1026 clean and hostile candidates against a
   pinned eager twin and FP32 prepared reference.
4. Run the global K1025 custom-op case under memcheck, filtered synccheck, and
   filtered racecheck.
5. Re-capture the retained FA4 host object, PTX/cubin/SASS hash, resource
   signature, and launch geometry and require equality to the accepted global
   predecessor.

## Correctness gates

- [ ] eager and Inductor K33/K34 sequential decode use one backend graph per
      facade with zero breaks, one cache op, explicit cache placeholders, and
      zero cache `get_attr`
- [ ] every K33/K34 whole-layer output is bitwise equal to its eager twin and
      every cache slot/counter mutation is byte-for-byte equal
- [ ] Inductor and eager-compiler Q1/K1025/capacity1026 match pinned eager,
      prepared O, and FP32 LSE policies
- [ ] hostile global tail slot 1025 cannot affect prepared O/LSE, projected
      output, active cache bytes, graph, or application keys
- [ ] repeated seeds, reordered cases, and one nondefault CUDA stream add no
      graph or application class beyond the declared global bound
- [ ] root/view K/V/counter addresses stay stable; K/V remain distinct; exact
      active slots and counter alone mutate
- [ ] malformed position, capacity exhaustion, B2, Q>1, metadata, active grad,
      reset/rebind/forged view, foreign/lazy/offloaded cache, and unsupported
      kwargs reject before compiled entry with byte-identical state
- [ ] existing eager EXP-0016, no-cache EXP-0023, explicit-view EXP-0025,
      raw-compile negative, fixed, packed-varlen, mask, and backward matrices
      remain unchanged

## Synchronization and generated code

- [ ] unfiltered memcheck passes global Q1/K1025
- [ ] project-kernel-filtered synccheck passes global Q1/K1025; any known
      vendor-library output-projection report remains separated
- [ ] project-kernel-filtered racecheck passes global Q1/K1025
- [ ] retained global FA4 host object, PTX/cubin/SASS hashes, resource
      signature, and launch geometry match the accepted predecessor
- [ ] Dynamo graph, Inductor files, mutation views, and FA4 application keys
      are separately inventoried and stay within the declared bound

## Measurement

Correctness-only. No timing, speedup, compiled-prefill, whole-model,
raw-`torch.compile(layer)`, local-cache, varlen-facade, training, B300, or
cross-architecture claim is authorized.

## Decision

Pending. Widen only the global runtime/evidence matrix and stop before local
cache work if any correctness, graph, mutation, sanitizer, or codegen gate
fails.

## Record

```bash
python scripts/record_result.py EXP-0026 \
  --kernel h100-global-static-cache-envelope \
  --arch sm_90 --decision <accept|reject|refine> \
  --hypothesis '<exact hypothesis above>' --bench <jsonl>
```
