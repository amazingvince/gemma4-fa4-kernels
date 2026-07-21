# H100 global deterministic-backward dispatch design brief

This is the pre-implementation ownership and synchronization gate for
EXP-0039. It adds an explicit deterministic backward route without changing
the accepted EXP-0038 fast default. It is a prerequisite for, not a substitute
for, the later bounded-memory owner-computes kernel.

## 1. Scope and invariants

- Target: H100 SM90a, exact BF16 global-causal Gemma 4 attention.
- Geometry: 32 Q heads, 4 KV heads, GQA 8, Q/K/V/D=512, scale exactly 1.0.
- Inputs and outputs remain distinct prepared Q/K/V and separate dQ/dK/dV.
- Fixed BSHD and native packed THD use the existing admitted length envelopes.
- Forward is unchanged. Only backward scheduling and reduction order change.
- `deterministic=False` remains the EXP-0038 two-main-launch default.
- `deterministic=True` is explicit and belongs in every compile/application
  cache key. Runtime lengths and cumulative values do not.

## 2. Route composition

The deterministic route deliberately reuses already-reviewed kernel families:

1. Two V256 dKV-only launches use the accepted pre-EXP-0037 slab route with
   FA4's deterministic GQA dK/dV semaphore protocol.
2. One EXP-0038 full-D dQ launch computes score/dP/dS once and emits low/high
   D256 dQ slabs sequentially.
3. The existing FP32 postprocess converts each complete accumulation tensor
   once to BF16.

This route has three main launches. The extra dKV launch is accepted only for
the opt-in deterministic contract; the fast default remains two launches.

## 3. Ordered ownership

The kernel grid remains K-major. Determinism comes from a fixed reduction
order, not assumed CTA launch order.

- dQ semaphore shape: `(batch, q_head, ceil(max_q / 64), 1)`.
- dK and dV semaphore shape: `(batch, kv_head, ceil(max_k / 32), 2)`.
- Each dQ M64 owner waits for its predecessor N32 tile, performs both low and
  high bulk reductions, drains the second reduction, then increments once.
- Each dKV N32 owner serializes the eight contributing Q heads using the
  upstream two-phase dK/dV semaphore protocol.
- Packed segments use independent batch-indexed semaphore rows. Empty-query
  segments schedule no dQ work and cannot advance another segment's lock.

For the two-slab dQ epilogue, deterministic serialization does not prove that
the low-slab bulk engine has released shared memory before high-slab reuse.
The store warp must therefore retain the explicit bulk wait before publishing
the high slab, then drain both slabs before releasing the N-tile semaphore.

## 4. Memory and cache contract

This experiment still uses whole-sequence FP32 dQ/dK/dV accumulation. It adds
only bounded INT32 semaphores, at most:

```text
4 * batch * (32 * ceil(max_q / 64) + 2 * 4 * ceil(max_k / 32)) bytes
```

The preflight estimator includes these bytes. A deterministic key is distinct
from every fast key, but fixed/packed SS/SM/MM scheduler classes remain
bounded. The later owner-computes experiment must remove the whole-sequence
FP32 requirement; EXP-0039 makes no such claim.

## 5. Gates and rollback

- Fixed S1/31/32/33/63/64/65/127/128/129 references and five exact repeats.
- Packed tiny/mixed/reversed/mixed-empty references, isolation, and exact
  repeated dQ/dK/dV.
- O, FP32 LSE, dO-only/LSE-only/combined gradients, and GQA ownership.
- Fixed and packed memcheck, synccheck, and racecheck.
- Generated resources, semaphore traffic, three-main-launch count, and bounded
  compile-cache inventory.
- S8K and S64K timing is characterization, not a speedup claim. Reject a
  deadlock, any non-bitwise repeated gradient, numerical regression, spill
  growth, cache-key explosion, or more than 25% S8K slowdown versus the
  explicit nondeterministic default.

Rollback is immediate: omit `deterministic=True`. The default then remains the
accepted EXP-0038 route and no deterministic objects are compiled.
