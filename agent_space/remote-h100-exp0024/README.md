# EXP-0024 retained H100 negative evidence

`first-discriminator-getattr-reject.json` is the unedited JSON emitted by the
candidate-6 probe on the pinned H100 environment. Its internal `status` field
means that the checks implemented by that probe revision passed; it is **not**
the experiment decision.

The subsequent manual contract audit found three forbidden FX nodes:

- `get_attr cache_k`
- `get_attr cache_v`
- `get_attr cache_length`

The cache tensors had been passed to the tensor-only Python function, but
PyTorch 2.8 lifted their upstream-marked static addresses into graph-module
buffers. That contradicts EXP-0024's requirement that every cache backing and
counter remain an explicit runtime tensor input and that no cache source be
closed over by FX. The Q1/K33 numerical, mutation, address, reference,
one-backend, zero-break, and fail-closed checks all passed, but this one ABI
falsifier is sufficient to reject EXP-0024 before any local or wider matrix.

The probe now treats those `get_attr` nodes as a hard failure. A later
experiment must predeclare and test a different tensor ABI; this artifact must
not be cited as accepted compiled-StaticCache support.
