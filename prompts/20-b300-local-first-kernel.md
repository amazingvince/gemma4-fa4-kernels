# First implementation prompt — B300 local d256 forward correctness

Implement the smallest correctness-only B300 SM103 forward path for the Gemma
4 local text attention contract. No performance claim is allowed in this task.

Required context:

- `docs/model-contract.md`
- `docs/design.md` local section
- complete `writing-cute-dsl-kernels` version, architecture, layout,
  synchronization, correctness, and agent-workflow references
- closest pinned FA4 SM100 generic-local and dedicated-d256 exemplars

Contract:

```text
BF16 Q/K/V/O; FP32 score/LSE accumulation
B,S,32Q/16KV,d256 full shape plus smaller GQA-2 tests
scale=1.0
causal left window 1023
fixed length first
text-only first; vision boundary follows as a separate change
return O and FP32 LSE
```

Before code, complete a kernel design brief with the exact source anchor,
layouts, ownership, stage state machine, resources, cache key, and rejection
cases. Adapt the nearest exemplar. Add a failing test, establish the failure,
make the minimum implementation, then run compile, reference matrix,
memcheck/synccheck/racecheck, and generated-code inspection.

Compare one-CTA and two-CTA only after both are correct; that comparison is a
separate experiment.
