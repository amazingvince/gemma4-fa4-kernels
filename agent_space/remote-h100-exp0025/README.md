# EXP-0025 retained H100 evidence

`first-discriminator.json` is the successful strengthened Q1/K33 discriminator
for the explicit full-storage-view ABI on the pinned H100 environment.

The captured Inductor graph has exactly one cache-aware custom op, one backend
attempt, zero graph breaks, explicit K/V/counter placeholders, and no
cache-related `get_attr`. The compiled output is bitwise equal to an
independently initialized pinned eager layer; only root cache slot 32 and the
root scalar counter change, byte-for-byte like eager, with stable root/view
addresses and distinct K/V storage. Prepared O and FP32 LSE retain the frozen
reference tolerances. Altered position, foreign cache metadata, a rebound root,
and a forged transport view all reject before compiled entry with unchanged
cache, graph, Inductor-cache, and FA4 application-key state.

This evidence accepts only global layer 5, B1/Q1/K33, Inductor, BF16,
inference/no-grad decode after eager prefill. It does not establish local
StaticCache, K1025, repeated decode, eager-compiler, sanitizer, raw
`torch.compile(layer)`, compiled prefill, full-model, performance, or B300
support.

Candidate regression gates also passed: the complete local suite reported
`415 passed, 105 skipped`, and the complete H100 suite reported `507 passed,
17 skipped, 1 xfailed`. The expected xfail is the already documented pinned
generic FA4 mask-adapter limitation for the vision future-token exception.
