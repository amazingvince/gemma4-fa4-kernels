# EXP-0026 retained H100 evidence

This directory records the accepted global layer-5 StaticCache decode envelope
at source revision `b5b8ecf1888c519c99142f14c375ca887caa6891` on the pinned
H100 environment.

`global-envelope-default.json` and `global-envelope-reverse.json` run different
seeds and opposite case orders. Both pass eager and Inductor sequential
K33/K34 decode plus independent K1025/capacity1026 decode. They retain one
semantic graph class, two physical-capacity shape signatures, zero graph
breaks, one cache custom op, three explicit cache placeholders, no cache
`get_attr`, bitwise eager whole-layer output/cache mutation, stable distinct
K/V storage, and an unchanged hostile unwritten tail. The largest prepared
output absolute error is 0.015625 and the largest FP32 LSE absolute error is
0.0000762939453125, within the frozen global policies.

`exp0025-regression.json` reruns the inherited guarded-view discriminator at
the same revision. It proves malformed position, foreign-cache metadata,
rebound root storage, and a forged transport view still reject before the
compiled entry with byte-identical cache state. The EXP-0026 matrices add B2,
Q2, metadata, active-grad, lazy-cache, offloaded-cache, and exhausted-capacity
rejections.

The `sanitizers/` directory contains the exact Inductor Q1/K1025 custom-op
case. Unfiltered memcheck reports zero errors. Project-kernel-filtered
synccheck reports zero errors, and filtered racecheck reports zero hazards.
Each companion JSON also passes the graph, mutation, eager, and prepared
reference assertions.

The strict H100 environment check has zero warnings and zero errors. At this
revision, local compileall and Ruff pass, the complete local suite reports
`415 passed, 105 skipped`, and the complete H100 suite reports `507 passed,
17 skipped, 1 xfailed`. The expected xfail remains the pinned Transformers
generic FA4 mask adapter's inability to encode the local vision future-token
exception.

`codegen-inventory.json` records bitwise equality to the accepted EXP-0023
global host object, PTX, inner cubin, SASS, and resource signatures. The
retained kernel still uses 168 registers, zero stack/local bytes, 1024 bytes
of static shared memory, and no SASS `LDL`/`STL`. `codegen/launch-grid.csv`
is the Nsight Systems launch inventory: the K1025 project kernel uses grid
`(32,1,1)` and block `(384,1,1)`, while the separately visible K1024 prefill
uses grid `(256,1,1)` and the same block. The diagnostic instance counts
include candidate, eager-twin, and prepared-reference calls; they are not a
performance measurement.

No evidence here accepts local StaticSlidingWindow cache mutation, compiled
prefill, raw or whole-model `torch.compile`, training, performance, or B300.
