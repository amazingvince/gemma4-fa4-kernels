# EXP-0028 retained H100 evidence

This directory records the refined local layer-0
`StaticSlidingWindowLayer` decode envelope at product-source revision
`829dc5bf2691d47349f8e94ebbe616c260740569` on the pinned H100
environment. The discriminator, default/reverse matrices, independent
negative matrix, three sanitizer JSONs, and EXP-0023/0025/0026 regression
JSONs were rerun after canonical formatting and recollected at final source
`ebcc1fc2623e565991151d9beeea29cbe17bfbd8`; their isolated cache paths carry
that revision label.

`first-discriminator.json` passes absolute positions 1023, 1024, and 1025
through one Inductor graph. K/V/cache outputs match a weight-identical eager
twin bitwise; the CUDA counter changes bytes and version only at the boundary
fill, remains saturated during both rolls, and the eager Python count advances
once after each successful transaction. The graph has one local cache op, 13
tensor plus four symbolic placeholders, zero `get_attr`, and zero graph breaks.

`local-envelope-default.json` and `local-envelope-reverse.json` use different
seeds and opposite orders. Each passes six cases: eager and Inductor K33/K34,
hostile-tail K33/K34, one nondefault stream, and eager and Inductor K1024
boundary plus two rolls. They retain one semantic/runtime graph signature and
the accepted native packed-varlen decode application key
`e7b213f0ae59536df7feec9f0202f6cdace2105999b143dc3c133cbda041f176`.
The largest prepared-output absolute error is 0.015625 and the largest FP32
LSE absolute error is 0.00002288818359375, within the frozen local policies.

The final default matrix contains 16 pre-entry rejections: B2, Q2, metadata,
wrong position, active grad mode, requires-grad input, tensor/Python counter
mismatch, lazy/offloaded/foreign/reset cache, rebound root, forged compiled
view, wrong capacity, and wrong layer class. Every rejection leaves cache
bytes/state unchanged and adds no graph, compiled entry, or FA4 application.
`negative-matrix.json` is the independently rerun negative-only evidence.

The `sanitizers/` repeated-roll case passes unfiltered memcheck with zero
errors, project-kernel-filtered synccheck with zero errors, and filtered
racecheck with zero hazards/errors/warnings. Every companion JSON also passes
the graph, eager/cache, counter, and prepared-reference assertions.

`codegen-inventory.json`, `codegen/resources.txt`, `codegen/launch-grid.csv`,
and `codegen/launch-probe.json` retain compact generated-code evidence. The
native decode host object is bitwise equal to the accepted EXP-0016 baseline.
The extracted cubin uses 168 registers, zero stack/local bytes, 1024 bytes of
static shared memory, HGMMA/TMA/barrier instructions, and no LDL/STL. Raw
fatbin/cubin/SASS/Nsight artifacts remain ignored scratch files.

`exp0023-regression.json`, `exp0025-regression.json`, and
`exp0026-regression.json` rerun the inherited guarded no-cache, explicit
global cache-view discriminator, and full global compiled-cache envelopes at
the current evidence revision; all pass. `h100-check.json` records the pinned
H100 environment, exact upstream revisions and managed patch hashes, and has
empty warnings and errors. Complete repository verification reported
`415 passed, 106 skipped, 8 warnings` locally and
`508 passed, 17 skipped, 1 xfailed, 8 warnings` on the H100. The retained
xfail is the documented generic Transformers FA4 mask limitation.

No evidence here accepts compiled prefill, B>1/Q>1, cached vision/document
metadata, training/backward through the facade, other layer indices, raw or
full-model compilation, performance, or B300.
