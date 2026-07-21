# EXP-0047 through EXP-0049 retained H100 evidence

This compact bundle retains the machine-readable reports for the pure-SDPA
control, the rejected automatic packed-GQA all-FA4 route, the two 30-step
local-route discriminators, and the refined unpacked-GQA all-FA4 route.

The reports are complete Axolotl callback outputs. They include all per-step
losses, gradient norms, CUDA-event timings, parameter/dtype evidence, peak
memory, route counts, native prepared geometry, dataset provenance, and pinned
software revisions. `comparison-predeclared.json` remains a failure record;
the experiment documents explain both failed predeclared checks rather than
weakening them after measurement.
