# EXP-0050 retained H100 evidence

This directory contains the complete three-seed prospective comparison for
Gemma 4 12B full-parameter BF16 S512 training. `comparison.json` is the
predeclared aggregate result. Each seed directory retains its pure-SDPA and
all-FA4 reports, dataset provenance, environment preflight, and the historical
single-pair comparison artifact.

The controls record no project attention routes or native geometry. The
candidates record all expected local d256 and global d512 FA4 routes with
distinct K/V. `matrix.log` is retained because it records the two
orchestration-only follow-ups described in the experiment document; neither
changed or reran a training report in the accepted clean matrix.
