# H100 production long-context soak design brief

EXP-0044 adds a reproducible validation harness for the already accepted H100
routes. It changes no kernel, dispatch rule, tolerance, or default flag and
makes no new speedup claim.

## Hypothesis and scope

Repeated fresh-process execution at the locked production boundaries will
retain the accepted tensor/gradient contracts, stable FA4 cache classes, and
release all process-owned GPU memory after each case. The bounded matrix is:

- default global square S65536 forward+backward, hot-L2, two warmups and five
  measured repetitions;
- native global lower-right Q1/K262144 analytic finite-score backward, three
  same-input repetitions on a nondefault stream, differentiating O and LSE;
- native local packed Q=K=262144 forward+backward in three fresh processes and
  seeds on nondefault streams.

The global square case is a stability/timing diagnostic against the accepted
EXP-0038/EXP-0040/EXP-0041 stack, not a new comparative performance result.
The asymmetric global and local maximum-context cases reuse their accepted
analytic/contract validators rather than attempting an intractable dense
reference.

## Harness contract

- Acquire the repository H100 lease around the entire harness.
- Run each case as a child process so allocator and compiled graph state cannot
  leak between independent cases.
- Save complete stdout/stderr and a JSON summary with exact commands, elapsed
  wall time, return code, and post-child `nvidia-smi` compute-process state.
- Fail on any child error, benchmark skip/error, nonempty post-child compute
  process, non-finite output/gradient contract, or maximum-context admission
  failure.
- Keep the exact EXP-0041 patch stack and result schema unchanged.

## Rollback

The harness and evidence can be removed without changing runtime code or the
upstream patch.
