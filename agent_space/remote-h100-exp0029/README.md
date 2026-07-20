# EXP-0029 retained H100 evidence

This directory contains the reproducible H100 performance ruler for the exact
Gemma 4 project adapters at product source
`86444608855a2baf9951d9c1bde92167618f3a61`.

## Method

- NVIDIA H100 80GB HBM3, SM90, 132 SMs
- driver 580.126.09; CUDA 12.8; PyTorch 2.8.0+cu128
- CuTe DSL 4.6.0.dev0; QuACK 0.5.3
- unlocked clocks, explicitly recorded in `h100-check.json` and
  `roofline-unlocked.json`
- ten warmups and thirty CUDA-event repetitions per point
- median, p25, p75, and IQR
- hot L2 and a cold mode using a 209,715,200-byte thrash buffer
- fwd, bwd, and fwd_bwd timed separately
- direct GPU input generation and an untimed correctness admission check

The measured roofline denominators were 780.646 BF16 TFLOP/s and 3.02456 TB/s
aggregate HBM read+write. These denominators and all timings are specific to
this unlocked pod run and are not general H100 specifications.

## Result

The hot-L2 records show global d512 attention dominates the model-weighted
runtime, especially backward. A one-repetition Nsight Systems trace of global
S8K fwd+bwd attributes 94.512 ms to the six backward mainloop launches. The
two dKV launches total approximately 22.231 ms; the four dQ launches total
approximately 72.281 ms. The trace is retained for launch attribution only and
is not used as the benchmark timing result.

`nsys-global-s8k-fwd-bwd.csv` is the portable stats export. The `.nsys-rep`
and `.sqlite` files are retained locally but need not be committed when the
portable export is sufficient. NCU counters could not be collected because
the pod denies access with `ERR_NVGPUCTRPERM`.

## Files

- `h100-check.json`: strict environment and idle-state preflight
- `roofline-unlocked.json`: measured denominators and warm clock sample
- `smoke-fa4-*.jsonl`: exact hot/cold fwd, bwd, and fwd_bwd results
- `nsys-global-s8k-fwd-bwd.csv`: Nsight Systems kernel summary and trace rows
- `global-s8k-fwd-bwd.nsys-rep`: local native trace report
- `global-s8k-fwd-bwd.sqlite`: local trace database
