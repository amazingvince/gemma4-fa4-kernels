# M0 remote baseline prompt

Target profile: <h100|b300>.

Use the project and CuTe skills. Verify the model contract and environment
before measuring. Create one baseline experiment record. Capture:

- GPU, compute capability, driver, CUDA toolkit, PyTorch runtime, CuTe DSL,
  Transformers/FA4 revisions, clock/power/temperature;
- fake-tensor compile and real-test results;
- current FA4 local d256 and global d512 behavior;
- fwd, bwd, and fwd+bwd timing regions separately;
- hot/cold L2 and cold/warm JIT distinctions;
- exact semantic limitations of every baseline.

Do not modify kernel code. Finish by proposing the smallest correctness-only
M1 task supported by the evidence.
