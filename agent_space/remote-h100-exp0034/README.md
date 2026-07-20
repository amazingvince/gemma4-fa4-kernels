# EXP-0034 retained H100 evidence

- `s128-admission.jsonl`: frozen output and separate dQ/dK/dV admission for
  FA4, automatic-GQA SDPA, and explicitly expanded SDPA.
- `global-s8k-*.jsonl`: hot/cold S8K CUDA-event distributions for FA4,
  automatic-GQA SDPA, and the canonical `sdpa_expanded` comparator.
- `global-s64k-*-screen.jsonl`: 2-warmup/5-repetition directional S64K screen.
- `sdpa-global-s8k-fwd_cuda_gpu_kern_sum.csv`: Nsight Systems CUDA kernel
  summary showing the automatic-GQA decomposition.

Environment: H100 80GB HBM3, driver 580.126.09, CUDA 12.8.93, PyTorch
2.8.0+cu128, CuTe DSL 4.6.0.dev0, unlocked clocks. The accepted FA4 patch was
verified exactly before measurement.
