# Global d512 design-spike prompt

Produce architecture-specific design briefs for H100 SM90a and B300 SM103.
Do not implement code in this session.

Operation:

```text
O = softmax(Q K^T, scale=1.0, causal) V
Q: 32 heads × 512
K,V: 4 heads × 512, distinct prepared operands
BF16 I/O, FP32 accumulation and LSE
GQA ratios 1/2/4/8 supported by the family
```

For each architecture, derive and compare at least two feasible schedules.
Show register/SMEM/TMEM arithmetic, output-D ownership, K and V staging,
softmax statistics exchange, pipeline state, epilogue, and backward ownership.

H100 must address why one warpgroup cannot safely own an M64×D512 FP32 output.
B300 must compare one-CTA slab accumulation with two-CTA D splitting and explain
how QK is shared or partitioned without silently changing the operation.

Use MLA and large-head kernels only as mechanics/provenance evidence. Reject
any design that assumes prepared K equals V or that combines dK and dV before
preparation adjoints.
