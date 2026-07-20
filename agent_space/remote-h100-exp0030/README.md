# EXP-0030 rejected H100 candidate

This directory retains the two hot-L2 screening records for
`dQ_single_wg=True` on the exact global S8K workload. The candidate passed the
S128 O/LSE/dQ/dK/dV reference probe for three nondefault-stream repetitions,
but improved median backward time by only 0.19% and combined time by only
0.42%. Both candidate IQRs overlap their EXP-0029 baselines.

The predeclared acceptance threshold was 3% with non-overlapping IQRs. The
candidate was therefore rejected before S64K, cold-L2, sanitizer, or generated
code confirmation, and the authoritative patch plus H100 checkout were
restored byte-for-byte to SHA256
`eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`.
