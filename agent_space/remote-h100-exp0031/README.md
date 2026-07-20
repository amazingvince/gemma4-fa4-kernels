# EXP-0031 compile rejection

The full-V512 M64xN16 dQ idea was first instantiated with the accepted two MMA
warpgroups. CuTe compilation stopped before code generation because N16 splits
into an N8 accumulator partition per warpgroup. The pinned QuACK
`reshape_acc_to_frgA` path requires an even accumulator-atom count to construct
the K16 dS operand and rejected this isolated N8 half.

No kernel launched and no correctness, sanitizer, resource, or performance
claim was made. The candidate patch SHA256 was
`ce2ace952addbfe8e581afcf75768701a2e676542c81c2d0572f0ce69ff7a1a7`.
The repository and H100 checkout were restored to the accepted EXP-0029 patch
SHA256 `eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`.
