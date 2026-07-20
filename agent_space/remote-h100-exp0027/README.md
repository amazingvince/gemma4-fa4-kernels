# EXP-0027 retained H100 rejection evidence

Candidate revision `e0179fe6093bc95f8d270d0ab30d26bfc77f7d96` was run on
the pinned RunPod H100 environment with PyTorch 2.8.0+cu128 and CUDA 12.8.

The first K1024 boundary call completed. The first saturated rollover call at
absolute position 1024 then failed the predeclared version gate: the CUDA
counter stayed bytewise equal to 1024 but its tensor version changed from 3 to
5 because it was declared mutable in the opaque custom-op schema. The pinned
upstream saturated branch does not mutate this tensor. EXP-0027 is therefore
rejected without widening its claims.

`first-discriminator-reject.json` records the exact exception and scope.
