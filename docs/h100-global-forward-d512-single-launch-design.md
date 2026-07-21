# H100 global D512 single-launch forward design brief

This is the pre-implementation gate for EXP-0041. It targets only the accepted
H100 global-causal BF16 forward contract and retains the exact two-V256-launch
EXP-0002 composition as an immediate rollback.

## Scope and invariants

- SM90a, BF16, 32 Q heads, 4 KV heads, GQA 8, Q/K/V/O dimension 512.
- Scale is exactly 1.0; K and V remain distinct prepared operands.
- Fixed BSHD and native packed THD preserve lower-right causal alignment and
  their current length envelopes.
- Backward remains the accepted EXP-0038/39/40 implementation. This experiment
  changes only the forward launch selected by those autograd wrappers.

## Accepted structure

The first M128 x N32 candidate was rejected at compile: SM90 WGMMA limits the
PV N-mode to 256. The accepted refinement uses M64 x N32 and two consumer
warpgroups. WG0 owns QK, online softmax, and O-low; WG1 owns O-high. WG0 stores
BF16 P and FP32 online-rescale factors in shared memory, and a pair of named
barriers makes the handoff explicit before both disjoint PV MMAs proceed.

The concrete mainloop shared-memory payload is:

```text
Q: M64 x D512 BF16              =  64 KiB
K: two N32 x D512 BF16 stages   =  64 KiB
V: two N32 x D512 BF16 stages   =  64 KiB
P: M64 x N32 BF16               =   4 KiB
row-scale exchange              =   4 KiB
core payload                    = 200 KiB
```

Alignment and barrier storage produce an observed 205,824-byte dynamic launch.
The fixed and packed objects use 168 registers/thread, 1 KiB static shared
memory, and zero stack/local memory. Each consumer retains only one O256 FP32
accumulator, avoiding both the illegal D512 WGMMA and a full-D register spill.

## Gates and rollback

- Fake and real compile before any dispatch change; record registers, local
  memory, dynamic shared memory, and generated launch count.
- Fixed and packed boundary/tail/rectangular reference matrices, exact LSE,
  repeated runs, GQA ownership, distinct K/V, and nondefault stream.
- Memcheck, synccheck, and racecheck for fixed and packed cases.
- Hot-L2 S8K and S64K forward measurements against the unchanged exact
  two-launch composition. Reject if the candidate is slower at either gate or
  if IQRs do not support the claimed direction.
- `FLASH_ATTENTION_GEMMA4_EXPERIMENT_FORWARD_D512_SINGLE_LAUNCH=0` is the
  tested explicit rollback. The accepted cooperative route is the default.
