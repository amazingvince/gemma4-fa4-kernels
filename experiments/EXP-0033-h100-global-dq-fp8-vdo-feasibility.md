# EXP-0033: H100 global dQ FP8 V/dO feasibility

- Date / author: 2026-07-20 / Codex
- Kernel family: global-d512-dq
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `configs/env/h100-compatible.env`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Design brief: `docs/h100-global-dq-fp8-feasibility.md`

## Invariant changed

None in the accepted kernel. The proposed opt-in research path would stage V
and dO as explicitly scaled FP8 for dP only, while retaining BF16 Q/K/dS and
FP32 accumulation. It would intentionally make dQ approximate.

## Hypothesis

An M64xN32 two-warpgroup dQ kernel with FP8 V/dO can stage the full V512
reduction in 218,112 bytes and reduce end-to-end global S8K backward time by at
least 10%, while meeting human-approved dQ error and saturation thresholds.

## Single change

Feasibility and design only. No kernel or default API change is authorized.

## Evidence

- [x] pinned CUTLASS Hopper example supports FP8 E4M3FN inputs with FP32 MMA
  accumulation
- [x] pinned FA4 interface explicitly does not support FP8 backward
- [x] shared-memory byte budget written down
- [x] exact BF16 default, distinct dK/dV, scale=1.0, and masks remain untouched
- [ ] human approval of approximate-gradient contract and scale metadata
- [ ] representative V/dO range and saturation measurements
- [ ] FP8 dP micro-kernel compile and SASS proof
- [ ] numerical, sanitizer, memory, and performance gates

## Decision

HOLD FOR HUMAN REVIEW

The idea is plausible and isolated, but using FP8 does more than solve a
storage problem: it changes dQ numerics and the public backward contract. The
exact BF16 V/dO slot-time-sharing experiment has priority unless an approximate
gradient is explicitly acceptable.
