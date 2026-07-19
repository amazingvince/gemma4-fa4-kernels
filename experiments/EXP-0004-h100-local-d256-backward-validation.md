# EXP-0004: H100 local d256 backward validation

- Date / author: 2026-07-19 / Codex
- Kernel family: local-d256-bwd
- Architecture: sm_90
- Upstream FA4 revision: `77aacb68d194ba9af1010eda5eac3e7c0df8e6f6`
- CuTe DSL / CUDA / PyTorch: `4.6.0.dev0` / 12.8 / `2.8.0+cu128`
- QuACK runtime helpers: `quack-kernels==0.5.3`
- Model-contract lock hash:
  `a8cdde81ab6d965b94423d88f0dca1039a0328f9ed0611059308dc4c960a4dbe`
- Environment-policy hash:
  `1e1767ac7a87ba5503a97aa38bbefcccef5fa88e9793f330bdd9b3658fcceb75`
- Strict environment artifact: `agent_space/h100-check-precommit.json`
- Exemplar path and revision: pinned `tests/cute/test_flash_attn.py`,
  `flash_attn/cute/testing.py`, `flash_attn/cute/interface.py`, and
  `flash_attn/cute/flash_bwd_sm90.py`

## Invariant changed

No model or kernel invariant changes. Keep BF16 BSHD inputs, B=1, exact
32Q/16KV GQA-2 geometry, d=256, causal W1024, scale 1.0, distinct K/V, and
separate dQ/dK/dV. This experiment changes only the validation policy used to
interpret expected BF16 numerical error. It does not alter or retroactively
accept the frozen EXP-0003 envelope.

## Hypothesis

The existing pinned SM90 M64 x N64, Q1/dO1/PdS1 local-d256 backward is
semantically correct over the declared fixed-length matrix: for each of dQ,
dK, and dV, its maximum absolute error against the FP32-accumulating reference
is no greater than twice the maximum error of an independent PyTorch BF16
attention path against that same reference, plus the reference's measured
BF16 quantization floor. It also returns finite, separate gradients on the
default and a nondefault CUDA stream without sanitizer findings.

This is the accuracy policy used by the pinned upstream CuTe tests. For a
candidate gradient `g`, high-precision gradient `g_ref`, and independent BF16
gradient `g_pt`, the predeclared gate is:

```text
max_abs(g - g_ref) <= 2 * max_abs(g_pt - g_ref) + quantization_atol
quantization_atol = 2 * max_abs((g_ref + 0.3 - 0.3) - g_ref)
```

All max and mean errors are recorded per gradient. EXP-0003's
`atol=0.125, rtol=0.05` result remains a rejection and is still printed as a
diagnostic; it is not weakened or reused as this experiment's decision rule.

## Diagnostic basis (not acceptance evidence)

The already-authorized EXP-0003 S128 input was re-evaluated only to choose a
falsifiable next hypothesis. An initial repository-BF16 diagnostic used the
same `logsumexp`/`exp` operator order as the high reference and produced overly
large baseline maxima `15.40625`, `13.84375`, and `0.625`. Review caught that
this did not mirror pinned upstream's independent `torch.softmax` path, so it
was discarded before acceptance. The corrected S128 BF16 baseline maxima were
`5.75`, `3.125`, and `0.21875` against candidate dQ/dK/dV maxima
`0.28857421875`, `0.375`, and `0.0625`. Upstream still marks SM90 backward
above d192 unsupported, so the full corrected matrix and hardware checks below
remain mandatory.

## Single change

Extend the existing backward probe with an explicit `upstream-relative`
comparison mode. The independent BF16 path must execute attention itself; it
must not call the candidate or reuse candidate O/LSE. Preserve the existing
`frozen` mode byte-for-byte in meaning so EXP-0003 remains reproducible.

Do not change the SM90 tile, stages, atom layouts, accumulation types, masks,
or public H100 adapter in this experiment. A read-only fake probe found that
an M64 x N48 Q2/dO1/PdS1, `dKV_swapAB=true` main kernel can compile, but its
current dK/dV postprocess rejects the reversed tiler with M-mode 128. Any such
follow-up would therefore require paired main-kernel/postprocess layout work;
it is not a config-only fallback and is not authorized here.

## Correctness matrix

Run seeds `3000 + S` at S=`1, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025`.
Each case must check:

- exact B1/S/32Q/16KV/d256 BF16 shapes and scale 1.0;
- distinct K/V inputs and distinct dQ/dK/dV output storage;
- finite dQ, dK, and dV;
- the predeclared upstream-relative maximum-error gate for each gradient;
- per-gradient candidate and BF16-baseline max/mean error reporting.

Then run S128 three times on the same inputs and once on a nondefault CUDA
stream. Every result must independently pass the same gate. Exact repeat
equality is reported but is not an acceptance requirement because GQA dK/dV
may use unordered FP32 accumulation.

- [x] exact fake-tensor backward compile
- [x] declared S matrix
- [x] separate finite dQ, dK, and dV
- [x] repeated-run check
- [x] nondefault-stream check

The fake compile returned exact BF16 shapes `dQ=(1,128,32,256)`,
`dK=(1,128,16,256)`, and `dV=(1,128,16,256)`. The real matrix passed every
upstream-relative gate. Cells below report candidate max/mean absolute error,
then independent-BF16 baseline max/mean error, then the resulting maximum
allowed error.

| S | dQ candidate; baseline; limit | dK candidate; baseline; limit | dV candidate; baseline; limit |
|---:|---|---|---|
| 1 | `0.00002277/0.00000187; 0/0; 0.001564` | `0.00002480/0.00000287; 0/0; 0.001564` | `0/0; 0/0; 0.03125` |
| 63 | `0.304688/0.009184; 3.75/0.030952; 7.75` | `0.34375/0.013293; 2.25/0.061415; 4.75` | `0.0625/0.000719; 0.171875/0.006542; 0.40625` |
| 64 | `0.257813/0.009161; 4/0.032638; 8.25` | `0.253906/0.013300; 3/0.064670; 6.25` | `0.0625/0.000737; 0.15625/0.006548; 0.375` |
| 65 | `0.306396/0.009441; 5.125/0.030275; 10.5` | `0.3125/0.013704; 5/0.060456; 10.125` | `0.0625/0.000759; 0.140625/0.006619; 0.34375` |
| 127 | `0.5/0.010445; 4.75/0.043734; 9.75` | `0.375/0.014463; 3.5/0.084836; 7.25` | `0.0625/0.000788; 0.203125/0.009045; 0.46875` |
| 128 | `0.288574/0.009792; 5.75/0.040948; 11.75` | `0.375/0.013713; 3.125/0.080069; 6.5` | `0.0625/0.000765; 0.21875/0.008642; 0.5` |
| 129 | `0.5/0.010111; 4/0.043203; 8.25` | `0.5/0.013993; 4/0.083219; 8.25` | `0.0625/0.000764; 0.203125/0.008621; 0.46875` |
| 1023 | `0.5/0.011382; 4.875/0.068795; 10` | `0.5/0.014870; 3.875/0.122805; 8` | `0.0625/0.000822; 0.425781/0.012574; 0.914063` |
| 1024 | `0.5/0.011355; 7.75/0.067263; 15.75` | `0.5/0.014847; 4.625/0.120769; 9.5` | `0.0625/0.000822; 0.362305/0.012565; 0.787109` |
| 1025 | `0.5/0.011427; 6/0.067598; 12.25` | `0.5/0.015015; 4.03125/0.120190; 8.3125` | `0.0625/0.000830; 0.335938/0.012709; 0.734375` |

S128 passed three same-input repetitions with bitwise-equal dQ, dK, and dV.
The nondefault-stream run also passed with the same reported errors. All
outputs were finite, BF16, correctly shaped, and backed by three distinct
storage allocations.

The exact matrix command was:

```bash
for seqlen in 1 63 64 65 127 128 129 1023 1024 1025; do
  python scripts/probe_h100_local_backward.py \
    --seqlen "$seqlen" --reference \
    --comparison-policy upstream-relative
done
python scripts/probe_h100_local_backward.py --seqlen 128 --reference \
  --comparison-policy upstream-relative --repeats 3
python scripts/probe_h100_local_backward.py --seqlen 128 --reference \
  --comparison-policy upstream-relative --nondefault-stream
```

## Synchronization and generated code

After correctness passes, run the exact S128 specialization under:

- [x] compute-sanitizer memcheck
- [x] compute-sanitizer synccheck
- [x] compute-sanitizer racecheck
- [x] IR/PTX/SASS observation
- [x] registers/spills/SMEM recorded

Sanitizer filters and artifact hashes must identify the main backward kernel,
not only preprocess/postprocess helpers. A compile or a helper-only sanitizer
run is not sufficient.

The full process was run at both the exact S128 specialization and the S129
partial-tile boundary:

```bash
compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 99 \
  python scripts/probe_h100_local_backward.py --seqlen 128 --reference \
    --comparison-policy upstream-relative
compute-sanitizer --tool synccheck --report-api-errors no --error-exitcode 99 \
  python scripts/probe_h100_local_backward.py --seqlen 128 --reference \
    --comparison-policy upstream-relative
compute-sanitizer --tool racecheck --report-api-errors no --error-exitcode 99 \
  python scripts/probe_h100_local_backward.py --seqlen 128 --reference \
    --comparison-policy upstream-relative
# The same three commands were repeated with --seqlen 129.
```

Memcheck and synccheck each reported `ERROR SUMMARY: 0 errors`; racecheck
reported `0 hazards displayed (0 errors, 0 warnings)` at both lengths. The
main-backward artifact is PTX 8.8 targeting `sm_90a`; it contains 32
`HGMMA.64x32x16.F32.BF16`, 12 `HGMMA.64x128x16.F32.BF16`, 24 `UTMALDG.4D`,
and five `WARPGROUP.DEPBAR` instructions. `cuobjdump` reports 168 registers,
zero stack, zero local memory, and 1 KiB static shared memory. The realized
M64 x N64 Q1/dO1/PdS1 configuration models 208 KiB of core dynamic storage.

Key artifact hashes:

```text
main backward PTX   55de942ff3191d3db4e20d5adffba3f26076d274220c026b73469dd8c3c95e16
main backward cubin ac8767ce5b580f1a7bdcac31f8c4e0b1fae1c3638ce01706710c8f0e636bb1d7
main backward SASS  f9c4c45324789979d88cf60ba301aec27ea57424c946449a6add5924d337df41
resource report      6ca3f2364642b483991ef12076ce65bd809746ef6333f0a779e0ed46b848a276
```

The production main-backward compile key at S128 is
`821e5a...`; the single-block helper key is `abb1b8...`. The retained raw
files are under ignored paths
`agent_space/collected-exp0004/exp-0004-artifacts/` locally and
`agent_space/exp-0004-artifacts/` remotely rather than the source bundle.

## Measurement

No performance measurement is authorized.

## Decision

**ACCEPT for the declared fixed-length local-d256 H100 text envelope.** Every
predeclared correctness, stream, repeat, sanitizer, and generated-code gate
passed without a kernel, tile, stage, mask, accumulation, or public-adapter
change. This acceptance does not rewrite EXP-0003: its frozen elementwise
comparison remains rejected and reproducible with the probe's default
`frozen` policy.

The diagnostic explanation is consistent across two controls. The pinned
FA4 implementation intentionally converts P to BF16 for dV and dS to BF16
before dQ/dK; a manual reference with those stage boundaries reduced mean
candidate-to-reference errors to `0.0000103`, `0.0000188`, and `0.00000647`
for dQ/dK/dV. The upstream-supported d128 control also failed EXP-0003's fixed
elementwise envelope (dQ/dK maxima `0.25`/`0.25`). These controls diagnose a
reference-rounding-policy mismatch, not permission to relax the old gate.

Upstream itself still excludes SM90 d256 backward from its broad test, so the
project claim remains limited to B1, exact 32Q/16KV GQA-2, d256, scale 1.0,
distinct K/V, causal W1024, BF16, and the tested S<=1025 matrix. No varlen,
vision-mask, long-context, performance, or B300 claim is made. The next
ordered experiment is global-d512 backward.
