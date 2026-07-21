# H100 M1 status

**Status date:** 2026-07-20

**Ordered gate result:** advanced through eager StaticCache active-prefix
execution, the scoped EXP-0023 guarded no-cache compile facade, the
EXP-0026 scoped global compiled-StaticCache decode envelope, and the EXP-0028
scoped local compiled-`StaticSlidingWindowLayer` decode envelope on the
pinned-Transformers H100 boundary. The project owns a
uniquely named attention/mask backend, preserves the exact Gemma 4 mask and prepared-operand
contracts, and routes fixed, padded, packed-varlen, lower-right, and long
no-grad calls inside their declared FA4 envelopes.
The H100 environment, fixed and packed local d256 paths, exact local
multimodal masking, and exact global d512 text forward/backward passed
their declared gates. EXP-0003's
fixed elementwise dQ/dK envelope remains rejected; EXP-0004 preserved that
result and accepted the unchanged local backward under a separately
predeclared upstream-relative BF16 oracle. EXP-0005 remains the historical
rejection of the unchanged direct asymmetric GQA-8 global backward. EXP-0006
accepts a structural dQ/dKV split under the exact B1, S<=1024, BF16,
32Q/4KV, GQA-8, d512, causal, scale-1.0, distinct-K/V contract. EXP-0007
accepts the exact fixed B1 local vision predicate; EXP-0008 accepts nonempty
packed local self-attention with `B>=1` and `1 <= Sq <= Sk <= 1025`, including
lower-right native text and custom vision/document masking. EXP-0009 extends
native packed text to `1 <= Sq <= Sk <= 262144`. EXP-0010 extends exact
vision/document metadata through the same maximum when the schedule fits the
`2^40` padded-score, 2 GiB metadata, and 10%-free-HBM ceilings. EXP-0011's
eager framework probe, maximum-context global-forward sentinel, focused
sanitizer cases, bounded cache inventory, and checksum-locked bundle verifier
pass. EXP-0011 is accepted for its scoped eager envelope. At that historical
boundary compiled/static-cache integration was unrun; EXP-0026 now accepts
only the later scoped global layer-5 decode facade. EXP-0012 extends the unchanged split global
backward through fixed S2048 and exactly composed lower-right/packed K2048,
with HBM preflight, independent O/LSE/gradient references, sanitizer-clean
fixed and packed cases, and byte-identical main objects. EXP-0013 replaces the
normal packed training composition with one native THD/cu-seqlens forward and
split backward for nonempty per-segment `1 <= Sq <= Sk <= 2048`. Fixed/native
parity, mixed long and square references, gradient-source/document/segment
isolation, odd noncontiguous upstream gradients, guarded memory, sanitizers,
bounded cache classes, and PTX/SASS resource evidence pass. Only the dedicated
native HBM-budget exception may select the retained exact composer; other
failures propagate. EXP-0014 extends only native packed THD/cu-seqlens global
backward to nonempty `1 <= Sq <= Sk <= 262144` segments under signed-INT32 and
guarded-HBM admission. Fixed BSHD and the exact composer remain capped at
S/K2048; for K>2048, budget rejection propagates before forward and cannot
fall back to the composer or FlexAttention. Tractable dense references,
Q1/K262144 and square-S32768 analytic oracles, eager Transformers execution,
sanitizers, bounded cache classes, and unchanged main-object bytes/resources
pass. EXP-0015 admits mixed packed plateaus on the accepted local and global
paths: every segment may satisfy `0 <= Sq <= Sk <= 262144`, including paired
empty and query-empty/key-nonempty segments, while aggregate Q/K totals and
exact maxima must remain positive. CPU guards, native local text, dense and
long sparse local metadata, native/composed global routing, an actual pinned
global layer with a fully padded row, exact-zero empty-slice gradients,
sanitizers, FakeTensor kernel compilation, bounded cache reuse, and unchanged
main-object bytes/resources pass. An all-empty physical workload remains an
explicit pre-backend rejection. EXP-0016 accepts B1 text-only eager
StaticCache active-prefix prefill/decode, including local rollover, with
strict prepared O/LSE references, pinned-layer operand capture, hostile-tail
isolation, stable cache storage, bounded cache keys, and clean project-owned
FA4 sanitizer runs. EXP-0017 through EXP-0022 reject successive raw
fullgraph/compiler candidates without weakening their frozen gates: they
establish cache/mask provenance, localize outer-layer drift, preserve opaque
whole-layer arithmetic and live inference weights, and finally localize the
remaining two-attempt PyTorch 2.8 restart to Dynamo scalar-source bookkeeping.

EXP-0023 accepts a different, explicit project-owned boundary: a guarded
tensor-only compile facade originally proven on pinned layers 0/local and 5/global, B1
BF16 no-cache text inference/no-grad, exact zero-based positions, and
`1 <= S <= 1024`. Every call validates all live module/config/weight state and
eleven scalar fields before compiled entry. Local/global × eager/Inductor at
S={1,32,33,1023,1024} is bitwise to eager with zero graph breaks, one
whole-layer op, exactly the public S1 and S>1 graph classes, bounded FA4 keys,
mutation/input/API fail-closed sweeps, replay/reorder/nondefault-stream reuse,
clean project-kernel sanitizers, and unchanged retained FA4 codegen. A separate
nondefault size-oblivious diagnostic produces one graph per family/backend.
This is not raw `torch.compile(layer)`, a compiled cache, full-model
compilation, or varlen facade support. EXP-0042 later widens only this guarded
no-cache facade to all 60 locked layer indices. All compiler benchmarks remain
unrun.

EXP-0024 rejects the first global compiled-StaticCache candidate because the
upstream-marked cache roots became FX `get_attr` buffers rather than explicit
runtime inputs, despite otherwise exact Q1/K33 behavior. EXP-0025 accepts the
refined full-storage-view transport at that narrow discriminator: the pinned
root cache stays guarded outside Dynamo while K, V, and the counter remain
explicit graph placeholders. EXP-0026 widens only that global layer-5 facade
to sequential K33/K34 and independent K1025/capacity1026 decode with stock
eager and Inductor backends. Different seeds and reversed case order retain
one semantic graph class, two capacity-shape signatures, zero breaks, exact
eager output/cache bytes, hostile-tail isolation, stable storage, clean K1025
project-kernel sanitizers, and unchanged retained global FA4 codegen.

EXP-0027 rejects the first local compiled-cache candidate because declaring
the saturated CUDA counter mutable changed its tensor version during rollover
even though its bytes remained 1024. EXP-0028 changes only counter ownership:
the local op mutates K/V and the eager guarded facade conditionally advances
the counter after successful compiled execution. The refined pinned local
layer-0 facade passes K33/K34 underfill, K1024 boundary fill, and repeated
rollover through absolute position 1025 under eager and Inductor. It retains
one semantic/runtime graph signature, exact eager output/cache/counter state,
stable storage, hostile-tail isolation, 16 fail-closed negative cases, clean
project-kernel sanitizers, and unchanged native local-varlen codegen. Compiled
prefill, cached vision/document metadata, other cache-layer indices, raw/full-model
compilation, training, compiled-facade performance, and B300 remain unsupported.

EXP-0029 establishes the unlocked-clock H100 performance ruler for exact,
semantically equivalent project FA4 calls. Global d512 backward is the first
tuning target: S8K backward measured 98.239 ms median and the dQ main kernels
dominated the profiled backward work. EXP-0030 rejected removing one dQ
warpgroup because its sub-percent apparent changes overlapped baseline noise.
EXP-0031 rejected splitting N16 into independently owned N8 fragments at the
compile gate. EXP-0032's reviewed full-V512 N16 one-warpgroup candidate passed
fake compile, S128 reference/repeat/nondefault-stream checks, and its memory
preflight, but was 3.93% slower at the first S8K backward gate; the accepted
patch was restored exactly and strict environment verification passed.

EXP-0034 admits two semantically equivalent PyTorch global baselines and shows
that the accepted FA4 path is already faster on the measured H100 workloads.
Against explicitly expanded fused SDPA, unlocked-clock S8K speedups are 3.20x
forward, 1.50x backward, and 1.60x combined; the reduced S64K screen is 5.65x,
1.40x, and 1.63x. Automatic-GQA SDPA decomposed into GEMM, softmax, and
elementwise kernels and was slower still. These are global causal BF16 claims
only, not local multimodal, B300, or universal-attention claims.

EXP-0035 accepts the reviewed exact-BF16 follow-up: two MMA warpgroups keep
full K512/V512 resident while Q and dO stream through one D256 slot over two
barrier generations. Both score/dP halves accumulate before dS is formed once,
reducing four dQ launches to two while preserving byte-identical dKV code. A
strengthened packed O+LSE racecheck found and drove an explicit statistic-load
rendezvous before slot release; final fixed and packed memcheck, synccheck, and
racecheck are clean. Post-fix unlocked-clock S8K backward is 57.683 ms versus
the 96.928 ms ruler (-40.5%), and S64K is 3435.331 ms versus 6039.759 ms
(-43.1%). Combined improves 38.4% and 40.6%. The exact path is enabled by
default, with `FLASH_ATTENTION_GEMMA4_EXPERIMENT_DQ_D256_STREAM=0` retaining
the prior slab-dQ rollback. These are H100 global-causal BF16 results only.

EXP-0037 accepts the next exact-BF16 step. One M64 x N32 dKV kernel keeps
Q512/K512/V512 resident, streams dO low/high/low-replay through one D256 slot,
forms dS and dK once, and reuses dead Q shared storage for sequential FP32
dK/dV reduction. This replaces two dKV slab launches with one and makes three
main launches the default. Fixed and packed references, bounded peak memory,
memcheck, synccheck, and racecheck pass. The compiled fused dKV object uses
168 registers, zero stack/local memory, 174,080 dynamic shared bytes, 96 HGMMA
and 63/66 UTMA instructions. Unlocked S8K backward is 51.442 ms (-10.8%
versus EXP-0035), S64K is 3053.413 ms (-11.1%), and combined improves 9.9%
and 10.3%. Set `FLASH_ATTENTION_GEMMA4_EXPERIMENT_DKV_D256_STREAM=0` for the
retained slab-dKV rollback. This is not a single-launch or deterministic path.

EXP-0038 accepts one more exact-BF16 launch fusion. The two dQ D256 output
halves now execute sequentially in one main kernel while reusing the same
accumulator registers, 64 KiB shared epilogue arena, and 160-participant
empty/full barriers. Fixed and packed references, mixed-empty ownership,
bounded memory/cache, nondefault stream, rollback, and all three sanitizers
pass. Nsight Systems shows exactly two backward main launches, one dKV and one
dQ. The dQ object uses 168 registers, zero local memory, and 201,728 dynamic
shared bytes. At unlocked clocks, S8K backward improves from 51.318 to 43.040
ms (-16.1%) and S64K from 3054.023 to 2579.334 ms (-15.5%), with disjoint
IQRs; combined improves 14.8% and 14.1%. The route is enabled by default.
Set `FLASH_ATTENTION_GEMMA4_EXPERIMENT_DQ_D512_SINGLE_LAUNCH=0` to restore
EXP-0037's two dQ launches. The complete backward remains two main launches
and gradients remain nondeterministic.

EXP-0039 adds and hardware-validates an explicit `deterministic=True` global
backward route while leaving EXP-0038 as the fast default. Two ordered V256
dKV launches and one ordered full-D dQ launch produce bitwise-identical O,
LSE, dQ, dK, and dV across five repeats for fixed and native packed cases.
Fixed/packed reference, ownership, isolation, nondefault-stream, memory,
cache, launch-count, memcheck, synccheck, and racecheck gates pass. Fresh
caches contain one dKV and one dQ main object per ABI; Nsight records three
main launches. At unlocked clocks, S8K backward costs 51.635 ms versus 42.722
ms fast (+20.9%), inside the declared 25% ceiling. The S64K smoke is 5364.229
ms versus 2578.563 ms (+108.0%), so deterministic mode is a correctness
option, not the long-context throughput route. Whole-sequence FP32
accumulation remains and is the next global-backward production limitation.

EXP-0040 makes one CTA own each `(batch, kv_head, N32)` tile and directly
stores final BF16 dK/dV after visiting all eight Q heads. This removes the
whole-sequence FP32 dK/dV workspaces and their postprocess launches. Fixed and
packed references, sanitizers, generated code, memory, and S8K/S64K gates
pass. Fast backward remains two main launches because dQ is independently
owned and still uses a whole-sequence FP32 accumulation buffer.

EXP-0041 accepts the H100 global D512 forward replacement. The direct M128
full-D candidate was rejected because SM90 WGMMA caps PV N at 256. The accepted
M64 x N32 kernel uses two consumer warpgroups with disjoint O256 ownership;
WG0 computes QK and online softmax once and shares BF16 P plus FP32 rescale
factors with WG1. Fixed boundaries, packed mixed/empty/isolation, exact
rollback parity, nondefault stream, fixed/native parity, and all three
sanitizers pass. Nsight records one forward main launch, 384 threads, 168
registers, zero stack/local memory, 1 KiB static plus 205,824 bytes dynamic
shared memory. Hot-L2 S8K improves 6.1936 to 5.5986 ms (-9.61%); the 30-sample
S64K gate improves 364.7140 to 356.3823 ms (-2.28%), with disjoint IQRs. The
route is default-on; set
`FLASH_ATTENTION_GEMMA4_EXPERIMENT_FORWARD_D512_SINGLE_LAUNCH=0` for the exact
two-V256 rollback. These are H100 global-causal BF16 claims only.

EXP-0042 widens the accepted guarded no-cache compiled facade from the
representative layer indices 0 and 5 to all 60 locked text layers. Every actual
pinned layer at S1 is bitwise to eager under both eager and Inductor backends,
with zero graph breaks, exactly two family graphs, and exactly two FA4 forward
application classes. The local/global S33/S1024 regression matrix remains
bitwise and retains the mutation, negative-input, replay, and nondefault-stream
guards. Each facade captures its construction-time index; same-family index
mutation fails before compiled entry. Layers 58/59 retain their exact pinned
terminal-family storage marker, but `num_kv_shared_layers=0` means no layer
consumes another layer's prepared K/V. Cache facades remain layer-0/layer-5
scoped.

EXP-0033 separately documents
an opt-in FP8 V/dO feasibility idea. It is not implemented or approved: pinned
FA4 does not support FP8 backward, and the proposal makes dQ approximate even
though the accepted dK/dV paths remain BF16.

M0 remains the semantic contract: scale is exactly `1.0`; K/V are distinct
prepared operands; backward returns separate dQ, dK, and dV; and the local
multimodal predicate is
`k > q - 1024 AND (k <= q OR same nonnegative vision block)`.

## Environment and provenance: PASS

Observed H100 identity:

```text
NVIDIA H100 80GB HBM3
compute capability 9.0, 81559 MiB
driver 580.126.09
CUDA toolkit 12.8 (nvcc V12.8.93)
PyTorch 2.8.0+cu128, runtime 12.8
CuTe DSL 4.6.0.dev0
quack-kernels 0.5.3
```

The strict H100 policy uses FA4 `[dev]`, never the CUDA-13 `cu13` extra.
ShellCheck 0.9.0 and CUDA-12.8-matched Nsight Systems 2024.6.2 are installed.
The profiler-required environment check passes with every required tool found.

FlashAttention is base revision
`77aacb68d194ba9af1010eda5eac3e7c0df8e6f6` plus exactly one H100 patch:

```text
patches/flash-attention/0004-sm90-gemma4-forward-d512-single-launch.patch
SHA256 9d14635e23199200f0b25cbd9d33f464d1dd119a527838cc96098e9b8b3d91dd
```

The patch carries EXP-0038's accepted full-D dQ single-launch route and
selects it by default. Together with EXP-0037's full-D dKV kernel, global
backward uses two main launches. The explicit EXP-0038 flag value `0`
restores the accepted EXP-0037 route.

The same patch carries EXP-0039's accepted explicit deterministic-backward
route. It is default-off, leaves EXP-0038 dispatch unchanged, and is scoped to
the exact H100 global-causal BF16 contract. Its fixed and packed gradients are
bitwise repeatable; it makes no speedup or bounded-memory owner-computes claim.

The cumulative patch also carries EXP-0040's accepted fast-default dKV route.
One CTA owns each `(batch, kv_head, N32)` tile, accumulates the eight GQA
Q-head contributions in FP32 registers, and directly stores final BF16 dK/dV.
This removes `16384 * padded_K` bytes of internal FP32 workspace and all dK/dV
postprocess launches. The dQ FP32 workspace remains, so the complete backward
is not yet bounded-memory. Setting
`FLASH_ATTENTION_GEMMA4_EXPERIMENT_OWNER_DKV=0` is the tested rollback;
`deterministic=True` continues to select EXP-0039.

The cumulative patch also carries EXP-0041's accepted cooperative global
forward. It is the fast default for fixed and native packed calls and emits
O512 plus one FP32 LSE in one launch. The environment flag value `0` restores
the exact EXP-0002 two-V256 composition without changing backward dispatch.

Transformers is base revision
`7ea2320c76117e6742364808a666ef6f2fb40a67` plus exactly one two-file H100
integration patch:

```text
patches/transformers/0001-gemma4-forward-vision-block-ids.patch
SHA256 ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79
```

The patch computes or accepts one authoritative vision-block tensor before
mask construction, uses that same tensor for the mask, and forwards it to the
registered attention interface even when a prebuilt generation-mask mapping
was supplied. The retained EXP-0018 patch additionally transports the exact cache
object, a private plain causal/sliding origin, and the selected mask recipient
to the project callback before layer cache mutation. That boundary passed its
real-cache immutability matrix, but EXP-0018's positive fullgraph route was
rejected on its frozen numerical gate and is not an accepted compatibility
claim. `scripts/check_env.py`
requires both pinned base revisions, both exact patch diffs and hashes, no
additional tracked or untracked upstream checkout changes, and imported
FA4/Transformers modules resolving inside the pinned checkouts.

The retained strict reports `agent_space/h100-check-precommit.json` and
`agent_space/h100-check-exp0006.json` both have SHA256
`b41270f33f67dda21af8c8b45d7f76c5287daa2c5a10598f3887f9a56cddc9d4`.
The EXP-0012 strict report is `agent_space/h100-check-exp0012.json`, SHA256
`3667043819fed0b5bcf6abaffa05a3aa2b2ae7157d22f37abd09918d82c65205`;
it records the updated exact FA4 patch and no warnings or errors.
The EXP-0013 strict report is `agent_space/h100-check-exp0013.json`, SHA256
`0ef779d2f175e821bd6520852d555b8759ce78e22ec402f270324398fbae9476`;
it records patch SHA256
`c1f5be0ef864fcd716309ae1add48a4c71b8da28578a983083bbba91054a8ee0`
and no warnings or errors.
The EXP-0014 strict report is `agent_space/h100-check-exp0014.json`, SHA256
`cfdcfe48c1f65325fbaa144e2a44294682095a45390ec6246143cd37a740e49b`;
it records patch SHA256
`97dd1dd7c9c8efb5f2b2fd06f601a1bcc8abbe9ebcf43e9ea86365769c2e2749`,
`applied_exactly: true`, and empty warnings/errors.
The EXP-0015 strict report is `agent_space/h100-check-exp0015.json`, SHA256
`18c46284dd978362523f0d1d8b73adfc5fd45bdfd0ace0f7ec0c3aa86c5efdae`;
it records current patch SHA256
`eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb`,
`applied_exactly: true`, and empty warnings/errors.
The EXP-0016 strict report is `agent_space/h100-check-exp0016.json`, SHA256
`18c46284dd978362523f0d1d8b73adfc5fd45bdfd0ace0f7ec0c3aa86c5efdae`;
it records the same exact environment/patch state and empty warnings/errors.
The EXP-0018 strict report is
`agent_space/remote-h100-exp0018/h100-check-exp0018.json`, SHA256
`3cef71f936c264dfebc8d521ca61b666dec8793b8152ca82a3f7c05bd52ecc4a`;
it records Transformers patch SHA256
`c812937e5a554c1887c2c16a0808f24437cb8b60b561e9fd5eacaa13fb277780`,
both exact patch stacks, and empty warnings/errors.
The EXP-0019 strict report is
`agent_space/remote-h100-exp0019/h100-check-exp0019.json`, SHA256
`4ee189fd65b8377723f8903b7bac3fd56537375a50029e6a0ce7c6594323fc72`;
it records revised Transformers patch SHA256
`ebeff866ce79b5f275df8f0565c3377283df1238bb9cd19c629c98c44b0d3b79`,
both exact patch stacks, and empty warnings/errors.
The EXP-0020 strict report is
`agent_space/remote-h100-exp0020/h100-check-exp0020.json`, with the same
SHA256 `4ee189fd65b8377723f8903b7bac3fd56537375a50029e6a0ce7c6594323fc72`;
no environment, dependency, or patch input changed and warnings/errors remain
empty.
The EXP-0021 and EXP-0022 strict reports are respectively
`agent_space/remote-h100-exp0021/h100-check-exp0021.json` and
`agent_space/remote-h100-exp0022/h100-check-exp0022.json`, both with the same
SHA256 `4ee189fd65b8377723f8903b7bac3fd56537375a50029e6a0ce7c6594323fc72`.
They confirm the identical pinned H100, dependency, and patch state with empty
warnings/errors.
EXP-0001 through EXP-0003 are machine-recorded against source revision
`5b9bfab072e8cc28a7e92c9e956608db591b246c`.
EXP-0004 is machine-recorded against its validated source revision
`49fbcad2e2b761d9de50312f03335e27236a8a13`.
EXP-0005 is machine-recorded against its rejected source revision
`d7ac7273aaed5c57923301afa6f052333e91c5b7`.
EXP-0006's accepted implementation source is
`185f11cbda15ae7bd4841968c3dd46f95b670282`.
EXP-0007's accepted implementation source is
`d1b7e4ad0b1ffff6e3190a4b4411603cd544afe4`.
EXP-0008's accepted implementation source is
`de6450a9cf5040a7432ed7641b230bb29f835248`.
EXP-0009's accepted implementation source is
`9c6b385dbae9f979aa2a38ecd0a2ed505a76cfcf`.
EXP-0010's accepted implementation source is
`12cfe711ad29139c7c78dcb355645ee5b9a70bb0`.
EXP-0011's accepted implementation source is
`e7f26bba9b6795e3022c733cff39e060075daf57`.
EXP-0012's accepted implementation source is
`ebe993c5b23aae66ecbcf90ee737988546482b9a`.
EXP-0013's accepted implementation source is
`87ff75b1b40b55149ec5beea7480ed9ac14c9146`.
EXP-0014's accepted implementation source is
`364ea6ab27513a42d1b3e9f7baf9213720c1a530`.
EXP-0015's accepted implementation and FakeTensor-test source is
`cca09c8211b3c643b9b311f5fec0798f84a9ea0f`.
EXP-0016's accepted eager StaticCache implementation source is
`c5ee7bec833c9617ccf323955bcafc80b72cd932`.
EXP-0017's rejected no-cache compiler candidate source is
`96cdfa16d70b304718856441e06c8cbcd8281f31`; the rejection preserves the
successful scoped FakeTensor/custom-op evidence without promoting it to
framework compatibility.
EXP-0018's rejected mask-boundary compiler candidate source is
`e9a5af6f88f8d2be74256da1c89a8926d6f89fdd`; its exact pre-entry cache
rejection remains retained, but the failed frozen S1023 Inductor numerical
gate prevents promotion to framework compatibility.
EXP-0019's rejected whole-layer ownership candidate source is
`0adfc0a2fe9df85e01b91d1bc846acf5d2f6ae12`.
EXP-0020's rejected inference-weight candidate source is
`f70c828ef3ea909949d3e43606a9c950776b6a8b`; its snapshot-free eager/ABI
evidence remains retained, but the two-attempt S1 Inductor result prevents
promotion to framework compatibility.
EXP-0021's rejected tensor-scalar candidate source is
`d35a97d6bbe52421f7a04e7d2ab196a2c251a733`; its explicit scalar transport
was removed in `3c2c68163e5491f60af0af6ac7e86178c75254fe`.
EXP-0022's rejected comptime candidate source is
`8e3c79e88fb1c76c29b5401bf7b123c5b0c83670`; its static-guard calls and
probe-only instrumentation were removed in
`dd3d45179204e85c6adefc5d6793f6c13525a3d7`.
EXP-0023's accepted guarded-facade product source is
`1756ec0df13d25ac8bd48d9c018ec438402db428`; the final matrix/probe source is
`f592971c09da16fc68db15ad588f94f6e1bde0be`.

The FA4 patch opens the exact `(Dqk,Dv)=(512,256)` SM90 forward specialization,
the reviewed split-backward ownership variants, and their packed THD/cu-seqlens
ABI, but is not itself a mask-mode guard. The project adapter is the semantic
guard: native packed training permits mixed segments satisfying
`0 <= Sq <= Sk <= 262144` only when aggregate Q/K totals and exact maxima are
positive, while fixed BSHD and the exact composer remain capped at S/K2048.
No-grad fixed and packed global forward calls also extend through the locked
K262144 maximum subject to output/LSE HBM preflight. Training preflights
retained forward state plus anticipated packed backward scratch, and the
patched low-level backward rechecks free HBM immediately before allocation. A
budget rejection with an active-query K>2048 propagates before forward instead
of selecting the composer or FlexAttention. Decreasing arrays, `Sq>Sk`, and
all-empty physical workloads fail before backend launch.

## Forward gates: PASS

### Local d256 fixed-length text forward

`fa4_local_text_forward` fixes BSHD BF16, 32Q/16KV for the model, d256,
scale 1.0, inclusive FA window `(1023, 0)`, causal text semantics, distinct
K/V, B=1/S<=1025, one split, no pack-GQA, BF16 O, and FP32 LSE. EXP-0001's
original matrix used contiguous inputs. EXP-0011 additionally validates the
pinned CuTe layout contract used by the framework boundary: unit D stride,
positive nonoverlapping outer strides divisible by eight BF16 elements, and a
16-byte-aligned base pointer. This admits the canonical no-copy BHSD-to-BSHD
transpose produced by pinned Transformers.

H100 evidence:

- fake compile: `1 passed, 36 deselected`;
- real focused run: `18 passed, 19 deselected`;
- sequence lengths `1,63,64,65,127,128,129,1023,1024,1025`;
- GQA ratios 1, 2, 4, and 8 at S=33;
- unchanged O tolerance `atol=0.03125, rtol=0.02`;
- unchanged LSE tolerance `atol=0.125, rtol=0`;
- exact repeat on a nondefault stream.

This is not a varlen or vision-mask result. See
`experiments/EXP-0001-h100-local-d256-forward.md`.

### Global d512 fixed-length text forward

EXP-0002 established an exact correctness composition. It splits V into two
contiguous d256 slabs, runs the same patched SM90
`(Dqk,Dv)=(512,256)` M128 x N32 specialization twice over identical Q/K,
requires identical FP32 LSE, and concatenates the two d256 outputs:

```text
concat(P @ V0, P @ V1) = P @ concat(V0, V1)
```

EXP-0041 retains that path as rollback and makes one cooperative M64 x N32
D512 launch the accepted default.

The training adapter rejects anything outside B=1/S<=2048 at this direct
entry point. EXP-0002's original evidence used contiguous inputs; EXP-0011
adds the same legal aligned dynamic-stride contract described for local
attention, including the canonical no-copy Transformers view.

H100 evidence:

- fake compile: `1 passed, 36 deselected`;
- real focused run: `17 passed, 20 deselected`;
- exact 32Q/4KV, GQA 8, causal, scale 1.0, distinct K/V;
- S=`1,31,32,33,63,64,65,127,128,129,511,512,513,1024`;
- frozen O envelope `atol=0.0625, rtol=0.03`, observed max abs `0.015625`;
- frozen LSE envelope `atol=0.25, rtol=0`, observed max abs
  `0.00015258789`;
- exact LSE agreement between slab launches and exact nondefault-stream repeat;
- filtered memcheck and synccheck: 0 errors each;
- filtered racecheck: 0 hazards, 0 errors, 0 warnings.

EXP-0012 retains that exact forward composition and adds backward evidence at
S1025 and S2048. Both pass O/FP32-LSE and independent dQ/dK/dV references;
S1025 also passes dO-only, true LSE-only, combined dO+dLSE, V-slab
superposition, and isolated GQA-head ownership. Three S2048 nondefault-stream
runs pass the frozen numerical policy. O/LSE/dV repeat bitwise; dQ/dK retain
the documented non-bitwise FP32 atomic reduction order.

Both original forward acceptances are scoped to the seeded-random matrices
above. Coordinate-coded, zero/repeated/large-logit, and adversarial-BF16 cases
remain explicit follow-up hardening before tuning or broader support.

The raw sanitizer instrumentation reports 34 CuTe/cuda-python
`cuGetProcAddress_v2` API-probe errors. Runs with `--report-api-errors no`
isolate and pass the memory/synchronization tools; both facts are recorded in
`experiments/EXP-0002-h100-global-d512-forward.md`.

Generated-code evidence for the asymmetric specialization:

- 168 registers, zero stack, zero local memory, 1 KiB static shared memory;
- 96 `HGMMA.64x32x16.F32.BF16` and 6
  `HGMMA.64x256x16.F32.BF16` instructions;
- 32 `UTMALDG`, 4 `UTMASTG`, and 6 `WARPGROUP.DEPBAR` instructions;
- 224 KiB modeled dynamic core storage;
- exact dynamic shared-memory launch metric remains unresolved because Nsight
  Compute returns `ERR_NVGPUCTRPERM` on this pod.

The two launches duplicate QK/softmax work and materialize V slabs. No speed or
efficiency claim is made.

## Local backward gates: EXP-0003 REJECT preserved; EXP-0004 PASS

The unchanged pinned M64 x N64 Q1/dO1/PdS1 local backward uses exact
32Q/16KV GQA-2, BF16, scale 1.0, causal W1024, distinct K/V, and separate
dQ/dK/dV. Upstream still excludes SM90 backward above d192 from its broad
test, so this is a project-scoped hardware validation rather than an
upstream/general support claim.

EXP-0003's exact fake compile produced separate shapes:

```text
dQ=(1,128,32,256)
dK=(1,128,16,256)
dV=(1,128,16,256)
```

Its first real S=128 comparison failed the frozen
`atol=0.125, rtol=0.05` envelope:

- first dQ failure: 552 / 1,048,576 mismatches, max abs `0.2890625`, max
  relative `1112.0`;
- diagnostic rerun without changing tolerance: dQ max/mean abs `0.28857422` /
  `0.0097916815`, dK `0.375` / `0.013713409`, and dV `0.0625` /
  `0.00076462259`;
- dQ and dK failed; dV passed. This rejection and the probe's default
  `frozen` policy remain intact.

EXP-0004 predeclared the numerical rule used by the pinned upstream CuTe
tests, with an independent PyTorch BF16 attention path as the baseline:

```text
max_abs(g - g_ref) <= 2 * max_abs(g_pt - g_ref) + quantization_atol
quantization_atol = 2 * max_abs((g_ref + 0.3 - 0.3) - g_ref)
```

No kernel, tile, stage, mask, accumulation type, or public adapter changed.
The exact B1/S/32Q/16KV/d256 matrix passed at
S=`1,63,64,65,127,128,129,1023,1024,1025`. Every case returned finite BF16
dQ/dK/dV with exact input shapes and three distinct output allocations. Across
the matrix, candidate maximum absolute error ranges were:

- dQ: `0.00002277` to `0.5`;
- dK: `0.00002480` to `0.5`;
- dV: `0` to `0.0625`.

Every value was below its independently computed upstream-relative limit; the
complete candidate/baseline max/mean table is retained in EXP-0004. Three
same-input S128 repetitions were bitwise equal for dQ, dK, and dV, and the
nondefault-stream run passed the same numerical gate.

Memcheck and synccheck reported zero errors, and racecheck reported zero
hazards/errors/warnings, at both S128 and the S129 partial-tile boundary. Main
backward generated-code evidence:

- PTX 8.8 targeting `sm_90a`;
- 32 `HGMMA.64x32x16.F32.BF16` plus 12
  `HGMMA.64x128x16.F32.BF16` instructions;
- 24 `UTMALDG.4D` and 5 `WARPGROUP.DEPBAR` instructions;
- 168 registers, zero stack, zero local memory, 1 KiB static shared memory;
- 208 KiB modeled core dynamic storage.

Two controls explain why EXP-0003 and EXP-0004 can legitimately have
different decisions. The pinned backward intentionally rounds P to BF16 for
dV and dS to BF16 before dQ/dK; a reference mirroring those stage boundaries
matched the candidate at mean errors `0.0000103`, `0.0000188`, and
`0.00000647`. The upstream-supported d128 control also failed EXP-0003's fixed
dQ/dK envelope, with maxima `0.25` and `0.25`. This diagnoses a
reference-rounding-policy mismatch; it does not retroactively loosen
EXP-0003.

Acceptance is limited to fixed-length B1, exact model GQA-2, d256, scale 1.0,
causal W1024 text attention, and the tested S<=1025 matrix. No backward GQA
1/4/8, varlen, vision-mask, global, long-context, performance, or B300 claim
is made. See EXP-0003 and EXP-0004 for the immutable reject/accept records.

## Global backward gates: EXP-0005 REJECT; EXP-0006/0012/0013/0014 PASS

EXP-0005 remains an immutable rejection of the unchanged path. The exact
two-slab forward presents each direct FA4 backward as d512 Q/K, d256 V/output,
and GQA-8. The pinned constructor rejects it before main compilation:

```text
AssertionError: GQA backward requires head_dim == head_dim_v
```

Head expansion bypassed that assertion in a diagnostic, but the unchanged
monolithic launch requested 345,088 bytes against SM90a's 232,448-byte limit.
EXP-0006 does not revise either result; it changes work ownership.

The original EXP-0006 composition ran one M64 x N32 dKV-only main launch and
two M64 x N32 dQ-only launches for each V256 slab. EXP-0035 retained the two
dKV launches but replaced four dQ launches with two exact D256-output kernels.
EXP-0037 retained those dQ objects byte-for-byte and replaced the two dKV slab
launches with one full-D512 dKV kernel. It accumulates both dP halves before
the nonlinear dS step, forms dK once, and uses low/high dV accumulators before
separate BF16 conversion. EXP-0038 replaces the two dQ launches with one
sequential D256-output dQ kernel. The accepted two-main-launch default
therefore computes:

```text
dQ = concat(dQ_low, dQ_high)
dK = dS.T @ Q
dV = concat(P.T @ dO_low, P.T @ dO_high)
```

The exact B1/BF16/32Q/4KV/GQA-8/d512/causal/scale-1.0/distinct-K/V
matrix passed at
S=`1,31,32,33,63,64,65,127,128,129,511,512,513,1024`. Every case returned
finite BF16 dQ/dK/dV with exact shapes and distinct storage and passed the
unchanged EXP-0004 upstream-relative numerical rule against independent FP32
and BF16 PyTorch references. Structured half-zero dO-slab superposition and
isolated-query-head GQA ownership checks also passed.

Three same-input S128 repeats produced bitwise-identical O and FP32 LSE.
Gradients were not bitwise identical because FP32 bulk/atomic reduction order
can vary, but every repeat independently passed the frozen numerical policy.
The nondefault-stream run passed the same contract; this is not a
deterministic-gradient claim.

Memcheck and synccheck reported zero errors, and racecheck reported zero
hazards/errors/warnings, at both S128 and the S129 partial-tile boundary.
Generated-code resource evidence for the fixed EXP-0006 main variants is:

- dKV-only: 222,208 bytes dynamic shared memory;
- each dQ-only D256 variant: 218,112 bytes dynamic shared memory;
- all variants: 168 registers, 1 KiB static shared memory, zero stack, and
  zero local memory.

The current EXP-0037 fused dKV main object uses 174,080 dynamic shared bytes,
168 registers, 1 KiB static shared memory, zero stack/local bytes, 96 HGMMA,
and 63 fixed / 66 packed UTMA instructions. The EXP-0038 dQ object uses
201,728 dynamic shared bytes, 168 registers, zero local memory, 68 HGMMA, and
56 UTMA instructions. Nsight Systems confirms one fused dKV plus one dQ main
launch per call. The packed dQ scheduler retains its accepted 24-byte stack
and 17 LDL / 8 STL instructions without spill growth; fixed dQ has zero stack.

EXP-0012 extends the same fixed scheduler and exact per-segment framework
composition through S/K2048 with guarded HBM admission. EXP-0013 adds native
packed THD/cu-seqlens execution for nonempty `B>=1` segments satisfying
`1 <= Sq <= Sk <= 2048`; it changes the ABI and packed workspace, not the
tile, stages, warp-group ownership, or model semantics. Only native HBM-budget
rejection may select the retained composer.

EXP-0013 passes equal S65 fixed/native parity, lower-right Q33/K1025, mixed
Q=[33,65]/K=[1025,2048], square S2048, dO-only/LSE-only/combined gradients,
hostile segment and document isolation, odd noncontiguous dO/dLSE, repeats,
nondefault stream, resource bounds, and focused sanitizers. Fixed main objects
remain byte-identical. Native main objects use 168 registers and 1 KiB static
shared memory; dKV has zero stack/local, while dQ-low/high have a 16-byte stack,
zero local memory, seven `LDL`, and four `STL` instructions. Generated shared
storage remains configured at 222,208 bytes for dKV and 218,112 bytes for dQ.

EXP-0014 changes only native admission and the matching managed upstream
varlen assertion. Native nonempty segments now satisfy
`1 <= Sq <= Sk <= 262144` under exact maxima, signed-INT32 cumulative/padded
totals, and guarded-HBM preflight. Q33/K2049, packed
Q=[33,65]/K=[2049,4097], and square S2049 pass the independent numerical
policy and segment isolation. Q1/K262144 and square S32768 pass bounded-memory
analytic O/LSE/separate-gradient oracles without a quadratic reference; the
full square S262144 is intentionally rejected by meta-tensor preflight before
forward. The pinned global `Gemma4TextAttention` layer executes S2049
backward through `fa4_global_varlen_native` with a finite hidden-state
gradient. Mixed K2048/K2049 memcheck, synccheck, and racecheck are clean, as
is Q33/K4097 memcheck. Long replays add no cache object or application key,
and all native main-object contents/resources remain byte-identical to
EXP-0013.

EXP-0015 changes only mixed-segment admission and host composition. Native
packed local/global requests may now contain leading, middle, or trailing
plateaus and `Sq=0<Sk` segments, provided their aggregate physical totals and
exact maxima remain positive. Empty-Q segments schedule no work, own no O/LSE
entries, and produce exact-zero dK/dV over their K/V slices. Neighboring
nonempty segments pass the retained numerical and hostile-mutation isolation
policies. The local long sparse scheduler omits zero-query segments, and the
global exact composer skips them without weakening its K2048 cap. Fresh cache
replays add no local object and no global object/application key; the retained
main objects are byte-identical to EXP-0010/0014. Mixed global and local long
sparse cases are clean under memcheck, synccheck, and racecheck.

The full d512 backward still uses six main launches, temporary FP32
accumulators, and atomic GQA reduction. No speed, efficiency,
deterministic-gradient, all-empty-workload, multimodal-global, or B300 claim
is made. See EXP-0005, EXP-0006, EXP-0012, EXP-0013, EXP-0014, and EXP-0015
for the immutable records.

## Local multimodal gate: EXP-0007 PASS

The exact fixed-length local predicate now has a dedicated SM90 custom-mask
path:

```text
k > q - 1024
AND (k <= q OR same nonnegative vision block)
```

It passes B1/BF16/32Q/16KV/GQA-2/d256/scale-1.0 forward O/FP32-LSE and
separate dQ/dK/dV at
S=`1,31,32,33,63,64,65,127,128,129,1023,1024,1025`. Coverage includes
all-text runtime IDs, ID zero, mixed and adjacent vision spans, adversarial
masked K/V sentinels, exact S1025 window edges, LSE-only and combined
gradients, and isolated q63/head9 transposed GQA ownership. Three repeats and
a nondefault stream passed.

Memcheck, synccheck, and racecheck are clean at S128 and S129; S1025 memcheck
is also clean. Generated code realizes M128 x N80 forward and M64 x N64
backward. Both main kernels use 168 registers and 1 KiB static shared memory;
backward has zero stack/local memory, while custom forward reports a 40-byte
stack frame and zero separate local allocation. EXP-0007 makes no performance
claim and does not promote a sparse schedule.

Acceptance is deliberately fixed-length and B1. The public `(1,S)` INT32 or
range-checked INT64 vision IDs become a private contiguous INT32 `(S,)`
auxiliary. `vision_block_ids=None` preserves the native text path. Packed
varlen uses the separate accepted EXP-0008 adapter below; the generic
Transformers 2D FA4 adapter remains xfailed.

## Packed local gate: EXP-0008 PASS

`fa4_local_varlen_forward` accepts packed THD Q `(Tq,32,256)` and distinct
K/V `(Tk,16,256)`, CUDA INT32 cumulative arrays, exact Python max lengths,
and optional packed K-stream vision/document IDs. Text-only calls retain
FA4's native lower-right causal window `(1023,0)`. Metadata calls disable the
native mask flags and use one complete predicate with
`q_abs = q + Sk - Sq`:

```text
same document
AND k > q_abs - 1024
AND (k <= q_abs OR same nonnegative vision block)
```

Forward O and FP32 LSE passed equal-length batches from S1 through S1025,
reordered segments, and native/custom lower-right matrices including
Q=`[1,31,64,129]`, K=`[33,64,128,1025]`. An independent q1/k1025 sentinel
proved the strict excluded-key-0/included-key-1 boundary. Backward passed
native and custom O-only, true `dout=None` LSE-only, and combined gradients
under EXP-0004's unchanged upstream-relative BF16 policy. LSE-only dV was
exactly zero; a transposed q1/k1025 sentinel proved exact dV ownership.

Structured packed GQA ownership, internal document blocking, repeated-ID
isolation, and hostile cross-sequence K/V mutation all passed. Three
nondefault-stream repeats were bitwise equal for O, LSE, dQ, dK, and dV.
Changing Tq/Tk totals, cumulative values, segment order, tensor contents, and
metadata contents did not create new native or custom forward/backward cache
objects.

Memcheck, synccheck, and racecheck are clean for custom packed single-block
`[63,64]/[64,64]` and multi-block `[64,65]/[64,65]` cases; q1/k1025 memcheck
is also clean. Generated code retains M128 x N80 forward and M64 x N64
backward. Both main kernels use 168 registers and 1 KiB static shared memory;
backward has zero stack/local memory, while packed custom forward reports a
104-byte stack frame and zero separately reported local allocation. See
EXP-0008 for all bounded cache keys and PTX/cubin/SASS hashes.

EXP-0008 acceptance is scoped to nonempty sequences with
`1 <= Sq <= Sk <= 1025` on SM90. It does not itself accept empty sequences,
production context above 1025, block sparsity, generic framework dispatch,
performance, or B300. EXP-0009 separately widens native text; EXP-0010 widens
metadata within its sparse resource envelope.

## Production-length native packed text gate: EXP-0009 PASS

The adapter now admits native packed text with `B>=1` and per-sequence
`1 <= Sq <= Sk <= 262144`, retaining exact BF16 32Q/16KV GQA-2 d256,
scale 1.0, distinct K/V, CUDA INT32 cumulative arrays, and lower-right causal
W1024 semantics. Metadata-bearing vision/document calls retain the S1025 guard.
No native kernel, tile, pipeline, barrier, mask callable, or generated-code
decision changed.

The bounded-work basis is the pinned scheduler itself. Forward calls
`BlockInfo.get_n_block_min_max` for each M tile to derive the causal/local
K-block interval from runtime sequence-local coordinates. Backward calls
`BlockInfo.get_m_block_min_max` for each N tile to derive the transposed bounded
Q-block interval. The adapter fixes `num_splits=1`, so long runtime maxima do
not engage the host split heuristic or create a new split specialization. This
claim does not rely on `seqlen_k_loaded`.

H100 evidence at implementation revision
`9c6b385dbae9f979aa2a38ecd0a2ed505a76cfcf` includes:

- S2048 numerical forward/`out_lse` backward against independent FP32 and BF16
  references, repeated three times on a nondefault stream. O, LSE, dK, and dV
  were bitwise equal; dQ had maximum pairwise drift `0.03125`, and every
  dQ/dK/dV repeat passed the unchanged upstream-relative BF16 policy;
- a finite exact-shape S32768 nondefault-stream `out_lse` smoke;
- a true Q=K=262144 `out_lse` forward/backward run after a corrected
  `45,231,374,336`-byte preflight with `84,465,025,024` bytes free;
- an analytic Q1/K262144 sentinel proving the strict excluded key at
  `q_abs-1024`, the first included key, exact `log(1024)` LSE, and dV ownership;
- hostile long-ragged isolation at Q=`[33,65]`, K=`[2049,4097]` against both
  references, without an observed-after-the-fact fixed dQ tolerance;
- zero memcheck/synccheck errors and zero racecheck hazards/errors/warnings at
  Q=`[64,65]`, K=`[2048,2049]`, plus zero memcheck errors for Q1/K262144.

Changing native long runtime totals and maxima added no cache objects. The
forward-only cache retained one object; a backward invocation retained four
objects total including forward, preprocess, main backward, and postprocess.
The native forward key remains
`e7b213f0ae59536df7feec9f0202f6cdace2105999b143dc3c133cbda041f176` and
the multi/multi main-backward key remains
`a3c7d28fb5372354d1d121353d7803b12e6ba713817d650b5a7288237c006ec7`.

Retained PTX is version 8.8 targeting `sm_90a`; forward and main backward keep
the accepted M128 x N80 and M64 x N64 code. Both main kernels use 168 registers,
1 KiB static shared memory, zero stack, and zero separately reported local
memory. Forward retains its HGMMA/TMA/dependency-barrier instruction path;
backward does likewise, with no LDL/STL spill traffic observed. Exact hashes
and instruction counts are in EXP-0009.

This EXP-0009 acceptance does not itself include metadata-bearing calls above
1025, empty segments, deterministic dQ, generic dispatch, performance, B300,
or another architecture. EXP-0010 separately accepts long metadata below.

## Production-length packed metadata gate: EXP-0010 PASS

Metadata-bearing local calls above S1025 now use an exact per-sequence fixed
block-sparse composition. The adapter builds Q128/K80 forward incidence and an
independent Q64/K64 backward incidence before transposing the latter into
K-block rows. A tile is present if and only if at least one in-range pair
satisfies the full document/W1024/causal-or-same-vision predicate; every
candidate remains partial and re-evaluates that predicate token by token.

The pinned SM90 sparse loader traces its empty-list branch even when runtime
mask counts are nonzero. Forward and backward therefore carry explicit
row-aligned zero-count/width-one full-list sentinels. They are compile plumbing,
not allowed tiles. Packed Q/K/V and both metadata tensors are split once with
`torch.split`; each sequence owns one fixed call, preserving separate
dQ/dK/dV without per-slice full-base scatter buffers.

Acceptance is bounded explicitly. Before host enumeration, rectangular sparse
storage is checked against 2 GiB and 10% of current free HBM. Construction
stops above `2^40` padded score slots, and final compact CUDA storage is checked
again. The maximum square rectangular bound is 94,027,776 bytes. A
semantically valid schedule above a ceiling is rejected, never approximated;
such schedules are outside EXP-0010 rather than silently claimed compatible.

H100 evidence at implementation revision
`12cfe711ad29139c7c78dcb355645ee5b9a70bb0` includes:

- 14 exact schedule tests, including exhaustive small metadata, independent
  forward/backward tile incidence, strict W1024/document cases, K262144
  sentinels, the maximum storage bound, and incremental work exhaustion;
- public adapter tests for the 2 GiB, 10%-free-HBM, and translated work-limit
  rejection paths;
- Q=`[33,65]`, K=`[2049,4097]` forward O/FP32-LSE and O-only, true LSE-only,
  and combined separate dQ/dK/dV references under the unchanged EXP-0004
  BF16 policy;
- three nondefault-stream repeats: O/LSE/dK/dV bitwise equal, dQ maximum
  pairwise drift `0.015625`, every repeat numerically valid;
- hostile repeated-ID packed/document isolation, Q1/K262144 strict-window
  ownership, and Q2049/K262144 far-future vision/different-document/far-past
  exclusion with `log(1025)` LSE and exact dV ownership;
- zero memcheck/synccheck errors and zero racecheck hazards/errors/warnings at
  Q=`[64,65]`, K=`[2048,2049]`;
- unchanged five-object bounded cache reuse across runtime lengths, metadata,
  segment order, contents, and compact widths;
- a real aggregate suite of `196 passed, 8 skipped, 1 xfailed`, plus a passing
  explicit sparse forward/backward fake compile.

Retained PTX 8.8 targets `sm_90a`. Forward contains 100 HGMMA, 56 TMA-load,
4 TMA-store, and 18 warpgroup arrive/dependency-barrier instructions. Main
backward contains 88 HGMMA, 32 TMA-load, and 20 warpgroup
arrive/dependency-barrier instructions. Both use 168 registers and 1 KiB
static shared memory. Main backward has zero stack/local and no LDL/STL.
Forward reports `LOCAL=0` but a 144-byte stack with 51 LDL/38 STL, so it is not
described as stack-traffic-free. Exact hashes and cache keys are in EXP-0010.

EXP-0010 is scoped to nonempty SM90 BF16 32Q/16KV GQA-2 d256 local attention,
scale 1.0, distinct prepared K/V, per-sequence `1 <= Sq <= Sk <= 262144`, and
requests inside the declared resource envelope. Empty segments, over-budget
schedules, deterministic dQ, generic framework dispatch/context offsets,
performance, B300, and other architectures remain excluded.

## Mixed empty packed-segment gate: EXP-0015 PASS

EXP-0015 supersedes only the earlier empty-segment exclusion. The current
packed contract accepts `B>=1` cumulative arrays with per-segment
`0 <= Sq <= Sk <= 262144`, including paired plateaus and `Sq=0<Sk`, when the
aggregate Q/K totals and the exact supplied maxima are positive. It preserves
all earlier geometry, mask, dtype, scale, distinct-K/V, memory, and sparse-work
guards. Decreasing cumulative arrays, `Sq>Sk`, malformed metadata, wrong
maxima, and all-empty physical workloads fail before backend dispatch.

CPU tests cover leading/middle/trailing plateaus, exact rejection ordering,
composer budget routing, padding restoration, and empty-slice gradient
ownership. On the real H100, native local text, dense vision/document, long
sparse metadata, and native global forward/backward pass the retained
O/FP32-LSE/separate-gradient policies. Empty-Q segments own no O/LSE entries
and their K/V slices receive exact-zero dK/dV. Hostile neighboring K/V
mutation leaves active segments unchanged. An actual pinned global
`Gemma4TextAttention` layer with one fully padded row selects native FA4,
restores exact zero O / `-inf` LSE, and returns exact-zero gradients for that
row.

Memcheck, synccheck, and racecheck are clean for global mixed-empty backward
and local long sparse vision/document backward. The H100 FakeTensor kernel
matrix reports `16 passed`, including local/global plateau inputs. This proves
the kernel wrappers can compile under FakeTensor; it does not claim framework
`torch.compile` or compiled static-cache model support. The fresh global cache
remains 28 objects / 16 unique contents / 1,956,400 bytes / 18 application
keys, while local text, dense-metadata, and sparse-metadata empty replays add no
objects. Main object hashes/resources are byte-identical to EXP-0010/0014.
No benchmark ran.

## Eager StaticCache active-prefix gate: EXP-0016 PASS

EXP-0016 admits only B1, text-only, no-active-backward eager requests backed by
the pinned `StaticCache`. The mask adapter snapshots the real scalar query
offset before `StaticLayer.update()` mutates its counter. Dispatch derives the
logical prefix as `q_offset + Sq - kv_offset`, proves one contiguous valid K
interval, and exposes zero-copy K/V prefix views to the retained FA4 route.
The logical interval must be valid, while unused physical capacity is
unreachable through the causal predicate and may contain hostile finite or NaN
values. Full and rolled local caches, where logical and physical K are equal,
retain their prior path.

The five actual pinned-layer cases cover local S32, local S1023 boundary and
first rollover, global S32/q1-K33, and global S1024/q1-K1025. Candidate active
Q/K/V are bitwise equal to operands captured from pinned eager attention. FA4
prepared O and FP32 LSE pass the frozen project-reference policies, and both
captured and candidate prepared O replay bitwise through the identical
`o_proj`. Clean and hostile caches are bitwise equal in prepared and full-layer
outputs; storage addresses remain stable, K/V stay distinct, and only intended
slots mutate. Eager/project and candidate/eager BF16 deltas are recorded
observations, not threshold gates.

Reset `position_ids` are not reinterpreted as packed metadata after trimming:
the pinned Transformers implementation deliberately skips position-ID packing
when a cache exists, and Q/K are already RoPE-prepared. Active vision or
document metadata, explicit cumulative arrays, B2 padding, nonzero underfilled
K offsets, gradients, FakeTensor, and malformed or overlapping prefix layouts
fail closed. Every fallback retains the original physical operands and mask
plan.

Fresh global capacity replays at logical Q1/K33 and physical K65/K129 retain
28 objects, 16 unique contents, 1,956,400 bytes, 18 backward application keys,
and two forward keys. The separate local gapped-layout replay retains one
object and one forward key; its SHA256 is
`366fe4840ce1f9f601e201e118dbe45ba4a86b27b813fa5ffb226c1c7a24b5ce`.
No CuTe kernel source changed, and retained main-object bytes/resources remain
unchanged.

Unfiltered memcheck is clean for local rollover and global K1025. Unfiltered
actual-layer synccheck reports NVIDIA `libcublasLt.so.12` output-projection
barriers. The project CuTe forward symbol was independently identified in the
retained object, and synccheck/racecheck filtered to
`kns=flash_attncuteflash_fwd_sm90` are clean for both cases. This is scoped
project-kernel evidence; the vendor-library report remains disclosed.

Implementation `c5ee7bec833c9617ccf323955bcafc80b72cd932` passes the final
`298 passed, 83 skipped, 8 warnings` local suite and
`369 passed, 16 skipped, 1 xfailed, 8 warnings` H100 suite. The strict artifact
is `agent_space/h100-check-exp0016.json`, SHA256
`18c46284dd978362523f0d1d8b73adfc5fd45bdfd0ace0f7ec0c3aa86c5efdae`.
This acceptance makes no framework compile, compiled StaticCache,
multimodal-cache, batched padded-cache, training, B300, or performance claim.

## Pinned Transformers integration gate: eager probe and sanitizers PASS

EXP-0011 registers one project-owned name, `gemma4_fa4_h100`, in both the
pinned Transformers attention and mask registries. Registration is idempotent
for the project callables and rejects a collision instead of replacing another
backend. The generic `flash_attention_4` entries remain untouched. Layer
routing is derived from the locked `layer_idx`, not a mutable layer label, and
the 60-layer test selects 50 local and 10 global paths. The Transformers-facing
call returns `(BSHD output, None)` because its second value is attention
weights; the project-internal result separately exposes FP32 LSE and an
observable route name.

The companion mask entry retains callable, padding/static mask, and Q/K offset
information in `Gemma4MaskPlan`. A native FA4 route is allowed only when a
structural fingerprint of the pinned Transformers masking combinators exactly
matches the locked causal, sliding, packed-document, and vision expression and
its captured metadata equals the authoritative runtime metadata. An arbitrary
callable, floating additive 4D mask, mismatched closure, or unclassifiable
expression never enters a native route. Such a request may use the preserved
exact FlexAttention mask only with autograd disabled; otherwise it fails
closed. Boolean/integer 4D masks and noncontiguous repeated document IDs fail
closed because the current fallback/composition cannot prove their exact
semantics.

Pinned Transformers presents prepared tensors as BHSD. The adapter's canonical
BHSD-to-BSHD transpose shares storage and is accepted without an unconditional
copy when the CuTe stride/alignment contract holds. Copies are localized to
packing/scattering valid tokens or materializing required V256 slabs. A
lower-right zero-Q prefix is made only by the guarded exact composer after
native HBM admission raises `GlobalBackwardBudgetExceeded`. Broadcast
`position_ids` and batch-1 2D padding/static masks are normalized across the
batch. Explicit cumulative arrays must retain every batch-row boundary, and
unsupported holey/static layouts retain their exact mask for the no-grad
fallback rather than being reinterpreted.

The eager dispatch envelope is:

| Layer/call | Accepted H100 route |
|---|---|
| Local B1 equal-length S<=1025 | `fa4_local_fixed`, including authoritative vision IDs |
| Local padded, packed, reset-position, document, or lower-right | `fa4_local_varlen`, through K262144 inside the EXP-0009/0010 envelopes |
| Global B1 equal-length with gradients | `fa4_global_fixed`, S<=2048 |
| Global B1 equal-length with gradients, S>2048 | native `fa4_global_varlen_native` through S262144 under guarded admission |
| Global B1 equal-length without gradients | `fa4_global_fixed` through S1024; `fa4_global_forward_only` for K>1024 through K262144 |
| Global batch/padded/packed/lower-right with gradients | native `fa4_global_varlen_native`; `0 <= Sq <= Sk <= 262144` per segment with positive totals/maxima; budget-only `fa4_global_varlen_composed_budget_fallback` only when every active-query K<=2048 |
| Global batch/padded/packed/lower-right without gradients | `fa4_global_varlen` when every K<=1024; `fa4_global_varlen_forward_only` when any K>1024, through K262144; both default to EXP-0041 single-launch forward |
| Exact non-native mask or unsupported eager layout | inference-only `flex_attention`, or explicit rejection when disabled |

The two long global forward routes preflight the selected output/LSE allocation
and reject when the estimate exceeds 80% of currently free HBM. The default is
EXP-0041's tuned single launch; the two-V256 allocation remains behind the
explicit rollback.

EXP-0012 validates Q33/K1025 lower-right training and packed Q=[33,65],
K=[1025,2048] training through the exact composer. EXP-0013 runs those shapes
through native THD/cu-seqlens backward, with independent dO+dLSE references
and exact-zero cross-segment gradients. Only the dedicated native budget
exception can select the composer; validation, contract, assertion, and
backend runtime failures propagate instead of silently changing routes.
EXP-0014 extends the native route through K262144. When any K exceeds 2048,
even the dedicated native budget exception propagates before forward because
the composer cannot represent that request; training never falls back to
FlexAttention.
EXP-0015 admits mixed packed plateaus on the local/global routes and exact
composer. A fully padded eager row beside a nonempty row selects native global
FA4, restores zero O / `-inf` LSE, and has exact-zero row gradients.

EXP-0011 evidence includes its original eight normal integration probe cases:

- zero-copy strided local S65 and global S33 O/LSE/separate-gradient checks;
- local padded lengths `[6,3]`, including exact zero O and `-inf` LSE tails;
- local lower-right Q3/K9 and global composed B2/S5 forward/backward;
- global no-grad Q1/K2048 and packed-varlen Q33/K2048 forward;
- pinned mask registration/transport, explicit-ID precedence, prebuilt
  generation-mask transport, registered backend execution, and an actual
  `Gemma4TextAttention` local forward/backward.

The current `--case all` adds five accepted long/native cases to that original
matrix: Q33/K1025 lower-right, mixed packed K2048, contiguous document splitting
with rebuilt cumulative arrays, odd-padded noncontiguous dO/dLSE views, and a
mixed batch with one fully padded row. All select `fa4_global_varlen_native`
when gradients are enabled.

The separate Q1/K262144 zero-score sentinel selected
`fa4_global_forward_only`, returned output `64/262144` exactly, and returned
LSE `log(262144)` within the declared FP32 tolerance. EXP-0011 memcheck,
synccheck, and racecheck are clean for the composed global B2/S5 training case
and the packed global Q33/K2048 forward-only case. EXP-0012 adds clean runs of
all three tools for fixed S1025 backward and mixed composed K2048 backward.
EXP-0013 adds memcheck/synccheck/racecheck for native mixed K2048 and the
33 one-token-segment scheduler case, plus a framework document-split memcheck
and a direct-native noncontiguous-upstream-gradient memcheck.
EXP-0014 adds clean mixed K=[2048,2049] memcheck/synccheck/racecheck and
Q33/K4097 memcheck. It also executes the pinned global attention module at
S2049 and the Q1/K262144 plus square-S32768 analytic sentinels.
EXP-0015 adds clean global mixed-empty and local long sparse-metadata runs
under all three tools, plus the actual global-layer empty-row case.

The rejected FlexAttention-backward diagnostic is retained: the first default
D512 tile exceeded H100 shared memory, and a smaller compiled B2/S5 candidate
produced a non-finite dQ. EXP-0011 therefore makes no Flex backward claim; the
production adapter rejects every gradient-capable fallback before launch.
Framework FakeTensor/raw `torch.compile(layer)` tracing is explicitly
unsupported and fails closed. The lower-level kernel-wrapper FakeTensor matrix
passes 16 cases, including mixed plateaus, and EXP-0023 separately accepts its
guarded tensor-only facade scope; neither proves raw/full-model compilation.
EXP-0013's
fresh cache has 28 objects, 16 unique contents, and
1,956,400 bytes. Fixed BSHD retains nine application keys and byte-identical
EXP-0012 main objects; native THD adds exactly nine application keys over the
SS/SM/MM scheduler classes and dKV/dQ-low/dQ-high variants. Runtime totals,
segment order, cumulative values, logical batch, legal strides, and long
replays add no specialization. EXP-0014's isolated replay retains the same
28-object/16-content/1,956,400-byte inventory, and K2049, K4097, S32768, and
K262144 add no key. No performance or B300 claim is made.
EXP-0015 retains that inventory; leading/middle/trailing empty replays in all
three native scheduler classes add no object or application key.

## Gate table

| Gate | Status | Evidence / stop condition |
|---|---|---|
| H100 identity | **PASS** | H100 80GB, CC 9.0, driver/toolkit above |
| Pinned CUDA-12.8 FA4 environment | **PASS** | Strict check including exact patch stack and profilers |
| CPU/model contract on H100 | **PASS** | Oracle status OK; full H100 suite below |
| Local d256 text forward | **PASS** | O/LSE, boundaries, GQA 1/2/4/8, stream repeat |
| Global d512 text forward | **PASS (single launch)** | EXP-0041 fixed/packed O/LSE, boundaries, rollback parity, sanitizers, SASS/resources, and S8K/S64K characterization |
| Local d256 backward | **PASS (scoped)** | EXP-0003 reject preserved; EXP-0004 matrix/oracle, stream/repeat, sanitizers, SASS |
| Global d512 backward | **PASS (composed/tuned)** | EXP-0005 reject preserved; EXP-0006 exact split path; EXP-0037 fused dKV; EXP-0038 single-launch dQ; EXP-0040 owner dKV default, fixed/packed references, sanitizers, bounded cache/memory, and S8K/S64K characterization |
| EXP-0038 implementation and record | **PASS** | Implementation `1dce18e`; strict environment, 429-pass local and 522-pass H100 suites, pinned HF oracle, schema record, exact patch stack, and rollback pass |
| EXP-0039 deterministic global backward | **PASS (opt-in)** | Implementation `e1074d8`; fixed/packed five-repeat bitwise gradients, clean sanitizers, three main launches, bounded cache/memory, S8K cost gate, and S64K smoke characterization |
| EXP-0039 implementation and record | **PASS** | Strict environment, 435-pass local and fresh 528-pass H100 suites, pinned HF oracle, schema record, exact patch stack, and fast-default rollback pass |
| EXP-0040 owner-computed dK/dV | **PASS** | Direct BF16 dK/dV ownership, no full-sequence FP32 dK/dV workspace or dKV postprocess, fixed/packed repeat/isolation, clean sanitizers, four-object cache, lower S8K/S64K medians, and tested EXP-0038/0039 rollback |
| EXP-0040 implementation and record | **PASS** | Implementation `8b28bef`; strict exact patch check, 438-pass local and fresh 531-pass H100 suites, pinned HF oracle, benchmark artifact, schema record, and rollback pass |
| EXP-0041 cooperative global forward | **PASS** | One QK/softmax pass, disjoint O256 owners, exact fixed/packed rollback parity, clean sanitizers, one main launch, and lower S8K/S64K medians |
| EXP-0041 implementation and record | **PASS** | Implementation `143ae11`; strict exact patch check, 443-pass local and 536-pass H100 suites, pinned HF oracle, benchmark artifact, schema record, and rollback pass |
| Multimodal local fwd/bwd | **PASS (fixed B1)** | EXP-0007 O/LSE/gradients, ownership, stream/repeat, sanitizers, SASS |
| Packed varlen local fwd/bwd | **PASS (scoped)** | EXP-0008 native/custom through S1025; EXP-0009 native text and EXP-0010 metadata through S262144 |
| Long vision/document metadata >1025 | **PASS (resource-scoped)** | EXP-0010 exact sparse fwd/bwd, references, isolation, K262144 sentinels, sanitizers, cache, SASS |
| Eager pinned-Transformers dispatch/context offsets | **PASS (functional/sanitizer scoped)** | Unique attention/mask pair; 50/10 routing; fixed/packed/lower-right/local vision/global no-grad paths; K262144 sentinel; focused sanitizers |
| EXP-0011 cache/artifact inventory | **PASS** | 15 paths / 9 unique contents / 976,336 bytes; bounded class/runtime reuse recorded |
| EXP-0011 final bundle record | **PASS** | Implementation `e7f26bb`; local/H100 suites, checksum verifier, and schema record pass |
| Long global backward through K2048 | **PASS (fixed/composed, resource-scoped)** | S1025/S2048 references, lower-right/packed ownership, six sanitizer runs, bounded memory/cache |
| EXP-0012 implementation and record | **PASS** | Implementation `ebe993c`; strict environment, local/H100 suites, schema record, and exact patch stack pass |
| Native packed global backward through K2048 | **PASS (resource-scoped)** | Native THD/cu-seqlens references, fixed parity, isolation, sanitizers, memory/cache/codegen evidence |
| EXP-0013 implementation and record | **PASS** | Implementation `87ff75b`; strict environment, local/H100 suites, schema record, and exact patch stack pass |
| Native packed global backward through K262144 | **PASS (resource-scoped)** | EXP-0014 tractable references, Q1/K262144 and S32768 analytic oracles, preflight rejection, integration, sanitizers, unchanged cache/codegen |
| EXP-0014 implementation and record | **PASS** | Implementation `364ea6a`; strict environment, 220-pass local and 291-pass H100 suites, schema record, and exact patch stack pass |
| Mixed empty packed segments | **PASS (positive-total scoped)** | EXP-0015 local/global references, exact-zero ownership, padded framework row, six sanitizer runs, FakeTensor wrappers, unchanged cache/codegen |
| EXP-0015 implementation and record | **PASS** | Implementation `cca09c8`; strict artifact, 313-pass H100 suite, schema record, and exact patch stack pass |
| Eager StaticCache active prefix | **PASS (B1 text/no-backward scoped)** | EXP-0016 local/global pinned layers, boundary/rollover, hostile tails, exact prepared operands, stable storage, bounded cache, project-kernel sanitizers |
| EXP-0016 implementation and record | **PASS** | Implementation `c5ee7be`; strict artifact, 369-pass H100 suite, schema record, and unchanged retained kernel objects |
| EXP-0017 no-cache fullgraph custom op | **REJECT** | Implementation `96cdfa1`; empty DynamicCache admission/mutation, unproven mask origin, default S1 graph specialization, and partial Inductor non-bitwise result falsified the declaration |
| EXP-0018 mask-boundary compiler provenance | **REJECT** | Implementation `e9a5af6`; 16/16 cache negatives passed before entry, but local default-Inductor S1023 exceeded the frozen full-layer tolerance |
| EXP-0019 whole-layer opaque compiler boundary | **REJECT** | Implementation `0adfc0a`; localization and local/eager bitwise smoke passed, but live weight metadata failed the strict ABI and the ownership-snapshot refinement produced two S1 backend captures |
| EXP-0020 inference-only pinned-weight boundary | **REJECT** | Implementation `f70c828`; snapshot-free local/eager S1 and focused ABI gates passed, but stock Inductor raised `TensorifyScalarRestartAnalysis` and then compiled the identical S1 graph, producing two backend attempts |
| EXP-0021 tensor-explicit scalar attestation | **REJECT** | Implementation `d35a97d`; CPU-FP64 scalar transport preserved eager/ABI gates but retained scalar nodes and the same two-attempt S1 restart |
| EXP-0022 comptime static scalar guards | **REJECT** | Implementation `8e3c79e`; all scalar inputs/nodes disappeared from FX, but the first identical graph still raised `TensorifyScalarRestartAnalysis` before the second backend attempt returned |
| EXP-0023 guarded compile facade | **PASS (explicit/scoped)** | Product `1756ec0`, final probe `f592971`; pinned layers 0/5, B1 BF16 no-cache text inference through S1024, bitwise local/global eager/Inductor, exact live pre-entry guards, bounded S1/S>1 graphs/FA4 keys, sanitizers, unchanged codegen |
| EXP-0042 all-layer guarded dispatch | **PASS (no-cache scoped)** | All 60 actual pinned layers bitwise at S1 under eager/Inductor, exact captured-index guards, zero graph breaks, two family graphs/FA4 classes, and S33/S1024 regression |
| EXP-0042 implementation and record | **PASS** | Implementation `a813edf`; 446-pass local and fresh 539-pass H100 suites, pinned HF oracle, strict exact patch, schema record, and bundle verifier pass |
| EXP-0024 first compiled StaticCache facade | **REJECT** | Candidate `5b28240`; global Q1/K33 arithmetic/mutation passed, but cache roots appeared as FX `get_attr` buffers rather than explicit inputs |
| EXP-0025 explicit StaticCache views | **PASS (global first-discriminator)** | Candidate `242421a`; global layer 5 Q1/K33 Inductor after eager K32 prefill, explicit K/V/counter placeholders, exact eager mutation/output, fail-closed root/view guards |
| EXP-0026 global StaticCache envelope | **PASS (global/scoped)** | Candidate `b5b8ecf`; eager/Inductor K33/K34 and K1025, opposite orders/seeds, hostile tail, exact mutation/output, K1025 sanitizers, unchanged codegen/launch encoding |
| EXP-0027 local mutable-counter cache facade | **REJECT** | Candidate `e0179fe`; K1024 boundary completed, but the first saturated roll changed the CUDA-counter tensor version despite unchanged bytes |
| EXP-0028 local cache counter transaction | **PASS (local/scoped)** | Candidate `829dc5b`; eager/Inductor K33/K34, K1024 boundary plus two rolls, exact counter/cache/output, opposite orders/seeds, hostile tail, 16 fail-closed negatives, sanitizers, unchanged native codegen |
| Raw fullgraph `torch.compile(layer)` | **UNSUPPORTED** | EXP-0017 through EXP-0022 remain rejected; EXP-0023 deliberately exposes a separately named guarded facade rather than changing this result |
| Compiled cache decode facade | **GLOBAL + LOCAL PASS (SCOPED)** | EXP-0026 accepts only pinned global layer 5; EXP-0028 accepts only pinned local layer 0. Both are B1/Q1 BF16 text inference/no-grad after eager prefill, not compiled prefill or a compiled model. |
| Benchmarks | **PASS (H100 GLOBAL, SCOPED)** | EXP-0029 accepted FA4 ruler; EXP-0034 S128 admission plus S8K hot/cold and reduced S64K comparisons against automatic-GQA and explicitly expanded SDPA |

## Exact verification commands and latest results

```bash
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  python scripts/probe_h100_local_static_cache_compile.py \
    --seed 27001 \
    --output agent_space/remote-h100-exp0028/first-discriminator.json
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  python scripts/probe_h100_local_static_cache_envelope.py \
    --seed 28001 \
    --output agent_space/remote-h100-exp0028/local-envelope-default.json
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  python scripts/probe_h100_local_static_cache_envelope.py \
    --seed 28002 --reverse-order \
    --output agent_space/remote-h100-exp0028/local-envelope-reverse.json

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  python scripts/probe_h100_global_static_cache_envelope.py \
    --seed 26001 --output /tmp/exp0026-global-default.json
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  python scripts/probe_h100_global_static_cache_envelope.py \
    --seed 26002 --reverse-order \
    --output /tmp/exp0026-global-reverse.json

bash scripts/remote/run.sh h100 env PYTHONPATH=src \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  python scripts/probe_h100_guarded_compile_facade.py \
    --family all --backend all --lengths 1,32,33,1023,1024 \
    --output agent_space/remote-h100-exp0023/h100-exp0023-expanded-guarded-facade-matrix.json
bash scripts/remote/run.sh h100 env PYTHONPATH=src \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  python scripts/probe_h100_guarded_compile_facade.py \
    --family all --backend all --lengths 1,32,33,1023,1024 \
    --scoped-size-oblivious \
    --output agent_space/remote-h100-exp0023/h100-exp0023-scoped-one-graph-matrix.json
bash scripts/remote/run.sh h100 pytest -q

bash scripts/remote/check.sh h100
bash scripts/remote/run.sh h100 \
  python scripts/check_env.py --profile h100 --expect-arch sm_90 \
    --strict --require-profilers --require-transformers
bash scripts/remote/run.sh h100 \
  python scripts/verify_model_contract.py --transformers
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_transformers_integration.py --case all
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/tmp/gemma4-fa4-exp0016-static-prefix-20260719-002 \
  python scripts/probe_h100_transformers_cache.py
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/tmp/gemma4-fa4-exp0016-local-prefix-20260719-001 \
  python scripts/probe_h100_local_varlen_cache.py --static-prefix-replay
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_transformers_integration.py \
    --case global-forward-only-max-context
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_transformers_integration.py \
    --case global-packed-empty-row --seed 11011
bash scripts/remote/run.sh h100 env FLASH_ATTENTION_FAKE_TENSOR=1 \
  pytest -q tests/test_h100_fa4.py -k fake_compile
bash scripts/remote/run.sh h100 pytest -q
```

EXP-0011's original `--case all` command passed its eight eager cases. The
current eighteen-case command additionally includes EXP-0012's two
long-backward cases, EXP-0013's document-split and noncontiguous-gradient
cases, EXP-0014's K2049/K4097 native lengths plus actual global-layer S2049
backward, EXP-0015's fully padded-row native case, and EXP-0016's five
StaticCache cases.
The separately guarded maximum-context command also passed.
Representative EXP-0011 sanitizer commands use the same probe with
`--case global-varlen-batch` and
`--case global-varlen-forward-only-long`:

```bash
for tool in memcheck synccheck racecheck; do
  bash scripts/remote/run.sh h100 compute-sanitizer --tool "$tool" \
    --report-api-errors no --error-exitcode 99 \
    python scripts/probe_h100_transformers_integration.py \
      --case global-varlen-batch
  bash scripts/remote/run.sh h100 compute-sanitizer --tool "$tool" \
    --report-api-errors no --error-exitcode 99 \
    python scripts/probe_h100_transformers_integration.py \
      --case global-varlen-forward-only-long
done
```

For `<tool>` equal to `memcheck`, `synccheck`, and `racecheck`, both cases
completed cleanly. Exact EXP-0011 cache hashes are recorded in that experiment
file. Historical local verification on the EXP-0016 implementation/test tree
`c5ee7bec833c9617ccf323955bcafc80b72cd932` is:

```text
298 passed, 83 skipped, 8 warnings
```

The aggregate H100 pytest gate on the implementation tree passed with:

```text
369 passed, 16 skipped, 1 xfailed, 8 warnings
```

The prior complete verification began at the EXP-0028 evidence revision
`49da9b8d8a199354d0ccdb8bd271b4d2301dc5f6`; after canonical formatting,
all EXP-0028 matrices/sanitizers and EXP-0023/0025/0026 regressions were
rerun at final source `ebcc1fc2623e565991151d9beeea29cbe17bfbd8`. The
complete results are:

```text
local: 415 passed, 106 skipped, 8 warnings
H100:  508 passed, 17 skipped, 1 xfailed, 8 warnings
```

The strict H100 artifact
`agent_space/remote-h100-exp0028/h100-check.json` has empty warnings and
errors. The inherited EXP-0023, EXP-0025, and EXP-0026 regression probes also
pass at that final source revision. The remote bundle verifier additionally
passes all 264 tracked checksums, compileall, Ruff lint/format, shell syntax
and ShellCheck, model-contract/skill/JSON validation, and the complete H100
suite. The schema-enforcing result ledger contains 28 valid entries after the
EXP-0028 append.

The current EXP-0039 implementation revision is
`e1074d8565d41d6cb9405c3091e305e2053e1c5f`. Its complete results are:

```text
local: 435 passed, 106 skipped, 8 warnings
H100:  528 passed, 17 skipped, 1 xfailed, 116 warnings (fresh final run)
HF oracle: 5 passed, 1 xfailed, 1 warning
```

The strict environment and exact patch-stack check has no warnings or errors.
The fresh H100 run passes compileall, Ruff lint/format, model-contract
validation, the complete H100 suite, and the pinned Transformers oracle. The
schema-enforcing result ledger contains 36 valid entries after the EXP-0039
H100 append. That record identifies the H100, CUDA, PyTorch, pinned upstream
revision, implementation SHA, and deterministic/fast S8K/S64K benchmark rows.
Focused EXP-0039 references, bitwise repeats, sanitizers, resource inspection,
launch-count profiling, memory bounds, and performance gates remain the
acceptance evidence; aggregate pytest is not a substitute. The 116 fresh-run
warnings are CuTe deprecations from compilation and existing exact-fallback
warnings, not failures.

The seventeen skips are hardware/environment-gated or FakeTensor-only tests in
normal real execution. The expected failure is the pinned Transformers generic
FA4 mask adapter, which
cannot encode Gemma's vision future-token exception. The warnings are one
retained PyTorch deprecation, upstream CuTe compile deprecations, and documented
exact-fallback warnings; the separate FakeTensor matrix reports upstream CuTe
warpgroup deprecations. Local,
global, and multimodal hardware acceptances come from the
explicit EXP-0004, EXP-0006, EXP-0007, EXP-0008, EXP-0009, EXP-0010, EXP-0011,
EXP-0012, EXP-0013, EXP-0014, EXP-0015, and EXP-0016 probe matrices and
sanitizer runs above; aggregate pytest is not presented as a substitute for
that evidence.

Representative reproduction commands follow. Run the EXP-0005 command from
its recorded source revision `d7ac7273aaed5c57923301afa6f052333e91c5b7`;
the current adapter intentionally selects EXP-0006 instead.

```bash
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0003-local-bwd \
  python scripts/probe_h100_local_backward.py --seqlen 128 --reference

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0004-local-bwd \
  python scripts/probe_h100_local_backward.py --seqlen 128 --reference \
    --comparison-policy upstream-relative

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_FAKE_TENSOR=1 \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0005-global-bwd-true-fake \
  python scripts/probe_h100_global_backward.py --seqlen 128

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0006-global-bwd-real \
  python scripts/probe_h100_global_backward.py --reference \
    --seqlen 128 --repeats 3 --nondefault-stream

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0006-global-bwd-real \
  python scripts/probe_h100_global_backward.py \
    --seqlen 33 --compile-only --structured

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0007-real \
  python scripts/probe_h100_local_backward.py --seqlen 129 \
    --vision-pattern adjacent --gradient-source out_lse --reference \
    --comparison-policy upstream-relative --structured-ownership

bash scripts/remote/run.sh h100 compute-sanitizer --tool memcheck \
  --report-api-errors no --error-exitcode 99 \
  python scripts/probe_h100_local_backward.py --seqlen 129 \
    --vision-pattern adjacent

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0008-real \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 1,31,64,129 --k-lengths 33,64,128,1025 \
    --vision-pattern all --document-pattern split \
    --gradient-source out_lse --reference \
    --comparison-policy upstream-relative

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0008-cache-custom \
  python scripts/probe_h100_local_varlen_cache.py --custom --backward

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0008-real \
  compute-sanitizer --tool racecheck --report-api-errors no \
    --error-exitcode 99 \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 64,65 --k-lengths 64,65 \
    --vision-pattern adjacent --document-pattern split \
    --gradient-source out_lse

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0009-real \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 2048 --k-lengths 2048 --gradient-source out_lse \
    --reference --comparison-policy upstream-relative --repeats 3 \
    --nondefault-stream --allow-nondeterministic-dq

bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 33,65 --k-lengths 2049,4097 \
    --gradient-source out_lse --long-text-isolation

bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 262144 --k-lengths 262144 --gradient-source out_lse

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0009-cache \
  python scripts/probe_h100_local_varlen_cache.py --long-text --backward

bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 33,65 --k-lengths 2049,4097 \
    --vision-pattern adjacent --document-pattern split \
    --gradient-source out_lse --reference \
    --comparison-policy upstream-relative --repeats 3 \
    --nondefault-stream --allow-nondeterministic-dq \
    --long-metadata-isolation

bash scripts/remote/run.sh h100 compute-sanitizer --tool racecheck \
  --report-api-errors no --error-exitcode 99 \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 64,65 --k-lengths 2048,2049 \
    --vision-pattern adjacent --document-pattern split \
    --gradient-source out_lse

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0010-cache \
  python scripts/probe_h100_local_varlen_cache.py \
    --custom --long-text --backward

bash scripts/remote/run.sh h100 env FLASH_ATTENTION_FAKE_TENSOR=1 \
  pytest -q \
    tests/test_h100_fa4.py::test_h100_local_sparse_empty_full_sentinel_fake_compile

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0012-cache-b \
  python scripts/probe_h100_global_backward.py \
    --seqlen 1024 --seqlen 1025 --seqlen 2048 --reference

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0012-cache-b \
  python scripts/probe_h100_global_backward.py \
    --seqlen 2048 --reference --repeats 3 --nondefault-stream

bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0012-cache-b \
  python scripts/probe_h100_global_backward.py \
    --seqlen 2048 --compile-only --record-memory

bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_transformers_integration.py \
    --case global-packed-long-backward --seed 12012

for tool in memcheck synccheck racecheck; do
  bash scripts/remote/run.sh h100 env \
    FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0012-cache-b \
    compute-sanitizer --tool "$tool" \
      --report-api-errors no --error-exitcode 99 \
    python scripts/probe_h100_global_backward.py --seqlen 1025 --compile-only
  bash scripts/remote/run.sh h100 \
    compute-sanitizer --tool "$tool" \
      --report-api-errors no --error-exitcode 99 \
    python scripts/probe_h100_transformers_integration.py \
      --case global-packed-long-backward --seed 12012
done
```

Representative EXP-0013 reproduction commands are:

```bash
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_global_varlen_backward.py \
    --q-lengths 65 --k-lengths 65 --reference --fixed-parity
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_global_varlen_backward.py \
    --q-lengths 33,65 --k-lengths 1025,2048 --reference --isolation \
    --record-memory
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_transformers_integration.py \
    --case global-document-split-backward
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_transformers_integration.py \
    --case global-native-varlen-noncontiguous-gradients
```

Representative EXP-0014 reproduction commands are:

```bash
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_global_varlen_backward.py \
    --q-lengths 33,65 --k-lengths 2049,4097 --reference --isolation \
    --record-memory
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_global_varlen_backward.py \
    --q-lengths 1 --k-lengths 262144 --analytic-zero \
    --analytic-pattern final --gradient-source out_lse --record-memory
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_global_varlen_backward.py \
    --q-lengths 32768 --k-lengths 32768 --analytic-zero \
    --analytic-pattern causal-boundaries --gradient-source out_lse \
    --record-memory
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_global_varlen_backward.py \
    --q-lengths 262144 --k-lengths 262144 \
    --preflight-only --expect-budget-rejection
```

Representative EXP-0015 reproduction commands are:

```bash
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_local_varlen_backward.py \
    --q-lengths 0,1,0,1,0 --k-lengths 0,1026,3,1026,0 \
    --gradient-source out_lse --reference \
    --comparison-policy upstream-relative \
    --vision-pattern mixed --document-pattern split
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_global_varlen_backward.py \
    --q-lengths 33,0,65,0 --k-lengths 1025,1,2048,0 \
    --gradient-source out_lse --reference --isolation
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_transformers_integration.py \
    --case global-packed-empty-row --seed 11011
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/.cache/exp-0015-cache \
  python scripts/probe_h100_transformers_cache.py
for tool in memcheck synccheck racecheck; do
  bash scripts/remote/run.sh h100 compute-sanitizer --tool "$tool" \
    --report-api-errors no --error-exitcode 99 \
    python scripts/probe_h100_global_varlen_backward.py \
      --q-lengths 0,33,0,65,0 --k-lengths 1,64,0,65,0 \
      --gradient-source out_lse
done
```

Representative EXP-0016 reproduction commands are:

```bash
bash scripts/remote/run.sh h100 \
  python scripts/probe_h100_transformers_integration.py --case all
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/tmp/gemma4-fa4-exp0016-static-prefix-20260719-002 \
  python scripts/probe_h100_transformers_cache.py
bash scripts/remote/run.sh h100 env \
  FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/tmp/gemma4-fa4-exp0016-local-prefix-20260719-001 \
  python scripts/probe_h100_local_varlen_cache.py --static-prefix-replay
for case in static-cache-local-first-roll static-cache-global-k1025; do
  bash scripts/remote/run.sh h100 compute-sanitizer --tool memcheck \
    --report-api-errors no --error-exitcode 99 \
    python scripts/probe_h100_transformers_integration.py --case "$case"
  for tool in synccheck racecheck; do
    bash scripts/remote/run.sh h100 compute-sanitizer --tool "$tool" \
      --kernel-name kns=flash_attncuteflash_fwd_sm90 \
      --report-api-errors no --error-exitcode 99 \
      python scripts/probe_h100_transformers_integration.py --case "$case"
  done
done
```

EXP-0028 closes the separately predeclared pinned local layer-0
`StaticSlidingWindowLayer` underfill/boundary/rollover gate while preserving
EXP-0016, EXP-0023, EXP-0025, and EXP-0026. The next compiler-integration work
must be selected and predeclared from the remaining independent widenings:
other 58 cache-layer indices, compiled prefill, cached vision/document
metadata, full-model compilation, or varlen facade inputs. EXP-0042 has closed
the no-cache layer-index widening only. Do not infer one widening
from another, skip ahead to performance tuning or B300, or rewrite the raw
`torch.compile(layer)` rejections.

## Deferred scope

B300/SM103, over-budget sparse schedules, deterministic local gradients,
raw/full-model `torch.compile`, compiled prefill, cached multimodal decode,
other-layer compiled-cache and varlen-facade integration,
backward GQA ratios beyond the exact validated model ratios (local 2 and
global 8), all-empty physical packed workloads, accepted FP8 backward, and
training-convergence claims remain deferred. H100 exact-BF16 global d512
performance tuning is active. Lower-level FakeTensor
kernel compilation and the scoped EXP-0023 facade are validated; neither proves
raw/full-model compiled execution. No H100 result is generalized to B300.
