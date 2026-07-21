# Repository verification report

**Assembly date:** 2026-07-20

## Completed in the current local environment

```text
python -m compileall -q src tests scripts benchmarks
  pass

python -m pytest -q
  435 passed, 106 skipped, 8 warnings on the EXP-0039 implementation tree
  e1074d8565d41d6cb9405c3091e305e2053e1c5f
  skipped: H100 execution gates and the unavailable local pinned Transformers
           oracle; the pinned oracle runs against the remote H100 checkout

python -m ruff check --no-cache .
  pass

python -m ruff format --check src tests scripts benchmarks
  pass

bash -n scripts/*.sh scripts/remote/*.sh
  pass

writing-cute-dsl-kernels static validator
  pass: 20 Markdown files, approximately 22,014 words

sha256sum -c skills/writing-cute-dsl-kernels/SHA256SUMS.txt
  pass for every preserved skill file

JSON parse of every repository *.json file
  pass

mocked SSH/rsync remote transport smoke test
  pass: leading ~/ remained expandable on the remote shell and the run command
  preserved argument quoting
```

## Completed on the H100 target

```text
strict environment check, including exact FA4 patch stack and profilers
  pass: H100 80GB, CC 9.0, driver 580.126.09, CUDA 12.8,
        PyTorch 2.8.0+cu128, CuTe DSL 4.6.0.dev0
        quack-kernels 0.5.3; FA4/Transformers imports bound to pinned checkouts

pytest -q
  528 passed, 17 skipped, 1 xfailed, 116 warnings on the fresh EXP-0039
  implementation tree e1074d8565d41d6cb9405c3091e305e2053e1c5f;
  warnings are CuTe deprecations from fresh compilation and existing exact
  fallbacks, not failures; focused numerical, analytic, deterministic-repeat,
  resource, sanitizer, cache, integration, and codegen results below remain
  the acceptance evidence

local d256 fixed-length text forward
  pass: O/LSE, W1024 boundaries, GQA 1/2/4/8, stream repeat

composed global d512 fixed-length text forward
  pass: O/LSE through S2048, exact slab LSE, stream repeat
  pass: filtered memcheck, synccheck, racecheck
  SASS: HGMMA BF16/F32 and TMA engaged; 168 registers, no local/stack spill

local d256 backward
  pass: EXP-0004 upstream-relative BF16 gate at
        S=1,63,64,65,127,128,129,1023,1024,1025
  pass: separate finite BF16 dQ/dK/dV, exact same-input repeats,
        nondefault stream, and memcheck/synccheck/racecheck at S128/S129
  SASS: 44 HGMMA BF16/F32 and 24 UTMALDG.4D instructions; 168 registers,
        no local/stack spill
  preserved reject: EXP-0003 fixed elementwise dQ/dK envelope at S128

global d512 backward
  pass: EXP-0006 split dKV-only and D256 dQ-only main launches at
        S=1,31,32,33,63,64,65,127,128,129,511,512,513,1024
  pass: independent FP32 and upstream-style BF16 references, slab
        superposition, isolated head ownership, repeats, and nondefault stream
  pass: memcheck/synccheck/racecheck at S128/S129
  resources: dKV 222208 B and dQ 218112 B dynamic shared memory; 168
             registers, 1 KiB static shared, zero local/stack for each variant
  preserved reject: EXP-0005 unchanged direct asymmetric-GQA path
  current default: EXP-0037 full-D dKV plus EXP-0038 one-launch full-D dQ,
                   exactly two backward main launches
  pass: fixed/packed references, mixed-empty ownership, bounded memory/cache,
        nondefault stream, explicit rollback, and fixed/packed
        memcheck/synccheck/racecheck
  resources: dQ 201728 B dynamic shared, 168 registers, zero local memory,
             68 HGMMA and 56 UTMA; fixed stack zero, packed stack 24 B
  performance at unlocked clocks: S8K bwd 43.040 ms (-16.1%) and fwd_bwd
        48.659 ms (-14.8%); S64K bwd 2579.334 ms (-15.5%) and fwd_bwd
        2935.302 ms (-14.1%) versus explicit EXP-0037 rollback
  deterministic option: EXP-0039 uses two ordered V256 dKV launches and one
        ordered full-D dQ launch; fixed/packed gradients are bitwise identical
        across five repeats and all three sanitizers are clean
  deterministic cost at unlocked clocks: S8K bwd 51.635 ms versus 42.722 ms
        fast (+20.9%, within the declared 25% gate); S64K smoke 5364.229 ms
        versus 2578.563 ms fast (+108.0%), so this is a correctness option,
        not the long-context throughput route

local d256 fixed multimodal forward/backward
  pass: EXP-0007 exact Gemma vision predicate through S1025
  pass: O/LSE, separate dQ/dK/dV, LSE-only/combined gradients,
        exact GQA ownership, repeats/nondefault stream
  pass: memcheck/synccheck/racecheck at S128/S129 and S1025 memcheck

local d256 packed native/custom forward/backward
  pass: EXP-0008 nonempty B>=1, 1<=Sq<=Sk<=1025, lower-right text and
        K-stream vision/document metadata
  pass: O/LSE, true dout=None LSE-only and combined dQ/dK/dV, exact
        W1024 forward/backward sentinels, document and packed-boundary isolation
  pass: exact O/LSE/dQ/dK/dV repeats on a nondefault stream and changed-total
        native/custom forward/backward cache reuse
  pass: single/multi-block memcheck/synccheck/racecheck and K1025 memcheck
  SASS: M128xN80 forward and M64xN64 backward; 168 registers and 1 KiB
        static shared; forward 104-byte stack, backward zero stack/local

local d256 packed native text at production lengths
  pass: EXP-0009 nonempty B>=1, 1<=Sq<=Sk<=262144 on H100 SM90
  pass: S2048 independent references, repeats and nondefault stream; O/LSE/dK/dV
        bitwise, dQ pairwise max 0.03125 and every gradient oracle valid
  pass: S32768 nondefault-stream smoke and true Q=K=262144 out_lse execution
        after a 45231374336-byte preflight with 84465025024 bytes free
  pass: exact Q1/K262144 boundary sentinel and hostile Q=[33,65],
        K=[2049,4097] packed-sequence isolation
  pass: memcheck/synccheck/racecheck at Q=[64,65], K=[2048,2049] and
        Q1/K262144 memcheck
  pass: long runtime lengths retained one forward/four backward-invocation
        cache objects and unchanged native PTX/cubin/SASS resources

local d256 packed vision/document metadata at production lengths
  pass: EXP-0010 nonempty B>=1, 1<=Sq<=Sk<=262144 inside the declared
        2^40 padded-score, 2 GiB metadata, and 10%-free-HBM envelope
  pass: exact Q128xK80 forward and independently generated/transposed
        Q64xK64 backward tile incidence, with the complete token predicate
  pass: tractable O/LSE and O-only/LSE-only/combined dQ/dK/dV references,
        hostile packed/document isolation, repeats, and nondefault stream
  pass: Q1/K262144 strict-window ownership and Q2049/K262144 future-vision,
        document-isolation, LSE, and dV-ownership sentinels
  pass: memcheck/synccheck/racecheck, bounded cache reuse, and explicit sparse
        forward/backward fake compilation
  SASS: HGMMA/TMA retained; main backward has zero stack/local traffic;
        forward has LOCAL=0 plus a recorded 144-byte stack and LDL/STL traffic

pinned Transformers eager integration
  pass: exact one-file H100 patch
        patches/transformers/0001-gemma4-forward-vision-block-ids.patch
        SHA256 773950a1f1feb04f5f2e6a1d66f8953ff8905e8ca9391f804089f169da59b671
  pass: unique gemma4_fa4_h100 backend; all 8 integration probe cases passed
        across local fixed/padded/lower-right, global fixed/composed-varlen,
        long forward-only, packed forward-only, and real Gemma4TextAttention
        transport/execution routes
  pass: exact Q1/K262144 forward-only sentinel; O=0.000244140625
        (=64/262144) and LSE=12.476649284362793
  pass: memcheck, synccheck, and racecheck report zero errors for global B2/S5
        composed training and Q33/K2048 packed no-grad
  pass: isolated cache contains 15 paths, 9 unique contents, 976336 bytes
  scope at the EXP-0011 revision: eager execution only; no torch.compile,
         static-cache, global backward beyond K1024, performance, or B300 claim

H100 global d512 backward through K2048
  pass: EXP-0012 fixed S1025/S2048 O/LSE and separate dQ/dK/dV references,
        including true LSE-only and combined dO+dLSE
  pass: Q33/K1025 lower-right and packed Q=[33,65], K=[1025,2048] framework
        routes; the packed upstream gradient has exact-zero cross-segment grads
  pass: three S2048 nondefault-stream repeats; every run is inside the frozen
        policy, with no deterministic dQ/dK claim
  pass: fixed S1025 and mixed packed memcheck/synccheck/racecheck are clean
  pass: S2048 measured peak 538181632 bytes <= 605552640-byte estimate
  pass: S65/S1024/S1025/S2048 use the same three main keys and object bytes as
        EXP-0006; dLSE adds only its expected bounded preprocess object
  scope: exact fixed/composed eager training through K2048; native packed is
         accepted separately in EXP-0013; K>2048, performance,
         compiled/static-cache, and B300 remain unrun

H100 native packed global d512 backward through K2048
  pass: EXP-0013 native THD inputs and INT32 cu-seqlens for every nonempty
        segment satisfying 1 <= Sq <= Sk <= 2048; exact 32Q/4KV/GQA-8/d512,
        scale 1.0, distinct K/V, and separate BF16 dQ/dK/dV are preserved
  pass: SS, SM, and MM scheduler classes compile; B33, asymmetric lower-right,
        mixed/reversed packed, Q=K=2048, O-only, true LSE-only, and combined
        gradient cases pass the frozen FP32/upstream-relative BF16 policies
  pass: hostile packed-segment isolation, document-split cu-seqlens rebuilding,
        and noncontiguous dO/dLSE pass; LSE-only dV is exactly zero
  pass: core B33 and mixed Q=[33,65], K=[1025,2048] memcheck, synccheck, and
        racecheck report zero errors/hazards/warnings; the document-split
        framework and direct-native noncontiguous-gradient memchecks also pass
  pass: Q=K=2048 measured peak 542932992 bytes <= 610304000-byte estimate;
        mixed long measured peak 135782912 bytes <= 138453504-byte estimate
  pass: fresh fixed/native cache audit records 28 objects, 16 unique contents,
        and 1956400 bytes; all nine fixed application keys remain and their
        three main-object contents match EXP-0012, while native SS/SM/MM runtime
        lengths reuse nine native keys
  codegen: native dKV uses 168 registers, zero stack/local, and 222208 bytes of
           configured shared storage; each dQ variant uses 168 registers,
           16-byte stack, zero local, and 218112 bytes of configured shared
           storage; retained SASS contains HGMMA/TMA/barrier paths documented
           in EXP-0013
  routing: eager packed/lower-right/document training selects
           fa4_global_varlen_native; only a native HBM-budget preflight failure
           may select the exact EXP-0012 fixed composer, while validation and
           runtime errors propagate
  scope: no empty segments, K>2048 training, deterministic-gradient,
         FakeTensor/torch.compile, compiled/static-cache, performance, or B300
         claim

EXP-0013 durable inventory
  pass: experiments/EXP-0013-h100-global-native-varlen-backward.md
  pass: scripts/probe_h100_global_varlen_backward.py and
        tests/test_h100_global_varlen_backward_probe.py
  pass: expanded Transformers integration/cache probes and focused tests
  pass: agent_space/h100-check-exp0013.json records the exact H100 environment
        and c1f5be0ef864fcd716309ae1add48a4c71b8da28578a983083bbba91054a8ee0
        managed FA4 patch

H100 native packed global d512 backward through K262144
  pass: EXP-0014 extends only native nonempty THD/cu-seqlens segments to
        1 <= Sq <= Sk <= 262144 under signed-INT32 and guarded-HBM admission;
        fixed BSHD and the exact composer remain capped at S/K2048
  pass: Q33/K2049, packed Q=[33,65]/K=[2049,4097], and square S2049 pass
        independent O/LSE and separate dQ/dK/dV policies; hostile packed
        mutation retains exact gradient isolation
  pass: Q1/K262144 and square S32768 bounded analytic oracles cover exact
        lower-right counts, forward LSE, BF16-staged dQ, separate dK/dV, and
        causal-boundary dV regions without a quadratic dense reference
  pass: full square S262144 meta preflight rejects before forward; K>2048
        budget rejection cannot enter the K2048 composer or FlexAttention
  pass: actual pinned Gemma4TextAttention global layer S2049 backward routes
        through fa4_global_varlen_native with a finite hidden-state gradient
  pass: mixed K=[2048,2049] memcheck/synccheck/racecheck and Q33/K4097
        memcheck report zero issues
  pass: isolated cache remains 28 objects, 16 unique contents, and 1956400
        bytes; K2049/K4097/S32768/K262144 add no application key, and native
        main-object contents/resources remain byte-identical to EXP-0013
  scope: no empty segments, deterministic-gradient, FakeTensor/torch.compile,
         compiled/static-cache, performance, or B300 claim

EXP-0014 durable inventory
  pass: experiments/EXP-0014-h100-global-native-long-backward.md
  pass: expanded long analytic/resource/integration/cache probes and tests
  pass: agent_space/h100-check-exp0014.json records the exact H100 environment
        and 97dd1dd7c9c8efb5f2b2fd06f601a1bcc8abbe9ebcf43e9ea86365769c2e2749
        managed FA4 patch with empty warnings/errors

H100 mixed empty packed segments
  pass: EXP-0015 admits per-segment 0 <= Sq <= Sk <= 262144 across the
        accepted local/global paths while requiring positive aggregate Q/K
        totals and positive exact maxima
  pass: paired-empty and query-empty/key-nonempty segments own no O/LSE entries
        and have exact-zero dK/dV slices; decreasing, Sq>Sk, wrong-maxima, and
        all-empty physical requests fail before backend launch
  pass: local native text, dense vision/document, and long sparse metadata plus
        global native/composer paths pass reference and hostile-isolation gates
  pass: actual pinned Gemma4TextAttention(layer_idx=5) with a fully padded row
        selects fa4_global_varlen_native and restores zero O / -inf LSE plus
        exact-zero empty-row gradients
  pass: lower-level FakeTensor matrix reports 16 passed, including local/global
        mixed plateaus; framework torch.compile/static-cache remains unclaimed
  pass: global and local long-sparse memcheck/synccheck/racecheck report zero
        errors/hazards/warnings
  pass: global cache remains 28 objects, 16 unique contents, 1956400 bytes, and
        18 application keys; all plateau replays add no key/object and main
        object contents/resources remain byte-identical to EXP-0010/0014
  scope: all-empty physical workloads, deterministic gradients, framework
         torch.compile/static-cache execution, performance, and B300 unclaimed

EXP-0015 durable inventory
  pass: experiments/EXP-0015-h100-empty-packed-segments.md
  pass: expanded local/global/integration/cache probes and focused tests
  pass: agent_space/h100-check-exp0015.json records the exact H100 environment
        and eff55191c308eab9e0477fdd0f2130505f34cba7c2e18a5b942b1ca08d5927cb
        managed FA4 patch with applied_exactly true and empty warnings/errors

H100 eager StaticCache active prefixes
  pass: EXP-0016 accepts eager B1 text-only prefill/decode with no active
        backward; underfilled physical K/V storage is exposed only through its
        offset-proven contiguous active prefix
  pass: five actual pinned-layer local/global cases cover local boundary and
        first rollover plus global q1/K33 and q1/K1025 decode
  pass: prepared Q/K/V match the pinned eager operands exactly; prepared O and
        FP32 LSE pass the frozen references; hostile physical tails do not
        affect prepared or full-layer output and K/V addresses stay stable
  pass: fresh global and local capacity/stride replays add no application
        cache class or changed retained main object
  pass: unfiltered memcheck is clean; project-kernel-filtered synccheck and
        racecheck are clean, while the unfiltered vendor cublasLt synccheck
        finding remains disclosed
  scope: that eager evidence does not itself prove framework compilation,
         multimodal/batched cache, training, performance, or B300

H100 compiled global and local cache decode
  pass: EXP-0024 rejects cache roots captured as FX get_attr; EXP-0025 proves
        explicit K/V/counter views at the global Q1/K33 discriminator
  pass: EXP-0026 accepts only pinned global layer 5, B1/Q1 BF16
        inference/no-grad decode after eager prefill, through sequential K34
        and independent K1025/capacity1026 on eager and Inductor backends
  pass: opposite case orders and different seeds retain one semantic graph
        class, two capacity-shape signatures, zero breaks, exact eager output
        and cache mutation, stable storage, and hostile-tail isolation
  pass: K1025 memcheck, filtered synccheck, and filtered racecheck are clean;
        retained host object, PTX/cubin/SASS, resources, and launch encoding
        are unchanged
  pass: EXP-0027 rejects a local candidate whose mutable schema changed the
        saturated CUDA-counter version despite unchanged counter bytes
  pass: EXP-0028 accepts only pinned local layer 0, B1/Q1 BF16
        inference/no-grad decode after eager prefill, through K33/K34
        underfill, K1024 boundary fill, and repeated rollover through absolute
        position 1025 on eager and Inductor backends
  pass: opposite orders/seeds retain one semantic/runtime graph signature,
        zero breaks, exact eager output/cache/counter state, stable storage,
        hostile-tail isolation, and 16 fail-closed negative cases
  pass: local repeated-roll memcheck, filtered synccheck, and filtered
        racecheck are clean; the native host object and retained codegen class
        are unchanged
  scope: compiled prefill, cached vision/document metadata, other layers,
         raw/full-model compile, training, performance, and B300 unclaimed

EXP-0016 durable inventory
  pass: experiments/EXP-0016-h100-eager-static-cache.md
  pass: expanded integration/global-cache/local-cache probes and focused tests
  pass: agent_space/h100-check-exp0016.json records the exact H100 environment
        and managed patch stack with empty warnings/errors

experiment ledger
  pass: EXP-0001/0002 accepted and EXP-0003 rejected against source
        5b9bfab072e8cc28a7e92c9e956608db591b246c
  pass: EXP-0004 accepted against source
        49fbcad2e2b761d9de50312f03335e27236a8a13
  pass: EXP-0005 rejected against source
        d7ac7273aaed5c57923301afa6f052333e91c5b7
  pass: EXP-0006 accepted against source
        185f11cbda15ae7bd4841968c3dd46f95b670282
  pass: EXP-0007 accepted against source
        d1b7e4ad0b1ffff6e3190a4b4411603cd544afe4
  pass: EXP-0008 accepted against source
        de6450a9cf5040a7432ed7641b230bb29f835248
  pass: EXP-0009 accepted against source
        9c6b385dbae9f979aa2a38ecd0a2ed505a76cfcf
  pass: EXP-0010 accepted against source
        12cfe711ad29139c7c78dcb355645ee5b9a70bb0
  pass: EXP-0011 accepted against implementation source
        e7f26bba9b6795e3022c733cff39e060075daf57
  pass: EXP-0012 accepted against implementation source
        ebe993c5b23aae66ecbcf90ee737988546482b9a
  pass: EXP-0013 accepted against implementation source
        87ff75b1b40b55149ec5beea7480ed9ac14c9146
  pass: EXP-0014 accepted against implementation source
        364ea6ab27513a42d1b3e9f7baf9213720c1a530
  pass: EXP-0015 accepted against implementation/test source
        cca09c8211b3c643b9b311f5fec0798f84a9ea0f
  pass: EXP-0016 accepted against implementation source
        c5ee7bec833c9617ccf323955bcafc80b72cd932
  pass: EXP-0017/0018/0019/0020/0021/0022 rejected against sources
        96cdfa16d70b304718856441e06c8cbcd8281f31,
        e9a5af6f88f8d2be74256da1c89a8926d6f89fdd,
        0adfc0a2fe9df85e01b91d1bc846acf5d2f6ae12,
        f70c828ef3ea909949d3e43606a9c950776b6a8b,
        d35a97d6bbe52421f7a04e7d2ab196a2c251a733, and
        8e3c79e88fb1c76c29b5401bf7b123c5b0c83670
  pass: EXP-0023 accepted against evidence source
        f592971c09da16fc68db15ad588f94f6e1bde0be
  pass: EXP-0024 rejected against source
        5b282405ac9f6c42eed3c6269e9ae851957afc10
  pass: EXP-0025 accepted at its global discriminator against source
        242421acbddcd744a62d9411f1b303b84ee238ec
  pass: EXP-0026 accepted against implementation source
        b5b8ecf1888c519c99142f14c375ca887caa6891
  pass: EXP-0027 rejected against implementation source
        e0179fe6093bc95f8d270d0ab30d26bfc77f7d96
  pass: EXP-0028 accepted against implementation source
        829dc5bf2691d47349f8e94ebbe616c260740569; the schema-valid result
        was appended after final-source H100 reruns
  pass: EXP-0038 accepted against implementation source
        1dce18eb5e53163942ebdf1bdd974910ca72e5be; the schema-valid H100
        result contains all twelve candidate/rollback benchmark rows
  pass: EXP-0039 accepted against implementation source
        e1074d8565d41d6cb9405c3091e305e2053e1c5f; the schema-valid H100
        result contains deterministic and fast S8K/S64K benchmark rows
```

See `docs/status.md` and EXP-0001 through EXP-0039 for exact commands,
tolerances, cache keys, artifact hashes, and scoped decisions.

## Not completed in the local environment

- The local Python environment does not contain the pinned and patched
  Transformers checkout, so that oracle is skipped locally. The H100 bootstrap
  installs the exact checkout, and the remote oracle and integration probes
  run against it.
- `shellcheck` is not installed locally. ShellCheck 0.9.0 is installed on the
  H100 and the complete bundle shell check passes there.
- Nsight Compute performance counters remain unavailable on the pod because
  the driver denies access (`ERR_NVGPUCTRPERM`). EXP-0006 dynamic shared-memory
  values come directly from retained generated MLIR, not Nsight metrics.
- B300 CUDA 13.3 / PyTorch 2.13.0 cu132 remains a separate, entirely unrun
  target-host gate.
- Over-budget sparse schedules, all-empty physical packed workloads,
  deterministic local gradients, bounded-memory global backward,
  raw/full-model `torch.compile`,
  compiled prefill, cached multimodal decode, other-layer/varlen-facade
  widening, and performance beyond the scoped H100 global-causal S8K/S64K
  gates remain unrun or unsupported.
  EXP-0015 accepts mixed plateaus only under positive aggregate totals/maxima;
  EXP-0016 accepts only eager B1 text-only StaticCache active prefixes with no
  active backward. Fixed BSHD and the budget-only composer remain capped at
  S/K2048.
- `scripts/remote/bootstrap.sh h100` completed package installation but its
  unconditional FlashAttention `git fetch` returned HTTP 403. This is recorded
  as a bootstrap failure, not hidden. The existing exact pinned checkout was
  patched and `scripts/remote/check.sh h100` subsequently passed with
  `applied_exactly: true` and empty warnings/errors.

## Remaining remote evidence

```bash
bash scripts/remote/probe.sh b300
bash scripts/remote/bootstrap.sh b300
bash scripts/remote/check.sh b300
```

Do not start B300 in the H100-only scope. The next H100 gate is a bounded-memory
owner-computes global backward with an explicit deterministic dispatch
contract. It must preserve
the accepted EXP-0010 sparse envelope, EXP-0011 eager dispatch contract,
EXP-0012 fixed/composed K2048 fallback, EXP-0013 native packed ABI, and
EXP-0014 resource-scoped native K262144 backward envelope, plus EXP-0015 mixed
plateau semantics and EXP-0016 eager B1 text-only StaticCache prefixes. The
guarded EXP-0023 no-cache facade, scoped EXP-0026 global compiled-cache decode,
and scoped EXP-0028 local compiled-cache decode, plus EXP-0038's two-main-launch
fast default/rollback and EXP-0039's explicit deterministic dispatch, must also
remain intact. Any next boundary must be
separately predeclared and may not silently broaden to compiled prefill, cached
multimodal decode, other layers, full-model execution, or B300.
