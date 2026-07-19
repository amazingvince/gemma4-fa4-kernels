# JIT, Framework Integration, Caching, and Versioning

## Version Discipline

CuTe DSL is distributed with CUTLASS and continues to evolve. The installed wheel or checked-out repository is the source of truth for code generation.

Record:

- `nvidia-cutlass-dsl` package version;
- CUTLASS git commit/tag if working from source;
- CUDA toolkit and driver;
- Python version;
- GPU and compute capability;
- framework version;
- exact API docs/examples consulted.

Pin these in reproducible environments. When upgrading, rerun correctness, sanitizer, compile-cache, and benchmark tests.

## API Verification Order

When an API is uncertain:

1. inspect the installed Python symbol/signature;
2. search the matching installed source;
3. search official examples in the same revision;
4. read generated API docs for that revision;
5. consult “latest” docs only to understand newer direction;
6. isolate any compatibility branch and test both sides.

Do not add a guessed call and depend on runtime error messages to discover the API.

## Host/JIT/Kernel Separation

A robust module has:

- ordinary Python public wrapper: validates inputs and chooses/fetches a compiled executor;
- `@cute.jit` launch function: constructs launch-time DSL objects and invokes the kernel;
- `@cute.kernel`: device indexing, movement, synchronization, and compute;
- optional pure/static helpers: layouts, atom selection, policy construction;
- reference implementation and tests outside the compiled path.

Keep framework-specific tensor handling out of the kernel body.

## JIT Arguments

Arguments are dynamic unless marked or represented as compile-time constants. Use explicit annotations for:

- scalar width/signedness;
- `cutlass.Constexpr` values;
- tensors/pointers and memory spaces;
- structured argument records.

Static callable policies, layout objects, and configuration can specialize code. Audit the cache key implications.

## Structured Arguments

For grouped problems or richer descriptors, use supported immutable structures such as named tuples, native structs, or frozen dataclasses according to installed-version support.

Prefer a flat, typed record over:

- mutable nested dictionaries;
- global state;
- arbitrary Python objects;
- containers whose length changes at runtime.

Validate array lengths and pointer ownership on the host.

## Static Versus Dynamic Layout Integration

### Direct framework tensor

Passing a framework tensor through the supported interop path typically preserves runtime shape/stride information dynamically, limiting recompilation. This is a good default for general operators.

### Explicit DLPack conversion

An explicit `from_dlpack` conversion can expose a layout with more static information, which may improve optimization but specialize the executor. Use it deliberately.

### Compact dynamic shape

When only certain extents vary, mark a compact shape dynamic while retaining useful stride order/divisibility. Validate that those guarantees hold for every caller.

### Dispatch strategy

A practical interface may have:

1. a dynamic general path;
2. one or two static/aligned hot paths;
3. a fallback to a framework implementation for unsupported shapes/devices.

Keep the number of specializations bounded and observable.

## Caching

### Implicit cache versus `cute.compile`

Current CuTe DSL documentation distinguishes two paths:

- ordinary decorated calls use the DSL's implicit in-memory/file cache;
- `cute.compile(...)` **bypasses that implicit cache and performs compilation**, returning a fixed reusable JIT executor.

Therefore an application-level cache must check its own normalized key *before* calling `cute.compile`. Calling `cute.compile` and only then inserting the result defeats the cache on every miss check.

A custom executor key normally includes every value that can change generated code, such as:

- package/repository revision and exact target architecture;
- dtype/accumulation/scale mode;
- tile, stage count, cluster/CTA-group, warp-role, and epilogue policy;
- compile-time booleans, callables, layout types, and code-generation options;
- relevant environment variables or toolchain choices.

Keep runtime tensor data, pointer identity, and ordinary dynamic dimensions out of the key unless the compiler intentionally specializes their type/layout contract. Normalize structured configuration so equal policies produce equal keys.

The implicit cache key is implementation-defined and currently incorporates generated MLIR plus DSL source/shared-library/environment state. Do not reproduce it manually; treat persistent cache artifacts as revision/toolchain/architecture bound.

Separate:

- compilation/setup timing;
- first module load/launch;
- steady-state launch.

Monitor cache hit/miss counts, variant count/size, compile latency, and invalidation after package, source, environment, driver, or toolkit changes. Do not autotune by recompiling identical candidates repeatedly.

## Fine-Grained Compilation and Export

CUTLASS 4.6 introduced experimental `cute.compile_to`, which gives callers control over the compiler output stage. Use it for artifact inspection or an explicitly designed AOT/export flow only after verifying the installed signature and output contract.

Record:

- requested output stage and compiler options;
- target architecture, toolkit, driver, package revision, and FFI mode;
- runtime module/loading requirements;
- cache and invalidation policy;
- a source path that can regenerate the artifact.

Do not substitute `compile_to` for normal executor caching, assume experimental outputs are stable across releases, or ship an artifact without target-device validation.

## Streams and Asynchrony

A framework-integrated operator must launch on the intended current stream or an explicitly passed stream. Avoid unnecessary host synchronization.

Correctness tests should include:

- non-default stream;
- producer operation before the kernel;
- consumer operation after the kernel;
- multiple streams with explicit events when supported;
- tensor lifetime across asynchronous execution.

A benchmark must synchronize around timing boundaries, not necessarily around every call.

## DLPack Lifetime

DLPack interop is zero-copy. The original allocation must stay alive as long as the compiled operation may access it. Preserve the owner/reference through asynchronous completion. Validate device, dtype, shape, stride, and alignment; DLPack transport does not guarantee a kernel-specific contract.

## Fake Tensors, TVM FFI, AOT, and Export

### Official fake-tensor contract

The CuTe DSL fake-tensor compilation flow is currently compatible only with the TVM FFI backend. A fake tensor describes shape/type/layout constraints but has no data and cannot be indexed. Compile the JIT function with TVM FFI enabled, then call the returned TVM-FFI function with compatible framework tensors.

Do not compile from a CuTe fake tensor and then invoke that executor with a `from_dlpack` tensor ABI; the official documentation identifies those ABIs as incompatible. Preserve explicit stride order, divisibility, and alignment constraints, and test their rejection paths.

### Production two-pass pattern

A production repository may layer its own framework fake-tensor mode and persistent cache on top. FlashAttention-4, for example, documents a two-pass workflow:

1. a parallel compile-only pass using framework fake tensors and persistent CuTe DSL caching;
2. a real-GPU execution pass that reuses the compiled artifacts.

Treat this as a project exemplar, not a drop-in CuTe API recipe. Reuse it only after verifying the target repository's wrapper, environment variables, cache format, and wheel version. Always retain a real-device correctness and sanitizer pass.

### Low-overhead invocation

For latency-sensitive calls, TVM FFI can accept framework tensors directly, use compiled argument checks, and integrate the current framework stream. Benchmark the actual wrapper because host overhead can dominate small kernels.

For TVM FFI, JAX, torch integration, or AOT/export:

- isolate the ABI wrapper;
- define stream and ownership semantics;
- version the compiled artifact against target architecture/toolkit;
- include shape/layout/alignment guards;
- retain a way to regenerate from source;
- test error handling for unsupported inputs.

Do not assume a compiled binary is portable across GPU architectures.

## Input Validation

At the public wrapper boundary, check:

- CUDA device and target capability;
- dtype and accumulation compatibility;
- rank and logical mode order;
- strides/contiguity required by the selected path;
- pointer alignment;
- dimension ranges and integer-width limits;
- aliasing;
- cluster/launch support;
- workspace size and lifetime;
- dynamic shared-memory limit;
- scale-factor metadata for quantized paths.

Use clear error messages that identify the violated contract and the available fallback.

## Current Limitation Awareness

Commonly documented limitations include a restricted Python subset, type stability across runtime control flow, restrictions on dynamic container behavior and early exits, limited layout integer widths, unsupported features/targets in some releases, and APIs still moving during beta.

Therefore:

- keep layout extents within supported integer range;
- avoid dynamic Python object mutation;
- express early exits as predicates or loop conditions;
- check current support for convolution, clustering features, OS/platform, and new GPU architectures;
- do not infer support from the C++ CuTe API alone.

## Compatibility Adapter Pattern

Put unstable helpers behind a small module:

```text
compat/
    version.py          # parse installed package/revision
    atoms.py            # choose verified atom/helper
    layouts.py          # call verified architecture layout utilities
    pipelines.py        # create version-specific pipeline objects
```

The kernel imports from the adapter; tests verify adapter behavior for supported versions. Do not write broad exception-catching fallbacks that silently choose a different kernel.

## Debug and Build Options Are Part of Reproducibility

Flags that retain PTX, add line information, enable a different FFI, choose a custom `ptxas`, or alter compiler options may change generated artifacts, invocation ABI, cache behavior, or performance. Record them in the environment snapshot and never compare benchmark results from silently different build modes.

Project-specific variables such as FlashAttention's CUBIN/SASS dump hooks are useful diagnostics but are not standard CuTe DSL APIs. Keep them behind repository tooling and verify their current names in source.

## Recompilation Audit

For a representative workload, log each unique compiled variant and its key. Investigate unexpected variants caused by:

- static tensor shapes;
- Python booleans or integers passed as constexpr;
- distinct layout object types;
- callable objects with unstable identity;
- framework tensors converted through different paths;
- autotune configuration objects not normalized.

A correct kernel with unbounded recompilation can be unusable in production.
