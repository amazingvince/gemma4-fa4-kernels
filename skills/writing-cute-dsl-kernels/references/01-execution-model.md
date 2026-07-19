# Execution Model, Types, and Control Flow

## Purpose

Use this reference to reason about what runs in ordinary Python, what executes during CuTe DSL compilation, and what becomes GPU runtime code. Many kernel defects begin by confusing these stages.

## The Three Worlds

| World | Runs where/when | Typical objects | Safe uses |
|---|---|---|---|
| Ordinary Python | Before invoking a compiled executor | framework tensors, configuration, benchmark harness, source inspection | input validation, selecting a specialization, caching executors, fallback dispatch |
| Meta-stage/JIT tracing | Host during compilation | proxy values, static layouts, `Constexpr`, Python classes/functions evaluated by the DSL | constructing layouts, selecting atoms, unrolling static structure, generating launch logic |
| Object-stage/device runtime | GPU after launch | dynamic scalar values, pointers/tensors, thread/block indices, registers, SMEM/TMEM | loads, compute, predicates, barriers, runtime loops and branches |

Python `print()` observes meta-stage values. Device-side `cute.printf()` observes object-stage values. A proxy printed during compilation is not the eventual data value.

## Decorators and Calls

The durable mental model is:

- A `@cute.jit` function is a JIT-compiled host-side entry point and may also be inlined into other DSL functions.
- A `@cute.kernel` function is a GPU kernel symbol launched from a JIT function.
- Python can call the JIT entry point.
- A JIT function can launch a kernel.
- JIT and kernel code may call compile-time helper functions that the DSL can inline.
- A kernel does not launch another `@cute.kernel` through the ordinary decorator call model.

Keep launch policy in the JIT wrapper and device indexing/compute in the kernel.

## Hybrid Compilation

The default preprocessor combines AST rewriting of structured control flow with tracing of operations. This lets runtime loops and branches remain structured while Python metaprogramming builds static pieces.

Tracing-only mode is suitable only for guaranteed straight-line code. It can freeze an observed branch or iteration count into the trace. Do not select it merely to reduce compile latency.

## Static and Dynamic Values

### Make a value static when

- it chooses an instruction family or architecture path;
- it changes tile shape, stage count, cluster shape, vector width, layout rank, or memory allocation;
- a Python loop must be unrolled;
- it selects a callable epilogue or algorithmic policy;
- static type/layout information unlocks a meaningful compiler optimization.

### Keep a value dynamic when

- it is a normal tensor extent or stride;
- one compiled kernel should cover many runtime shapes;
- specializing it creates a large cache surface;
- it does not alter generated code structure.

### Specialization budget

List every constexpr axis and its possible values. Multiply cardinalities. Include dtype, transpose, alignment class, tile, stage count, epilogue, architecture, and optional feature flags. Treat an unbounded result as a design defect.

### Divisibility and alignment information

When runtime values have known divisibility, communicate only guarantees that are true for every call. Incorrect divisibility or alignment metadata can cause miscompilation or invalid vector access.

## Layout Staticness

A layout can contain static and dynamic shape/stride components. A fully static layout may become part of the JIT signature and cause specialization. Framework tensors passed directly are generally interpreted through dynamic layout information; an explicit DLPack conversion can create a more static view depending on how it is constructed.

Do not confuse:

- **logical shape known to Python** with
- **shape encoded as a static DSL type**.

A static layout used with a mismatched runtime allocation is dangerous. Validate the contract at the wrapper boundary.

## Scalar and Container Semantics

CuTe DSL supports a useful Python subset, not arbitrary Python semantics.

- Use DSL/native scalar types deliberately when width and signedness matter.
- Dynamic Python integer/float values may lower to default native widths; do not rely on host Python’s unbounded integer semantics.
- Python lists, tuples, dictionaries, and class structure are usually meta-time structure. Do not expect arbitrary dynamic mutation on the GPU.
- Use supported struct-like JIT arguments for grouped metadata rather than deeply nested mutable Python objects.
- Avoid global mutable state in compiled functions.
- Keep returned values and dependent types simple and supported by the installed version.

## Control Flow

### Runtime loops

Use a runtime loop for dynamic extents or when code size would grow excessively. The loop-carried values must maintain compatible types. Avoid changing a value from scalar to tuple/tensor or changing a layout type across iterations.

### Compile-time loops

Use a constexpr/unrolled loop when trip count is small, fixed, and unrolling exposes instruction-level structure. Avoid unrolling long K loops or shape-dependent loops: compile latency and code size can dominate.

### Compiler-pipelined loops

Where supported, compiler software-pipeline annotations can express prefetch stages. Treat such features as architecture- and version-specific. Inspect generated code and compare against an explicit pipeline before relying on them.

### Branches

A compile-time branch selects one generated path. A runtime branch emits device control flow. Runtime branches inside a participating barrier group must not skip synchronization transitions.

### Unsupported or hazardous flow

Dynamic `break`, `continue`, early return, and type-changing branch values are commonly restricted. Rewrite into:

- a loop condition;
- an active predicate;
- a static branch;
- or a state variable with type-stable updates.

Never put a return or divergent exit between a producer acquire and its matching commit/release.

## Function and Class Design

Prefer small helpers with one purpose:

- create a layout;
- choose an atom;
- construct a pipeline;
- run a copy step;
- execute an epilogue.

A helper that depends only on static values is a metaprogramming unit. A helper that manipulates runtime tensors must remain inside the supported DSL subset.

For policy objects or dataclasses:

- keep configuration immutable/frozen;
- make static fields explicit;
- avoid mutating dynamic state through ordinary Python attributes inside the kernel;
- expose a narrow callable method if used as an epilogue.

## Low-Level Instruction Escape Hatch

Use the narrowest supported layer that expresses the operation:

1. public CuTe/CUTLASS DSL operation or atom;
2. a current `cute.arch` or NVIDIA GPU dialect wrapper;
3. an isolated NVVM/LLVM-dialect helper;
4. inline PTX only when the pinned toolchain exposes no supported wrapper.

A low-level helper is a versioned architecture component, not an ordinary utility function. It must:

- declare the exact target SM, CUDA toolkit, and CuTe DSL revision it was verified against;
- have a small typed interface with explicit operand widths, register classes, address spaces, side effects, and convergence assumptions;
- state required proxy, memory, or execution fences around the instruction;
- remain behind a compile-time target/feature guard with a tested supported fallback or clear rejection path;
- include positive, edge, and negative tests plus PTX/SASS inspection proving the intended instruction was emitted;
- be reviewed whenever the compiler changes its NVVM wrappers, string/enumeration operands, constraints, or lowering.

Do not copy `llvm.inline_asm`, NVVM dialect calls, constraints, or clobbers from a production repository merely because they compile. Production code such as FlashAttention demonstrates that escape hatches can be useful, but installed source and generated code determine whether a particular wrapper remains valid.

## Call-Boundary Checklist

Before compiling:

- [ ] Device and stream are known.
- [ ] Tensor dtype, shape, stride order, pointer alignment, and aliasing are validated.
- [ ] Static arguments are hashable/stable and intentional.
- [ ] Dynamic arguments do not accidentally carry static layouts.
- [ ] Architecture specialization matches the device.
- [ ] Invalid input produces a host-side error or a documented fallback.

Before returning:

- [ ] The kernel launch uses the intended stream.
- [ ] Any framework lifetime required by DLPack/pointers remains valid.
- [ ] Asynchronous execution is not prematurely synchronized except for correctness or timing.
- [ ] Compiled executor caching is reused for repeated calls.
