# Kernel Module Shape

This is a structural scaffold, not a copy-paste API template. Verify every symbol against the installed CuTe DSL revision.

```python
# Ordinary Python layer
def public_op(inputs, *, policy=None, stream=None):
    """Validate contract, choose architecture/alignment path, fetch cached executor."""
    validate_device_dtype_shape_stride_alignment(inputs)
    target = query_device_capability(inputs)
    config = select_bounded_static_config(target, inputs, policy)
    executor = get_or_compile_executor(config, dynamic_signature(inputs))
    return executor(inputs, stream=stream)


# JIT launch layer
@cute.jit
def launch_op(dynamic_tensors, dynamic_sizes, static_config: cutlass.Constexpr):
    # Construct verified tensors/layout views and launch dimensions.
    # Allocate workspace/SMEM size from static policy.
    kernel(dynamic_tensors, dynamic_sizes, static_config).launch(
        grid=[...],
        block=[...],
        cluster=[...] if required else None,
        smem=...,
    )


# Device layer
@cute.kernel
def kernel(dynamic_tensors, dynamic_sizes, static_config):
    # 1. identify CTA/role
    # 2. create problem tensors and CTA coordinate
    # 3. local_tile into CTA-owned regions
    # 4. create coordinate tensors and predicates
    # 5. create SMEM/TMEM storage and architecture atom
    # 6. derive TiledCopy/TiledMMA partitions
    # 7. initialize pipeline/barriers
    # 8. prologue
    # 9. steady-state copy/compute
    # 10. drain all async/MMA groups
    # 11. predicated epilogue
    # 12. complete stores and release/deallocate resources
    pass
```

## Module Boundaries

```text
op.py                  public wrapper and dispatch
jit.py                 JIT launch function and executor cache
kernel_common.py       target-independent formulas/predicates
arch_sm80.py           verified warp-MMA policy
arch_sm90.py           verified WGMMA/TMA policy
arch_sm100.py          verified tcgen05/TMEM policy
compat/                version-specific helper adapters
reference.py           trusted implementation
tests/                 correctness, rejection, specialization tests
bench/                 reproducible benchmark and autotuner
```

Use fewer files for a small kernel, but preserve the conceptual boundaries.
