"""Pinned CuTe DSL mask callables loaded only in the H100 environment."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from flash_attn.cute import utils


@cute.jit
def _read_fixed_block_id(
    block_ids: cute.Tensor,
    index: cute.TensorSSA,
    seqlen,
) -> cute.TensorSSA:
    """Clamp padded tile coordinates before reading the runtime ID tensor."""

    index_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
    index_fragment.store(index)
    value_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
    safe_index = cutlass.min(index_fragment[0], seqlen - 1)
    value_fragment[0] = block_ids[safe_index]
    return value_fragment.load()


@cute.jit
def gemma4_local_vision_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors,
) -> cute.TensorSSA:
    """Return the complete fixed-length Gemma local keep predicate.

    ``aux_tensors[0]`` is a contiguous INT32 ``(S,)`` tensor normalized from
    the public B=1 input. The adapter admits equal Q/K lengths only, so Q and
    KV indices share that coordinate system. ``batch``, ``head``, and
    ``seqlen_info`` remain in the required FA4 callable signature; only the
    sequence lengths affect this model-wide mask.
    """

    vision_ids = aux_tensors[0]
    q_block = _read_fixed_block_id(
        vision_ids,
        q_idx,
        seqlen_info.seqlen_q,
    )
    kv_block = _read_fixed_block_id(
        vision_ids,
        kv_idx,
        seqlen_info.seqlen_k,
    )
    zero = utils.scalar_to_ssa(0, cutlass.Int32)
    window = utils.scalar_to_ssa(1024, cutlass.Int32)
    within_left_window = kv_idx > q_idx - window
    causal_or_same_vision = (kv_idx <= q_idx) | ((q_block == kv_block) & (q_block >= zero))
    return within_left_window & causal_or_same_vision
