#!/usr/bin/env python3
"""Prove packed local FA4 compilation is independent of runtime payload values."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch

from gemma4_fa4.h100 import fa4_local_varlen_forward
from gemma4_fa4.model_spec import SLIDING_ATTENTION


def _require_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("the packed-varlen cache probe requires an SM90 CUDA device")


def _make_inputs(
    q_lengths: list[int],
    k_lengths: list[int],
    seed: int,
    *,
    requires_grad: bool,
):
    spec = SLIDING_ATTENTION
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        sum(q_lengths),
        spec.num_q_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        sum(k_lengths),
        spec.num_kv_heads,
        spec.head_dim_qk,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=requires_grad,
    )
    v = torch.randn(
        sum(k_lengths),
        spec.num_kv_heads,
        spec.head_dim_v,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=requires_grad,
    )
    cu_q = torch.tensor(
        [0, q_lengths[0], sum(q_lengths)],
        device="cuda",
        dtype=torch.int32,
    )
    cu_k = torch.tensor(
        [0, k_lengths[0], sum(k_lengths)],
        device="cuda",
        dtype=torch.int32,
    )
    return q, k, v, cu_q, cu_k


def _metadata(k_lengths: list[int], *, reverse: bool):
    vision_parts = []
    document_parts = []
    for seqlen in k_lengths:
        positions = torch.arange(seqlen, device="cuda", dtype=torch.int32)
        vision = torch.where(positions % 5 < 3, positions // 5, -1)
        documents = (positions >= seqlen // 2).to(torch.int32)
        if reverse:
            vision = torch.flip(vision, dims=(0,))
            documents = 1 - documents
        vision_parts.append(vision)
        document_parts.append(documents)
    return torch.cat(vision_parts), torch.cat(document_parts)


def _run_payload(
    q_lengths: list[int],
    k_lengths: list[int],
    *,
    seed: int,
    custom: bool,
    reverse_metadata: bool,
    backward: bool,
) -> None:
    q, k, v, cu_q, cu_k = _make_inputs(
        q_lengths,
        k_lengths,
        seed,
        requires_grad=backward,
    )
    kwargs = {}
    if custom:
        vision, documents = _metadata(k_lengths, reverse=reverse_metadata)
        kwargs = {"vision_block_ids": vision, "document_ids": documents}
    out, lse = fa4_local_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        **kwargs,
    )
    if backward:
        torch.autograd.grad(out, (q, k, v), torch.ones_like(out))
    torch.cuda.synchronize()
    if not torch.isfinite(out).all() or not torch.isfinite(lse).all():
        raise AssertionError("packed-varlen cache payload produced non-finite output")


def _objects(cache_dir: Path) -> dict[str, str]:
    result = {}
    for path in sorted(cache_dir.rglob("*.o")):
        relative = path.relative_to(cache_dir).as_posix()
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom", action="store_true", help="exercise packed metadata callable")
    parser.add_argument("--backward", action="store_true", help="include the public backward path")
    parser.add_argument(
        "--long-text",
        action="store_true",
        help="vary native runtime maxima above the EXP-0008 ceiling",
    )
    args = parser.parse_args()
    if args.long_text and args.custom:
        parser.error("--long-text is native-only until the sparse metadata schedule is accepted")
    _require_h100()
    raw_cache_dir = os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR")
    if not raw_cache_dir:
        parser.error("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR must name an isolated cache directory")
    cache_dir = Path(raw_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if _objects(cache_dir):
        parser.error("the cache directory must not contain pre-existing object files")

    if args.long_text:
        first_q, first_k = [64, 65], [2048, 4097]
        second_q, second_k = [129, 33], [8193, 2049]
    else:
        first_q, first_k = [33, 65], [64, 65]
        second_q, second_k = [65, 34], [65, 63]

    # Both calls have the same batch count and one/multi-block selectors.
    # Their packed totals, cumulative payloads, segment order, tensor contents,
    # and (in long mode) runtime maxima differ. Custom mode also changes
    # metadata contents. None of those runtime values may specialize code.
    _run_payload(
        first_q,
        first_k,
        seed=8801,
        custom=args.custom,
        reverse_metadata=False,
        backward=args.backward,
    )
    first = _objects(cache_dir)
    if not first:
        raise AssertionError("first payload did not retain a compiled object")
    _run_payload(
        second_q,
        second_k,
        seed=8802,
        custom=args.custom,
        reverse_metadata=True,
        backward=args.backward,
    )
    second = _objects(cache_dir)
    if second != first:
        added = sorted(set(second) - set(first))
        removed = sorted(set(first) - set(second))
        raise AssertionError(
            f"runtime payload changed cache objects: added={added}, removed={removed}"
        )

    mode = ("custom" if args.custom else "native") + ("-backward" if args.backward else "-forward")
    if args.long_text:
        mode += "-long-text"
    print(f"cache_reuse mode={mode} objects={len(second)}")
    for relative, digest in second.items():
        print(f"object={relative} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
