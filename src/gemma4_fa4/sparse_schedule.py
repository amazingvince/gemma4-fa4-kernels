"""Exact tile-incidence schedules for Gemma 4 local metadata attention."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass


def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class SparseScheduleWorkLimitExceeded(RuntimeError):
    """Raised before an exact sparse schedule can exceed its work budget."""


@dataclass(frozen=True)
class SparseTileRows:
    """Ordered compact adjacency from row tiles to candidate column tiles."""

    rows: tuple[tuple[int, ...], ...]
    row_block_size: int
    column_block_size: int
    column_blocks: int

    def __post_init__(self) -> None:
        if self.row_block_size <= 0 or self.column_block_size <= 0:
            raise ValueError("sparse tile sizes must be positive")
        if self.column_blocks <= 0:
            raise ValueError("a sparse schedule requires at least one column block")
        for row in self.rows:
            if tuple(sorted(set(row))) != row:
                raise ValueError("sparse row indices must be sorted and unique")
            if any(index < 0 or index >= self.column_blocks for index in row):
                raise ValueError("sparse row index is out of range")

    @property
    def compact_width(self) -> int:
        """Rectangular compact-index width; one slot is retained for empty rows."""

        return max(1, *(len(row) for row in self.rows))

    @property
    def candidate_pairs(self) -> int:
        return sum(len(row) for row in self.rows)

    @property
    def storage_bytes(self) -> int:
        """INT32 compact storage including the SM90 empty-full sentinel."""

        return 4 * len(self.rows) * (3 + self.compact_width)

    def padded_indices(self) -> tuple[tuple[int, ...], ...]:
        width = self.compact_width
        return tuple(row + (0,) * (width - len(row)) for row in self.rows)


@dataclass(frozen=True)
class Gemma4LocalSparseSchedule:
    """Forward and transposed-backward schedules for one packed sequence."""

    q_length: int
    k_length: int
    forward: SparseTileRows
    backward: SparseTileRows

    @property
    def candidate_pairs(self) -> int:
        return self.forward.candidate_pairs + self.backward.candidate_pairs

    @property
    def storage_bytes(self) -> int:
        return self.forward.storage_bytes + self.backward.storage_bytes

    def scheduled_score_slots(self, num_q_heads: int) -> int:
        """Padded physical score slots across forward and backward launches."""

        if num_q_heads <= 0:
            raise ValueError("num_q_heads must be positive")
        forward = (
            self.forward.candidate_pairs
            * self.forward.row_block_size
            * self.forward.column_block_size
        )
        backward = (
            self.backward.candidate_pairs
            * self.backward.row_block_size
            * self.backward.column_block_size
        )
        return num_q_heads * (forward + backward)


def sparse_tile_rows_storage_upper_bound(
    row_length: int,
    column_length: int,
    *,
    row_block_size: int,
    column_block_size: int,
) -> int:
    """Worst-case compact bytes including the SM90 empty-full sentinel."""

    if row_length <= 0 or column_length <= 0:
        raise ValueError("sparse row and column lengths must be positive")
    if row_block_size <= 0 or column_block_size <= 0:
        raise ValueError("sparse block sizes must be positive")
    row_blocks = _ceildiv(row_length, row_block_size)
    column_blocks = _ceildiv(column_length, column_block_size)
    return 4 * row_blocks * (3 + column_blocks)


def gemma4_local_sparse_storage_upper_bound(
    q_length: int,
    k_length: int,
    *,
    forward_block_size: tuple[int, int] = (128, 80),
    backward_block_size: tuple[int, int] = (64, 64),
) -> int:
    """Worst-case forward plus transposed-backward sparse CUDA storage."""

    if q_length <= 0 or k_length <= 0 or q_length > k_length:
        raise ValueError("sparse scheduling requires 1 <= Sq <= Sk")
    forward = sparse_tile_rows_storage_upper_bound(
        q_length,
        k_length,
        row_block_size=forward_block_size[0],
        column_block_size=forward_block_size[1],
    )
    backward = sparse_tile_rows_storage_upper_bound(
        k_length,
        q_length,
        row_block_size=backward_block_size[1],
        column_block_size=backward_block_size[0],
    )
    return forward + backward


def _index_document_positions(document_ids: Sequence[int]) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = {}
    for index, document_id in enumerate(document_ids):
        positions.setdefault(document_id, []).append(index)
    return positions


def _index_vision_pair_blocks(
    vision_block_ids: Sequence[int],
    document_ids: Sequence[int],
    *,
    k_block_size: int,
) -> dict[tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]]:
    """Map each (document, vision) pair to ordered K blocks and maxima."""

    pair_block_maxima: dict[tuple[int, int], dict[int, int]] = {}
    for k_index, (vision_id, document_id) in enumerate(
        zip(vision_block_ids, document_ids, strict=True)
    ):
        if vision_id < 0:
            continue
        pair = (document_id, vision_id)
        pair_block_maxima.setdefault(pair, {})[k_index // k_block_size] = k_index
    return {
        pair: (tuple(by_block.keys()), tuple(by_block.values()))
        for pair, by_block in pair_block_maxima.items()
    }


def _add_document_interval(
    candidates: set[int],
    key_positions: Sequence[int],
    *,
    start: int,
    end: int,
    k_block_size: int,
) -> None:
    first = bisect_left(key_positions, start)
    stop = bisect_right(key_positions, end)
    for position_index in range(first, stop):
        candidates.add(key_positions[position_index] // k_block_size)


def _candidate_q_to_k_rows(
    vision_block_ids: Sequence[int],
    document_ids: Sequence[int],
    document_positions: dict[int, list[int]],
    *,
    q_length: int,
    k_length: int,
    q_block_size: int,
    k_block_size: int,
    sliding_window: int,
    max_candidate_pairs: int | None,
    direction: str,
) -> SparseTileRows:
    """Build exact Q-tile to K-tile incidence for the complete Gemma mask."""

    q_offset = k_length - q_length
    num_q_blocks = _ceildiv(q_length, q_block_size)
    num_k_blocks = _ceildiv(k_length, k_block_size)

    pair_blocks = _index_vision_pair_blocks(
        vision_block_ids,
        document_ids,
        k_block_size=k_block_size,
    )

    rows: list[tuple[int, ...]] = []
    candidate_pairs = 0
    for q_block in range(num_q_blocks):
        q_start = q_block * q_block_size
        q_end = min(q_length, q_start + q_block_size)
        q_abs_start = q_offset + q_start
        q_abs_end = q_offset + q_end - 1
        candidates: set[int] = set()
        q_positions_by_document: dict[int, list[int]] = {}
        pair_thresholds: dict[tuple[int, int], int] = {}
        for q_absolute in range(q_abs_start, q_abs_end + 1):
            document_id = document_ids[q_absolute]
            q_positions_by_document.setdefault(document_id, []).append(q_absolute)
            vision_id = vision_block_ids[q_absolute]
            if vision_id < 0:
                continue
            pair = (document_id, vision_id)
            pair_thresholds.setdefault(pair, q_absolute)

        # For each document, merge the exact union of strict-left-window
        # intervals. This remains exact even for a nonstandard Q tile wider
        # than the model window or discontiguous document IDs.
        for document_id, q_positions in q_positions_by_document.items():
            key_positions = document_positions[document_id]
            interval_start = max(0, q_positions[0] - sliding_window + 1)
            interval_end = q_positions[0]
            for q_absolute in q_positions[1:]:
                next_start = max(0, q_absolute - sliding_window + 1)
                if next_start <= interval_end + 1:
                    interval_end = q_absolute
                    continue
                _add_document_interval(
                    candidates,
                    key_positions,
                    start=interval_start,
                    end=interval_end,
                    k_block_size=k_block_size,
                )
                interval_start = next_start
                interval_end = q_absolute
            _add_document_interval(
                candidates,
                key_positions,
                start=interval_start,
                end=interval_end,
                k_block_size=k_block_size,
            )

        for pair, threshold in pair_thresholds.items():
            blocks_and_maxima = pair_blocks.get(pair)
            if blocks_and_maxima is None:
                continue
            blocks, maxima = blocks_and_maxima
            first_future = bisect_right(maxima, threshold)
            candidates.update(blocks[first_future:])
        row = tuple(sorted(candidates))
        rows.append(row)
        candidate_pairs += len(row)
        if max_candidate_pairs is not None and candidate_pairs > max_candidate_pairs:
            raise SparseScheduleWorkLimitExceeded(
                f"exact {direction} sparse schedule exceeds its "
                f"{max_candidate_pairs}-tile construction budget"
            )

    return SparseTileRows(
        rows=tuple(rows),
        row_block_size=q_block_size,
        column_block_size=k_block_size,
        column_blocks=num_k_blocks,
    )


def _transpose_q_to_k_rows(q_to_k: SparseTileRows) -> SparseTileRows:
    rows: list[list[int]] = [[] for _ in range(q_to_k.column_blocks)]
    for q_block, k_blocks in enumerate(q_to_k.rows):
        for k_block in k_blocks:
            rows[k_block].append(q_block)
    return SparseTileRows(
        rows=tuple(tuple(row) for row in rows),
        row_block_size=q_to_k.column_block_size,
        column_block_size=q_to_k.row_block_size,
        column_blocks=len(q_to_k.rows),
    )


def build_gemma4_local_sparse_schedule(
    vision_block_ids: Sequence[int],
    document_ids: Sequence[int],
    *,
    q_length: int,
    k_length: int,
    sliding_window: int = 1024,
    forward_block_size: tuple[int, int] = (128, 80),
    backward_block_size: tuple[int, int] = (64, 64),
    num_q_heads: int = 32,
    max_scheduled_score_slots: int | None = None,
) -> Gemma4LocalSparseSchedule:
    """Build exact-coverage forward and backward candidate schedules.

    All returned candidates are partial tiles. Callers must still apply the
    complete token-level Gemma mask to every candidate element.
    """

    if q_length <= 0 or k_length <= 0 or q_length > k_length:
        raise ValueError("sparse scheduling requires 1 <= Sq <= Sk")
    if len(vision_block_ids) != k_length or len(document_ids) != k_length:
        raise ValueError("vision/document metadata must match Sk")
    if sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    if any(size <= 0 for size in (*forward_block_size, *backward_block_size)):
        raise ValueError("sparse block sizes must be positive")
    if num_q_heads <= 0:
        raise ValueError("num_q_heads must be positive")
    if max_scheduled_score_slots is not None and max_scheduled_score_slots <= 0:
        raise ValueError("max_scheduled_score_slots must be positive")

    document_positions = _index_document_positions(document_ids)
    forward_tile_score_slots = num_q_heads * forward_block_size[0] * forward_block_size[1]
    max_forward_pairs = (
        None
        if max_scheduled_score_slots is None
        else max_scheduled_score_slots // forward_tile_score_slots
    )

    forward = _candidate_q_to_k_rows(
        vision_block_ids,
        document_ids,
        document_positions,
        q_length=q_length,
        k_length=k_length,
        q_block_size=forward_block_size[0],
        k_block_size=forward_block_size[1],
        sliding_window=sliding_window,
        max_candidate_pairs=max_forward_pairs,
        direction="forward",
    )
    forward_work = forward.candidate_pairs * forward_tile_score_slots
    remaining_work = (
        None if max_scheduled_score_slots is None else max_scheduled_score_slots - forward_work
    )
    backward_tile_score_slots = num_q_heads * backward_block_size[0] * backward_block_size[1]
    max_backward_pairs = (
        None if remaining_work is None else remaining_work // backward_tile_score_slots
    )
    backward_q_to_k = _candidate_q_to_k_rows(
        vision_block_ids,
        document_ids,
        document_positions,
        q_length=q_length,
        k_length=k_length,
        q_block_size=backward_block_size[0],
        k_block_size=backward_block_size[1],
        sliding_window=sliding_window,
        max_candidate_pairs=max_backward_pairs,
        direction="backward",
    )
    backward = _transpose_q_to_k_rows(backward_q_to_k)
    schedule = Gemma4LocalSparseSchedule(
        q_length=q_length,
        k_length=k_length,
        forward=forward,
        backward=backward,
    )
    if (
        max_scheduled_score_slots is not None
        and schedule.scheduled_score_slots(num_q_heads) > max_scheduled_score_slots
    ):  # pragma: no cover - the incremental guards above enforce this
        raise SparseScheduleWorkLimitExceeded("exact sparse schedule exceeds its work budget")
    return schedule
