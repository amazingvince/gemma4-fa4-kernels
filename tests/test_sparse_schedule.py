from __future__ import annotations

from itertools import product

import pytest
import torch

from gemma4_fa4.masks import gemma4_attention_mask
from gemma4_fa4.sparse_schedule import (
    SparseScheduleWorkLimitExceeded,
    SparseTileRows,
    build_gemma4_local_sparse_schedule,
    gemma4_local_sparse_storage_upper_bound,
)


def _allowed(vision, documents, q_length):
    k_length = len(vision)
    return gemma4_attention_mask(
        batch_size=1,
        q_len=q_length,
        kv_len=k_length,
        device="cpu",
        sliding_window=1024,
        vision_block_ids=torch.tensor([vision], dtype=torch.int32),
        document_ids=torch.tensor([documents], dtype=torch.int32),
        allow_vision_bidirectional=True,
    )[0, 0]


def _q_to_k_tile_rows(allowed, block_size):
    q_block_size, k_block_size = block_size
    rows = [set() for _ in range((allowed.shape[0] + q_block_size - 1) // q_block_size)]
    for q_index, k_index in allowed.nonzero().tolist():
        rows[q_index // q_block_size].add(k_index // k_block_size)
    return tuple(tuple(sorted(row)) for row in rows)


def _transpose_rows(rows, column_blocks):
    transposed = [[] for _ in range(column_blocks)]
    for row_index, columns in enumerate(rows):
        for column_index in columns:
            transposed[column_index].append(row_index)
    return tuple(tuple(row) for row in transposed)


def _assert_exact_coverage(vision, documents, q_length, *, fwd=(2, 3), bwd=(3, 2)):
    schedule = build_gemma4_local_sparse_schedule(
        vision,
        documents,
        q_length=q_length,
        k_length=len(vision),
        forward_block_size=fwd,
        backward_block_size=bwd,
    )
    allowed = _allowed(vision, documents, q_length)
    expected_forward = _q_to_k_tile_rows(allowed, fwd)
    expected_backward_q_to_k = _q_to_k_tile_rows(allowed, bwd)
    expected_backward = _transpose_rows(
        expected_backward_q_to_k,
        (len(vision) + bwd[1] - 1) // bwd[1],
    )
    assert schedule.forward.rows == expected_forward
    assert schedule.backward.rows == expected_backward
    for tile_rows in (schedule.forward, schedule.backward):
        for row in tile_rows.rows:
            assert row == tuple(sorted(set(row)))
            assert all(0 <= index < tile_rows.column_blocks for index in row)
    return schedule


@pytest.mark.parametrize(
    ("vision", "documents", "q_length"),
    [
        ([-1] * 9, [0] * 9, 9),
        ([-1, 7, 7, -1, 7, 7, -1, -1, 7], [0] * 9, 5),
        ([3, 3, 3, 3, 3, 3, 3, 3], [0, 0, 0, 0, 1, 1, 1, 1], 4),
        ([0, 0, -1, 0, 0, -1, 0], [2, 2, 2, 2, 2, 2, 2], 3),
        ([5, -1, 5, -1, 5, -1, 5, -1, 5, -1], [1, 2] * 5, 7),
    ],
)
def test_sparse_schedule_covers_adversarial_metadata(vision, documents, q_length):
    _assert_exact_coverage(vision, documents, q_length)


def test_sparse_schedule_exhaustive_small_metadata():
    q_length, k_length = 3, 4
    for vision in product((-1, 0, 1), repeat=k_length):
        for documents in product((0, 1), repeat=k_length):
            _assert_exact_coverage(vision, documents, q_length)


def test_sparse_schedule_includes_far_future_vision_but_not_far_past():
    k_length, q_length = 2050, 1
    q_absolute = k_length - 1
    vision = [-1] * k_length
    documents = [0] * k_length
    vision[q_absolute] = 9
    vision[0] = 9
    schedule = _assert_exact_coverage(
        vision,
        documents,
        q_length,
        fwd=(128, 80),
        bwd=(64, 64),
    )
    assert 0 not in schedule.forward.rows[0]
    assert (q_absolute - 1023) // 80 in schedule.forward.rows[0]

    q_length = k_length
    schedule = _assert_exact_coverage(
        vision,
        documents,
        q_length,
        fwd=(128, 80),
        bwd=(64, 64),
    )
    assert (k_length - 1) // 80 in schedule.forward.rows[0]


def test_sparse_schedule_model_maximum_text_is_window_bounded():
    k_length = 262_144
    schedule = build_gemma4_local_sparse_schedule(
        [-1] * k_length,
        [0] * k_length,
        q_length=1,
        k_length=k_length,
    )
    assert schedule.forward.candidate_pairs <= 14
    assert schedule.backward.candidate_pairs <= 17
    assert schedule.storage_bytes < 128 * 1024


def test_sparse_schedule_filters_document_mismatch_from_causal_band():
    vision = [-1] * 2049
    documents = [0] * 2048 + [1]
    schedule = _assert_exact_coverage(
        vision,
        documents,
        q_length=1,
        fwd=(128, 80),
        bwd=(64, 64),
    )
    assert schedule.forward.rows == ((2048 // 80,),)
    assert all(not row for row in schedule.backward.rows[:-1])
    assert schedule.backward.rows[-1] == (0,)


def test_sparse_schedule_model_maximum_future_vision_and_document_isolation():
    k_length, q_length = 262_144, 2049
    q_absolute = k_length - q_length
    vision = [-1] * k_length
    documents = [0] * k_length
    vision[q_absolute] = 7
    vision[-1] = 7
    mismatched_future = k_length - 161
    vision[mismatched_future] = 7
    documents[mismatched_future] = 1
    vision[0] = 7

    schedule = build_gemma4_local_sparse_schedule(
        vision,
        documents,
        q_length=q_length,
        k_length=k_length,
    )
    assert (k_length - 1) // 80 in schedule.forward.rows[0]
    assert mismatched_future // 80 not in schedule.forward.rows[0]
    assert 0 not in schedule.forward.rows[0]
    assert 0 in schedule.backward.rows[(k_length - 1) // 64]
    assert 0 not in schedule.backward.rows[mismatched_future // 64]


def test_sparse_schedule_model_maximum_storage_upper_bound_is_pinned():
    assert gemma4_local_sparse_storage_upper_bound(262_144, 262_144) == 94_027_776


def test_sparse_schedule_stops_during_work_budget_construction():
    with pytest.raises(SparseScheduleWorkLimitExceeded, match="construction budget"):
        build_gemma4_local_sparse_schedule(
            [0] * 64,
            [0] * 64,
            q_length=64,
            k_length=64,
            forward_block_size=(8, 8),
            backward_block_size=(4, 4),
            max_scheduled_score_slots=32 * 8 * 8,
        )


def test_sparse_schedule_rejects_invalid_contract():
    with pytest.raises(ValueError, match="Sq <= Sk"):
        build_gemma4_local_sparse_schedule([-1], [0], q_length=2, k_length=1)
    with pytest.raises(ValueError, match="match Sk"):
        build_gemma4_local_sparse_schedule([-1], [], q_length=1, k_length=1)
    schedule = build_gemma4_local_sparse_schedule(
        [-1],
        [0],
        q_length=1,
        k_length=1,
        forward_block_size=(2, 2),
        backward_block_size=(3, 2),
    )
    assert schedule.forward.rows == schedule.backward.rows == ((0,),)


def test_sparse_tile_rows_reject_duplicate_or_out_of_range_indices():
    with pytest.raises(ValueError, match="sorted and unique"):
        SparseTileRows(((1, 1),), 2, 2, 2)
    with pytest.raises(ValueError, match="out of range"):
        SparseTileRows(((2,),), 2, 2, 2)
