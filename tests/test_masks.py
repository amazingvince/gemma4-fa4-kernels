import pytest
import torch

from gemma4_fa4.masks import gemma4_attention_mask


def _m(**kwargs):
    return gemma4_attention_mask(batch_size=1, device="cpu", **kwargs)[0, 0]


def test_sliding_window_exact_lower_bound():
    mask = _m(q_len=6, kv_len=6, sliding_window=4)
    # q=5 sees k=2..5 because HF uses k > q - 4.
    assert mask[5].tolist() == [False, False, True, True, True, True]
    assert not mask[5, 1]
    assert mask[5, 2]


def test_text_local_is_causal():
    mask = _m(q_len=4, kv_len=4, sliding_window=4)
    assert not mask[1, 2]
    assert mask[1, 1]


def test_local_vision_span_is_bidirectional_inside_window():
    blocks = torch.tensor([[-1, 7, 7, 7, -1]], dtype=torch.int32)
    mask = _m(
        q_len=5,
        kv_len=5,
        sliding_window=5,
        vision_block_ids=blocks,
        allow_vision_bidirectional=True,
    )
    assert mask[1, 3]  # future token in the same positive vision block
    assert not mask[0, 1]  # text block id -1 is not bidirectional


def test_global_ignores_vision_bidirectionality():
    blocks = torch.tensor([[-1, 7, 7, 7, -1]], dtype=torch.int32)
    mask = _m(
        q_len=5,
        kv_len=5,
        sliding_window=None,
        vision_block_ids=blocks,
        allow_vision_bidirectional=False,
    )
    assert not mask[1, 3]


def test_document_boundary_blocks_cross_document_attention():
    docs = torch.tensor([[1, 1, 2, 2]], dtype=torch.int32)
    mask = _m(q_len=4, kv_len=4, sliding_window=None, document_ids=docs)
    assert not mask[3, 1]
    assert mask[3, 2]


def test_key_padding_mask():
    valid = torch.tensor([[True, True, False, False]])
    mask = _m(q_len=4, kv_len=4, sliding_window=None, key_padding_mask=valid)
    assert not mask[:, 2:].any()


def test_invalid_query_offset_is_rejected():
    with pytest.raises(ValueError, match="query positions"):
        gemma4_attention_mask(
            batch_size=1,
            q_len=4,
            kv_len=4,
            q_start=2,
            device="cpu",
            sliding_window=None,
        )


def test_default_query_alignment_is_lower_right():
    mask = _m(q_len=2, kv_len=5, sliding_window=None)
    # Query rows represent absolute positions 3 and 4.
    assert mask[0].tolist() == [True, True, True, True, False]
    assert mask[1].tolist() == [True, True, True, True, True]


def test_vision_future_exception_has_no_right_window_cap():
    blocks = torch.tensor([[5, 5, 5, 5, 5, 5]], dtype=torch.int32)
    mask = _m(
        q_len=6,
        kv_len=6,
        sliding_window=3,
        vision_block_ids=blocks,
        allow_vision_bidirectional=True,
    )
    # HF's sliding overlay is only k > q - W. Future same-block keys satisfy
    # that lower bound, so the vision exception is not capped on the right.
    assert mask[2, 4]
    assert mask[2, 5]
    # Past keys remain limited by the left window.
    assert not mask[5, 0]
