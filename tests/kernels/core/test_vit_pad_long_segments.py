# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ViT varlen attention with long segments run at a padded head dim.

`_varlen_attn_pad_long_segments` takes its varlen kernel as an argument, so the
partition, gather/scatter and pad arithmetic can be checked against a reference
implementation on any device. Only the gate that selects the path needs ROCm.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.v1.attention.ops import vit_attn_wrappers
from vllm.v1.attention.ops.vit_attn_wrappers import _varlen_attn_pad_long_segments

# Real segment lengths are thousands of tokens; the tests shrink the threshold so
# the reference attention stays cheap. `SEGMENTS` covers one shape per branch of
# the wrapper: long among short and several long among short (split, each long
# padded on its own plus one gathered short batch), a single long with no short
# partner (padded whole, not split), and all long -- which is neither split nor
# padded, because with no short segment being dragged along there is nothing for
# the split to save and no near-equal-segment guarantee to make padding safe.
SEGMENTS = [
    pytest.param([12, 96, 7, 20], id="one_long_among_short"),
    pytest.param([96, 5, 80, 9, 11], id="two_long_among_short"),
    pytest.param([96], id="single_long_no_short_group"),
    pytest.param([96, 72, 88], id="all_long"),
    pytest.param([70, 96, 3], id="short_segment_of_length_three"),
]
THRESHOLD = 64
HEAD_DIM = 72
PADDED_HEAD_DIM = 128
NUM_HEADS = 4

# The head dim of every kernel launch each shape is expected to produce, sorted:
# `PADDED_HEAD_DIM` for a segment padded to the tuned dim, `HEAD_DIM` for one
# handed through unpadded. Sorted because the number and shape of the launches is
# the contract; the order in which they are issued is not.
EXPECTED_LAUNCH_HEAD_DIMS = {
    (12, 96, 7, 20): [PADDED_HEAD_DIM, HEAD_DIM],
    (96, 5, 80, 9, 11): [PADDED_HEAD_DIM, PADDED_HEAD_DIM, HEAD_DIM],
    (96,): [PADDED_HEAD_DIM],
    (96, 72, 88): [HEAD_DIM],
    (70, 96, 3): [PADDED_HEAD_DIM, PADDED_HEAD_DIM, HEAD_DIM],
}


def _reference_varlen(
    q,
    k,
    v,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    causal,
    softmax_scale,
    **kwargs,
):
    """Unbatched SDPA per segment, in the packed (total, heads, dim) layout.

    Stands in for `flash_attn_varlen_func`. Deliberately ignores `max_seqlen_*`:
    those are kernel launch bounds, so a correct caller may pass any sufficient
    value and the result must not depend on which.
    """
    assert not causal and dropout_p == 0.0
    bounds = cu_seqlens_q.tolist()
    out = torch.empty_like(q)
    for start, end in zip(bounds[:-1], bounds[1:]):
        seg = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1),
            k[start:end].transpose(0, 1),
            v[start:end].transpose(0, 1),
            scale=softmax_scale,
        ).transpose(0, 1)
        out[start:end] = seg
    return out


def _spy(varlen_fn, launches):
    """Wrap a varlen kernel to record the head dim of each launch it receives."""

    def wrapper(q, k, v, **kwargs):
        launches.append(q.shape[-1])
        return varlen_fn(q, k, v, **kwargs)

    return wrapper


def _make_inputs(segments, dtype=torch.float32, device="cpu"):
    total = sum(segments)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(segments).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    gen = torch.Generator(device="cpu").manual_seed(0)
    qkv = [
        torch.randn(total, NUM_HEADS, HEAD_DIM, generator=gen, dtype=dtype).to(device)
        for _ in range(3)
    ]
    return (*qkv, cu_seqlens)


@pytest.mark.parametrize("segments", SEGMENTS)
def test_pad_long_segments_matches_unpartitioned(segments, monkeypatch):
    """Padding is numerically inert, so the split must reproduce one whole call.

    Zeros in the padded tail of k contribute nothing to any dot product and the
    padded tail of v is sliced away, with `softmax_scale` left at the unpadded
    value -- so this is an equality check, not an approximation check.

    That inertness is also why the launch plan has to be asserted separately. Every
    branch here is numerically identical to every other by construction, so values
    alone cannot witness which one ran: on `all_long`, where the wrapper neither
    splits nor pads, `got` and `want` would be the same call on the same tensors
    and the comparison could not fail for any reason. Padding every segment instead
    would pass just as silently. The spy is what makes each shape discriminating.
    """
    monkeypatch.setattr(vit_attn_wrappers, "_VIT_PAD_MIN_SEQLEN", THRESHOLD)
    q, k, v, cu_seqlens = _make_inputs(segments)
    scale = HEAD_DIM**-0.5

    launches: list[int] = []
    got = _varlen_attn_pad_long_segments(
        q,
        k,
        v,
        cu_seqlens,
        scale,
        _spy(_reference_varlen, launches),
        {},
        PADDED_HEAD_DIM,
    )
    want = _reference_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(segments),
        max_seqlen_k=max(segments),
        dropout_p=0.0,
        causal=False,
        softmax_scale=scale,
    )
    assert (
        sorted(launches, reverse=True) == EXPECTED_LAUNCH_HEAD_DIMS[tuple(segments)]
    ), f"unexpected launch plan for {segments}: {launches}"
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("segments", SEGMENTS)
def test_rows_are_routed_back_to_their_own_positions(segments, monkeypatch):
    """Guard the gather/scatter indices and the pad round-trip directly.

    The equality test above would still pass if two segments of equal length had
    their outputs swapped. Here each row of q carries its own packed-buffer index
    and the kernel is the identity, so a row is correct only if it left and
    returned to the same position -- through `torch.arange`/`torch.cat` on the
    short group and through pad-to-128-then-slice on the long ones.
    """
    monkeypatch.setattr(vit_attn_wrappers, "_VIT_PAD_MIN_SEQLEN", THRESHOLD)
    _, _, _, cu_seqlens = _make_inputs(segments)
    total = sum(segments)
    q = (
        torch.arange(total, dtype=torch.float32)
        .view(total, 1, 1)
        .expand(total, NUM_HEADS, HEAD_DIM)
        .contiguous()
    )

    def identity_varlen(q_, k_, v_, **kwargs):
        return q_

    out = _varlen_attn_pad_long_segments(
        q, q, q, cu_seqlens, 1.0, identity_varlen, {}, PADDED_HEAD_DIM
    )
    torch.testing.assert_close(out, q, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the padded path is gated on ROCm aiter"
)
def test_below_threshold_matches_the_unpartitioned_kernel():
    """No segment reaches the threshold, so the wrapper must not partition at all.

    Bit-exactness against a direct varlen call is the claim worth testing: the
    optimisation is opt-in on shape, and a batch that does not qualify has to
    come back byte-identical to what it would have produced without this change.
    """
    pytest.importorskip("aiter")
    from aiter import flash_attn_varlen_func

    from vllm.v1.attention.ops.vit_attn_wrappers import vit_flash_attn_wrapper

    segments = [512, 300, 128]
    assert max(segments) < vit_attn_wrappers._VIT_PAD_MIN_SEQLEN
    q, k, v, cu_seqlens = _make_inputs(segments, torch.bfloat16, "cuda")
    scale = HEAD_DIM**-0.5

    got = vit_flash_attn_wrapper(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        1,
        True,
        None,
        scale,
        cu_seqlens,
        torch.tensor(max(segments), dtype=torch.int32),
    )
    want = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(segments),
        max_seqlen_k=max(segments),
        dropout_p=0.0,
        causal=False,
        softmax_scale=scale,
        window_size=(-1, -1),
    )
    torch.testing.assert_close(got.squeeze(0), want, rtol=0, atol=0)


# `cu_seqlens` does not always end at the last real segment. For encoder CUDA graph
# capture `prepare_encoder_metadata` pads it up to a fixed sequence count by repeating
# the final offset, so the tail of the batch is zero-length segments. They address no
# rows, which is exactly why no output value can witness them -- but classified as
# "short" they make `short_spans` non-empty on every captured batch, which forces the
# split+pad leg onto shapes that must not take it. The launch plan is the observable.
ZERO_PAD_COUNTS = [pytest.param(1, id="one_empty"), pytest.param(5, id="five_empty")]


@pytest.mark.parametrize("segments", SEGMENTS)
@pytest.mark.parametrize("n_empty", ZERO_PAD_COUNTS)
def test_trailing_empty_segments_do_not_change_the_launch_plan(
    segments, n_empty, monkeypatch
):
    """Graph-capture padding must be inert, in the launch plan and in the values.

    Without the filter this fails on `all_long` and `single_long_no_short_group` --
    the two shapes whose whole point is *not* to split -- and it fails silently:
    `torch.cat` over empty aranges yields an empty index, so the extra shorts call
    writes nothing and every value still matches. Asserting the plan is what makes
    the regression visible; asserting the values is what keeps the filter honest.
    """
    monkeypatch.setattr(vit_attn_wrappers, "_VIT_PAD_MIN_SEQLEN", THRESHOLD)
    scale = HEAD_DIM**-0.5
    q, k, v, cu_seqlens = _make_inputs(segments)
    # Same totals, so `_make_inputs` returns the same q/k/v and the two runs are
    # comparable tensor-for-tensor rather than only distributionally.
    _, _, _, padded_cu_seqlens = _make_inputs([*segments, *([0] * n_empty)])

    plain_launches: list[int] = []
    padded_launches: list[int] = []
    plain = _varlen_attn_pad_long_segments(
        q,
        k,
        v,
        cu_seqlens,
        scale,
        _spy(_reference_varlen, plain_launches),
        {},
        PADDED_HEAD_DIM,
    )
    padded = _varlen_attn_pad_long_segments(
        q,
        k,
        v,
        padded_cu_seqlens,
        scale,
        _spy(_reference_varlen, padded_launches),
        {},
        PADDED_HEAD_DIM,
    )

    assert (
        sorted(padded_launches, reverse=True)
        == EXPECTED_LAUNCH_HEAD_DIMS[tuple(segments)]
    ), f"{n_empty} empty segments changed the plan for {segments}: {padded_launches}"
    assert sorted(padded_launches) == sorted(plain_launches)
    torch.testing.assert_close(padded, plain, rtol=0, atol=0)


def test_all_empty_segments_return_without_launching_a_kernel(monkeypatch):
    """An entirely empty batch must not reach the span arithmetic or the kernel.

    The gate only enters the helper once `max_seqlen` reaches the threshold, and a
    caller may pass a `max_seqlen` that is not derived from these bounds (the Qwen3-VL
    encoder passes an override during capture), so this shape is not provably
    unreachable -- and `max()` over no spans raises rather than degrading. Asserting
    on zero launches rather than one pins the part that matters: no kernel is handed
    a `max_seqlen` of zero, which is a value no other path through this helper
    produces and so a value no other test covers.
    """
    monkeypatch.setattr(vit_attn_wrappers, "_VIT_PAD_MIN_SEQLEN", THRESHOLD)
    cu_seqlens = torch.zeros(4, dtype=torch.int32)
    q = torch.zeros(0, NUM_HEADS, HEAD_DIM)

    launches: list[int] = []
    out = _varlen_attn_pad_long_segments(
        q,
        q,
        q,
        cu_seqlens,
        HEAD_DIM**-0.5,
        _spy(_reference_varlen, launches),
        {},
        PADDED_HEAD_DIM,
    )

    assert launches == [], "an empty batch must not reach the varlen kernel at all"
    assert out.shape == q.shape
