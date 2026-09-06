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
