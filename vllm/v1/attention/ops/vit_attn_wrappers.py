# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains ops for ViT attention to be compatible with torch.compile
as there are operations here not supported by torch.compile (for instance,
`.item()` in flash attention)

Using these ops and wrapping vision blocks with `torch.compile` can speed up
throughput in vision models by ~5% relative on H100, and improve token
latencies by ~7% (see qwen2_5_vl for example usage)

To use these ops, you must have a recent version of PyTorch installed (>= 2.4.0)
"""

import itertools
from collections.abc import Callable
from typing import Any

import einops
import torch
import torch.nn.functional as F

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import direct_register_custom_op

# ROCm/CK ships no tuned FMHA instance for head_dim 72 (Qwen3-VL's ViT: hidden 1152
# over 16 heads), so those calls fall back to a generic path. Zero-padding the head
# dim up to a width that *is* tuned recovers the difference, but the pad costs three
# extra allocations and copies -- worth it only once a segment is long enough for the
# kernel gain to dominate. Map head dims that benefit to the width to pad them to.
_VIT_PAD_HEAD_DIM = {72: 128}

# Minimum segment length, in tokens, before padding pays for itself. Below this the
# pad is a net loss, so short segments stay on the native path in a single varlen
# call of their own. Chosen from a shape sweep on gfx950; it is a heuristic, not a
# fitted constant, but it is load-bearing rather than decorative: re-running the same
# shapes with this lowered to 1024 -- where the newly-eligible segments are far too
# short for the pad to pay -- turns 0.75-0.88x into 0.40-0.50x.
_VIT_PAD_MIN_SEQLEN = 8192


def _varlen_attn_pad_long_segments(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None,
    varlen_fn: Callable[..., torch.Tensor],
    kwargs: dict[str, Any],
    padded_head_dim: int,
) -> torch.Tensor:
    """Run segments at or above `_VIT_PAD_MIN_SEQLEN` with a padded head dim.

    Qwen3-VL packs images of very different sizes into one varlen call, so a batch is
    usually a few long segments among many short ones. Each long segment gets its own
    padded call; everything else is gathered into one unpadded call sized by the
    longest of *those*, then both are scattered back into the packed layout. That split
    is what stops the short segments being launched at a long segment's cost.

    When every segment is long there is nothing to separate, and what happens
    next turns on how many there are: a lone segment is padded, while a batch of
    them is handed back to the plain unpadded call -- see the comments below for
    the measurements behind both.

    Only called once `max_seqlen` has already shown that some segment reaches the
    threshold, so the host sync below is not on the common path.
    """
    head_dim = q.shape[-1]
    with gpu_sync_allowed():
        # Segment boundaries are needed as Python ints, for the slice bounds
        # below and for the launch bounds of each sub-call.
        bounds = cu_seqlens.tolist()

    def pad_head_dim(x: torch.Tensor) -> torch.Tensor:
        # softmax_scale is unchanged: the zero tail contributes exactly zero to every
        # dot product, so padding is numerically inert rather than approximate.
        padded = x.new_zeros(x.shape[0], x.shape[1], padded_head_dim)
        padded[..., :head_dim] = x
        return padded

    spans = list(zip(bounds[:-1], bounds[1:]))
    short_spans = [(s, e) for s, e in spans if e - s < _VIT_PAD_MIN_SEQLEN]
    # Recomputed from the spans rather than threaded down from `max_seqlen`, which
    # is equal to it only by way of an invariant the caller enforces. Deriving it
    # here keeps this helper correct on its own arguments.
    max_span = max(e - s for s, e in spans)

    if not short_spans:
        # Every segment is long, so there is no short segment being dragged along at a
        # long segment's cost -- the thing the split exists to prevent. Splitting here
        # only trades one launch for N, which on a uniform batch sitting at the
        # threshold measured 2.7x SLOWER than the unsplit baseline. So do not split.
        if len(spans) > 1:
            # Nor pad, once there is more than one segment. The obvious move -- pad the
            # whole batch as one call -- is fast when the segments are near-equal but
            # collapses when they are not: at 2 heads a 32x spread measured 0.14-0.23x
            # against this same unpadded call, tracking a known step function in the
            # padded kernel's cost with batch size. There is no measured-safe choice
            # here at low head counts, so take the one that cannot regress.
            return varlen_fn(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_span,
                max_seqlen_k=max_span,
                dropout_p=0.0,
                causal=False,
                softmax_scale=scale,
                **kwargs,
            )
        # A single long segment: pad it. This is the common real shape -- one image
        # large enough to reach the gate, alone on its rank -- and the best cell
        # measured, 1.30-1.34x at both head counts.
        res = varlen_fn(
            pad_head_dim(q),
            pad_head_dim(k),
            pad_head_dim(v),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_span,
            max_seqlen_k=max_span,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
            **kwargs,
        )
        return res[..., :head_dim]

    # Uninitialised is safe because the two writes below partition `spans` exactly:
    # every span is either at or above the threshold (written by the loop) or below it
    # (written via `index`), so no row of `out` is left unset.
    out = torch.empty_like(q)
    for start, end in spans:
        seg_len = end - start
        if seg_len < _VIT_PAD_MIN_SEQLEN:
            continue
        seg_cu = torch.tensor(
            [0, seg_len], dtype=cu_seqlens.dtype, device=cu_seqlens.device
        )
        res = varlen_fn(
            pad_head_dim(q[start:end]),
            pad_head_dim(k[start:end]),
            pad_head_dim(v[start:end]),
            cu_seqlens_q=seg_cu,
            cu_seqlens_k=seg_cu,
            max_seqlen_q=seg_len,
            max_seqlen_k=seg_len,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
            **kwargs,
        )
        out[start:end] = res[..., :head_dim]

    if short_spans:
        # Built from aranges rather than a mask, because `nonzero` would synchronize.
        index = torch.cat(
            [
                torch.arange(start, end, device=cu_seqlens.device)
                for start, end in short_spans
            ]
        )
        short_lens = [end - start for start, end in short_spans]
        short_cu = torch.tensor(
            [0, *itertools.accumulate(short_lens)],
            dtype=cu_seqlens.dtype,
            device=cu_seqlens.device,
        )
        short_max = max(short_lens)
        out[index] = varlen_fn(
            q[index],
            k[index],
            v[index],
            cu_seqlens_q=short_cu,
            cu_seqlens_k=short_cu,
            max_seqlen_q=short_max,
            max_seqlen_k=short_max,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
            **kwargs,
        )
    return out


def flash_attn_maxseqlen_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {}
    if is_rocm_aiter:
        from aiter import flash_attn_varlen_func

        kwargs["window_size"] = (-1, -1)
    else:
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        if not current_platform.is_rocm() and fa_version is not None:
            kwargs["fa_version"] = fa_version

    q_len = q.size(1)
    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=q.device
        )
    if max_seqlen is None:
        max_seqlen = q_len
    else:
        # `flash_attn_varlen_func` needs a Python int for kernel launch bounds.
        with gpu_sync_allowed():
            max_seqlen = max_seqlen.item()

    q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    padded_head_dim = _VIT_PAD_HEAD_DIM.get(q.shape[-1])
    # `max_seqlen` is already the largest segment length, as a Python int, on both
    # branches above: when supplied it comes from `compute_max_seqlen`, a per-segment
    # max taken host-side over a numpy array; when not, `cu_seqlens` was synthesised
    # uniform at `q_len` just above, so `q_len` is that maximum. (The two cannot
    # disagree -- the caller asserts `cu_seqlens` and `max_seqlen` are both set or
    # both None.) So this answers "does any segment reach the threshold?" for free,
    # with no extra sync on the batches where the answer is no, which is nearly all
    # of them.
    #
    # Skipped under graph capture: the partition below is data dependent, and unlike
    # `max_seqlen` -- where a conservative capture value stays valid on replay --
    # there is no partition that is correct for every batch the graph will be
    # replayed against.
    if (
        is_rocm_aiter
        and padded_head_dim is not None
        and max_seqlen >= _VIT_PAD_MIN_SEQLEN
        and not torch.cuda.is_current_stream_capturing()
    ):
        output = _varlen_attn_pad_long_segments(
            q,
            k,
            v,
            cu_seqlens,
            scale,
            flash_attn_varlen_func,
            kwargs,
            padded_head_dim,
        )
    else:
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale,
            **kwargs,
        )
    context_layer = einops.rearrange(output, "(b s) h d -> b s h d", b=batch_size)
    return context_layer


def flash_attn_maxseqlen_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


direct_register_custom_op(
    op_name="flash_attn_maxseqlen_wrapper",
    op_func=flash_attn_maxseqlen_wrapper,
    fake_impl=flash_attn_maxseqlen_wrapper_fake,
)


def vit_flash_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    is_rocm_aiter: bool,
    fa_version: int | None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.flash_attn_maxseqlen_wrapper(
        q,
        k,
        v,
        batch_size,
        is_rocm_aiter,
        fa_version,
        scale,
        cu_seqlens,
        max_seqlen,
    )


def vit_aiter_fp8_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    batch_size: int,
    output_dtype: torch.dtype,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return rocm_aiter_ops.fp8_attn_wrapper(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        batch_size,
        output_dtype,
        scale,
        cu_seqlens,
        max_seqlen,
    )


def triton_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

    q_len = q.size(1)
    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=q.device
        )
    if max_seqlen is None:
        max_seqlen = q_len
    else:
        # `context_attention_fwd` needs a Python int.
        with gpu_sync_allowed():
            max_seqlen = max_seqlen.item()

    q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    output = torch.empty_like(q)
    context_attention_fwd(
        q,
        k,
        v,
        output,
        b_start_loc=cu_seqlens[:-1],
        b_seq_len=cu_seqlens[1:] - cu_seqlens[:-1],
        max_input_len=max_seqlen,
        is_causal=False,
        sliding_window_q=None,
        sliding_window_k=None,
        softmax_scale=scale,
    )

    context_layer = einops.rearrange(output, "(b s) h d -> b s h d", b=batch_size)
    return context_layer


def triton_attn_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


direct_register_custom_op(
    op_name="triton_attn_wrapper",
    op_func=triton_attn_wrapper,
    fake_impl=triton_attn_wrapper_fake,
)


def vit_triton_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.triton_attn_wrapper(
        q,
        k,
        v,
        batch_size,
        scale,
        cu_seqlens,
        max_seqlen,
    )


def apply_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Input shape:
    (batch_size x seq_len x num_heads x head_size)
    """
    q, k, v = (einops.rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
    output = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
    )
    output = einops.rearrange(output, "b h s d -> b s h d ")
    return output


# TODO: Once we have a torch 2.10, we can use tensor slices
# so we won't need to wrap this in custom ops
def torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    # Never remove the contiguous logic for ROCm
    # Without it, hallucinations occur with the backend
    if current_platform.is_rocm():
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    if cu_seqlens is None:
        return apply_sdpa(q, k, v, scale=scale, enable_gqa=enable_gqa)

    outputs = []

    # `torch.split` needs Python int sizes.
    with gpu_sync_allowed():
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    q_chunks = torch.split(q, lens, dim=1)
    k_chunks = torch.split(k, lens, dim=1)
    v_chunks = torch.split(v, lens, dim=1)
    for q_i, k_i, v_i in zip(q_chunks, k_chunks, v_chunks):
        output_i = apply_sdpa(q_i, k_i, v_i, scale=scale, enable_gqa=enable_gqa)
        outputs.append(output_i)
    context_layer = torch.cat(outputs, dim=1)
    return context_layer


def torch_sdpa_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
    cu_seqlens: torch.Tensor | None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return torch.empty_like(q)


direct_register_custom_op(
    op_name="torch_sdpa_wrapper",
    op_func=torch_sdpa_wrapper,
    fake_impl=torch_sdpa_wrapper_fake,
)


def vit_torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return torch.ops.vllm.torch_sdpa_wrapper(
        q, k, v, scale, cu_seqlens, enable_gqa=enable_gqa
    )


def flashinfer_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    o_data_type: torch.dtype | None = None,
) -> torch.Tensor:
    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

    is_reshaped = q.dim() == 4

    if is_reshaped:
        reshape_batch_size = q.shape[0]
        q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    # cuDNN <= 9.10.2.21 requires q, k to be contiguous
    # this comes with no cost for ViTs with RoPE because
    # RoPE has already made q and k contiguous.
    q, k = q.contiguous(), k.contiguous()

    assert cu_seqlens is not None
    assert max_seqlen is not None
    assert sequence_lengths is not None
    assert len(cu_seqlens) % 2 == 0, "cu_seqlens must be divisible by 2"
    cu_seqlength = len(cu_seqlens) // 2
    batch_offsets_qko = cu_seqlens[:cu_seqlength].view(-1, 1, 1, 1)
    batch_offsets_v = cu_seqlens[cu_seqlength:].view(-1, 1, 1, 1)
    sequence_lengths = sequence_lengths.view(-1, 1, 1, 1)
    # `cudnn_batch_prefill_with_kv_cache` needs Python ints for the
    # max-token-per-seq bounds.
    with gpu_sync_allowed():
        max_seqlen = max_seqlen.item()

    output, _ = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        scale,
        workspace_buffer,
        max_token_per_sequence=max_seqlen,
        max_sequence_kv=max_seqlen,
        actual_seq_lens_q=sequence_lengths,
        actual_seq_lens_kv=sequence_lengths,
        causal=False,
        return_lse=False,
        batch_offsets_q=batch_offsets_qko,
        batch_offsets_k=batch_offsets_qko,
        batch_offsets_v=batch_offsets_v,
        batch_offsets_o=batch_offsets_qko,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        o_data_type=o_data_type,
    )

    if is_reshaped:
        output = einops.rearrange(output, "(b s) h d -> b s h d", b=reshape_batch_size)

    return output


def vit_flashinfer_wrapper_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    o_data_type: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.empty_like(q, dtype=o_data_type or q.dtype)


direct_register_custom_op(
    op_name="flashinfer_wrapper",
    op_func=flashinfer_wrapper,
    fake_impl=vit_flashinfer_wrapper_fake,
)


def vit_flashinfer_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    o_data_type: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.flashinfer_wrapper(
        q,
        k,
        v,
        scale,
        workspace_buffer,
        cu_seqlens,
        max_seqlen,
        sequence_lengths,
        q_scale,
        k_scale,
        v_scale,
        o_data_type,
    )
