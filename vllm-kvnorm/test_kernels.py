# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the KV-norm kernels, layout resolver and scoring.

The layout and scoring tests are CPU-only; the kernel tests need CUDA + Triton
and are skipped otherwise.

Run: pytest test_kernels.py -v
"""

from __future__ import annotations


import pytest
import torch

from vllm_kvnorm.kernels import HAS_TRITON, token_scores, token_scores_torch
from vllm_kvnorm.layout import UnsupportedLayout, resolve_layer_layout

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")
requires_triton = pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="CUDA+Triton required")


# --------------------------------------------------------------------------- #
# Layout resolution
# --------------------------------------------------------------------------- #


def test_fused_head_major_matches_flash_attention_split():
    """The FlashAttention layout must split exactly as flash_attn.py does."""
    num_blocks, block_size, heads, head_size = 4, 16, 2, 8
    t = torch.randn(num_blocks, heads, block_size, 2 * head_size)

    layout = resolve_layer_layout(
        "l0", t, block_size=block_size, num_kv_heads=heads, head_size=head_size
    )

    assert layout.layout_kind == "fused_head_major"
    assert layout.k_view.shape == (num_blocks, block_size, heads, head_size)

    # Reference: exactly the expression used by vLLM's FlashAttention backend.
    k_ref, v_ref = t.transpose(1, 2).split(head_size, dim=-1)
    torch.testing.assert_close(layout.k_view, k_ref)
    torch.testing.assert_close(layout.v_view, v_ref)


def test_split_token_major_layout():
    num_blocks, block_size, heads, head_size = 3, 16, 4, 8
    t = torch.randn(2, num_blocks, block_size, heads, head_size)

    layout = resolve_layer_layout(
        "l0", t, block_size=block_size, num_kv_heads=heads, head_size=head_size
    )

    assert layout.layout_kind == "split_token_major"
    torch.testing.assert_close(layout.k_view, t[0])
    torch.testing.assert_close(layout.v_view, t[1])


def test_split_head_major_layout():
    num_blocks, block_size, heads, head_size = 3, 16, 4, 8
    t = torch.randn(2, num_blocks, heads, block_size, head_size)

    layout = resolve_layer_layout(
        "l0", t, block_size=block_size, num_kv_heads=heads, head_size=head_size
    )

    assert layout.layout_kind == "split_head_major"
    torch.testing.assert_close(layout.k_view, t[0].transpose(1, 2))


def test_layout_views_are_copy_free():
    t = torch.randn(4, 2, 16, 16)
    layout = resolve_layer_layout("l0", t, block_size=16, num_kv_heads=2, head_size=8)
    assert layout.k_view._base is not None
    assert layout.k_view.data_ptr() == t.data_ptr()


def test_asymmetric_head_size_v():
    """head_size != head_size_v must split at the right offset."""
    num_blocks, block_size, heads, d, dv = 2, 16, 2, 8, 4
    t = torch.randn(num_blocks, heads, block_size, d + dv)
    layout = resolve_layer_layout(
        "l0", t, block_size=block_size, num_kv_heads=heads, head_size=d, head_size_v=dv
    )
    assert layout.k_view.shape[-1] == d
    assert layout.v_view.shape[-1] == dv


def test_non_tensor_cache_rejected():
    with pytest.raises(UnsupportedLayout, match="not a Tensor"):
        resolve_layer_layout("mamba", [torch.randn(2)], block_size=16, num_kv_heads=1, head_size=8)


def test_quantised_dtype_rejected():
    t = torch.zeros(2, 2, 16, 16, dtype=torch.int8)
    with pytest.raises(UnsupportedLayout, match="dtype"):
        resolve_layer_layout("l0", t, block_size=16, num_kv_heads=2, head_size=8)


def test_unknown_shape_rejected():
    t = torch.randn(5, 7, 9)
    with pytest.raises(UnsupportedLayout, match="cannot map"):
        resolve_layer_layout("l0", t, block_size=16, num_kv_heads=2, head_size=8)


# --------------------------------------------------------------------------- #
# Kernel correctness
# --------------------------------------------------------------------------- #


def _reference_scores(k_view, v_view, block_table, num_tokens):
    """Explicit per-token loop, written differently from token_scores_torch."""
    block_size = k_view.shape[1]
    out = []
    for t in range(num_tokens):
        blk, off = int(block_table[t // block_size]), t % block_size
        k = torch.linalg.vector_norm(k_view[blk, off].float(), dim=-1)
        v = torch.linalg.vector_norm(v_view[blk, off].float(), dim=-1)
        out.append((v / k.clamp_min(1e-6)).mean())
    return torch.stack(out)


@requires_cuda
def test_torch_path_matches_explicit_reference():
    torch.manual_seed(0)
    num_blocks, block_size, heads, head_size = 8, 16, 3, 32
    t = torch.randn(num_blocks, heads, block_size, 2 * head_size, device="cuda")
    layout = resolve_layer_layout(
        "l0", t, block_size=block_size, num_kv_heads=heads, head_size=head_size
    )
    # Shuffled, non-contiguous table: the paged indirection must be honoured.
    block_table = torch.tensor([5, 1, 7, 0], dtype=torch.int32, device="cuda")
    num_tokens = 50  # partial final block

    got = token_scores_torch(layout.k_view, layout.v_view, block_table, num_tokens)
    ref = _reference_scores(layout.k_view, layout.v_view, block_table, num_tokens)
    torch.testing.assert_close(got, ref)


@requires_triton
@pytest.mark.parametrize("head_size", [8, 32, 64, 96, 128])
@pytest.mark.parametrize("heads", [1, 2, 8])
def test_triton_matches_torch(head_size, heads):
    torch.manual_seed(head_size * 100 + heads)
    num_blocks, block_size = 16, 16
    t = torch.randn(
        num_blocks, heads, block_size, 2 * head_size, device="cuda", dtype=torch.bfloat16
    )
    layout = resolve_layer_layout(
        "l0", t, block_size=block_size, num_kv_heads=heads, head_size=head_size
    )
    block_table = torch.tensor([11, 3, 0, 8, 15], dtype=torch.int32, device="cuda")
    num_tokens = 5 * block_size - 7

    got = token_scores(layout.k_view, layout.v_view, block_table, num_tokens)
    ref = token_scores_torch(layout.k_view, layout.v_view, block_table, num_tokens)
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)


@requires_triton
def test_one_score_per_token():
    t = torch.randn(8, 2, 16, 32, device="cuda")
    layout = resolve_layer_layout("l0", t, block_size=16, num_kv_heads=2, head_size=16)
    table = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    out = token_scores(layout.k_view, layout.v_view, table, 30)
    assert out.shape == (30,), out.shape
    assert out.dtype == torch.float32


@requires_triton
def test_score_is_mean_over_heads_of_v_over_k():
    """The emitted scalar must equal mean_h(||V_h|| / ||K_h||)."""
    torch.manual_seed(11)
    heads, head_size = 4, 32
    t = torch.randn(4, heads, 16, 2 * head_size, device="cuda")
    layout = resolve_layer_layout("l0", t, block_size=16, num_kv_heads=heads, head_size=head_size)
    table = torch.tensor([2], dtype=torch.int32, device="cuda")

    got = token_scores(layout.k_view, layout.v_view, table, 16)
    k = torch.linalg.vector_norm(layout.k_view[2].float(), dim=-1)  # (16, heads)
    v = torch.linalg.vector_norm(layout.v_view[2].float(), dim=-1)
    torch.testing.assert_close(got, (v / k).mean(dim=-1), rtol=1e-4, atol=1e-4)


@requires_triton
def test_triton_handles_all_layouts():
    """Same numbers regardless of which physical layout the backend chose."""
    torch.manual_seed(7)
    num_blocks, block_size, heads, head_size = 8, 16, 4, 32
    table = torch.tensor([2, 6, 1], dtype=torch.int32, device="cuda")

    base_k = torch.randn(num_blocks, block_size, heads, head_size, device="cuda")
    base_v = torch.randn(num_blocks, block_size, heads, head_size, device="cuda")

    fused = torch.cat([base_k, base_v], dim=-1).permute(0, 2, 1, 3).contiguous()
    split = torch.stack([base_k, base_v]).contiguous()
    lf = resolve_layer_layout("f", fused, block_size=block_size, num_kv_heads=heads, head_size=head_size)
    ls = resolve_layer_layout("s", split, block_size=block_size, num_kv_heads=heads, head_size=head_size)

    torch.testing.assert_close(
        token_scores(lf.k_view, lf.v_view, table, 40),
        token_scores(ls.k_view, ls.v_view, table, 40),
    )


@requires_triton
def test_reads_only_listed_blocks():
    """Blocks absent from the block table must not influence the result."""
    torch.manual_seed(3)
    t = torch.randn(8, 2, 16, 32, device="cuda")
    layout = resolve_layer_layout("l0", t, block_size=16, num_kv_heads=2, head_size=16)
    table = torch.tensor([1, 4], dtype=torch.int32, device="cuda")

    before = token_scores(layout.k_view, layout.v_view, table, 32)
    t[7] += 1000.0  # untouched block
    after = token_scores(layout.k_view, layout.v_view, table, 32)
    torch.testing.assert_close(before, after)


@requires_triton
def test_zero_key_norm_does_not_produce_nan():
    t = torch.zeros(4, 2, 16, 32, device="cuda")
    layout = resolve_layer_layout("l0", t, block_size=16, num_kv_heads=2, head_size=16)
    table = torch.tensor([0], dtype=torch.int32, device="cuda")
    assert torch.isfinite(token_scores(layout.k_view, layout.v_view, table, 16)).all()


@requires_triton
def test_asymmetric_head_size_v_kernel():
    """head_size != head_size_v must still reduce correctly."""
    d, dv, heads = 32, 16, 2
    t = torch.randn(4, heads, 16, d + dv, device="cuda")
    layout = resolve_layer_layout(
        "l0", t, block_size=16, num_kv_heads=heads, head_size=d, head_size_v=dv
    )
    table = torch.tensor([1], dtype=torch.int32, device="cuda")
    got = token_scores(layout.k_view, layout.v_view, table, 16)
    ref = token_scores_torch(layout.k_view, layout.v_view, table, 16)
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# Tensor-parallel reduction maths
# --------------------------------------------------------------------------- #
#
# Under TP each rank holds only a shard of the KV heads, so its score is a mean
# over that shard. The connector all-reduces the shard means and divides by
# tp_size. These tests verify that recovers the true all-head mean, without
# needing multiple GPUs: sharding is simulated by slicing the head axis.


def _sharded_score(k_view, v_view, block_table, num_tokens, tp_size):
    """Mimic the connector: per-rank score on a head shard, then mean of means."""
    heads = k_view.shape[2]
    per_rank = max(1, heads // tp_size)
    partials = []
    for r in range(tp_size):
        lo = (r * per_rank) % heads
        sl = slice(lo, lo + per_rank)
        partials.append(
            token_scores_torch(k_view[:, :, sl], v_view[:, :, sl], block_table, num_tokens)
        )
    return sum(partials) / tp_size


@requires_cuda
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_tp_shard_means_recover_all_head_mean(tp_size):
    """all_reduce(shard means) / tp_size == the unsharded all-head mean."""
    torch.manual_seed(tp_size)
    heads, head_size = 8, 32
    t = torch.randn(6, heads, 16, 2 * head_size, device="cuda")
    layout = resolve_layer_layout("l0", t, block_size=16, num_kv_heads=heads, head_size=head_size)
    table = torch.tensor([1, 3], dtype=torch.int32, device="cuda")

    whole = token_scores_torch(layout.k_view, layout.v_view, table, 32)
    sharded = _sharded_score(layout.k_view, layout.v_view, table, 32, tp_size)
    torch.testing.assert_close(sharded, whole, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_tp_replicated_heads_still_uniform():
    """When tp_size > num_kv_heads vLLM replicates heads; the mean stays uniform."""
    torch.manual_seed(99)
    heads, head_size, tp_size = 2, 32, 4  # each head replicated twice
    t = torch.randn(4, heads, 16, 2 * head_size, device="cuda")
    layout = resolve_layer_layout("l0", t, block_size=16, num_kv_heads=heads, head_size=head_size)
    table = torch.tensor([2], dtype=torch.int32, device="cuda")

    whole = token_scores_torch(layout.k_view, layout.v_view, table, 16)
    sharded = _sharded_score(layout.k_view, layout.v_view, table, 16, tp_size)
    torch.testing.assert_close(sharded, whole, rtol=1e-5, atol=1e-6)
