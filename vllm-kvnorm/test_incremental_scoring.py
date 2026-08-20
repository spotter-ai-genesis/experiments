# SPDX-License-Identifier: Apache-2.0
"""Scoring a sequence across steps must equal scoring it in one shot.

This is the property the whole design rests on: the connector never sees a
finished request's KV, only the slice each step writes, so the accumulation of
those slices has to reproduce what a single pass over the full sequence would
give.

It is also where the bugs live. An earlier version passed the per-layer output
buffer straight to the kernel, which *stores* rather than accumulates, so only
the last layer survived -- scores were wrong by ~100% and every end-to-end check
still passed, because they only ever compared the connector against itself.

Exercises ``_WorkerSide.score_step`` directly with synthetic metadata, so it
needs a GPU but no engine.

Run: pytest test_incremental_scoring.py -v
"""

from __future__ import annotations

import pytest
import torch

from vllm_kvnorm.connector import StepRequest, StreamingMetadata, _WorkerSide
from vllm_kvnorm.kernels import token_scores
from vllm_kvnorm.layout import resolve_layer_layout

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BLOCK_SIZE = 16


def _make_worker(num_layers: int, heads: int, head_size: int, num_blocks: int = 32):
    """A worker with real KV tensors, bypassing Flowcept and vLLM entirely."""
    torch.manual_seed(num_layers * 1000 + heads * 10 + head_size)
    caches = [
        torch.randn(num_blocks, heads, BLOCK_SIZE, 2 * head_size, device="cuda")
        for _ in range(num_layers)
    ]
    layouts = [
        resolve_layer_layout(
            f"l{i}", c, block_size=BLOCK_SIZE, num_kv_heads=heads, head_size=head_size
        )
        for i, c in enumerate(caches)
    ]

    w = _WorkerSide.__new__(_WorkerSide)
    w._layers = layouts
    w._block_size = BLOCK_SIZE
    w._state = {}
    w._emitted = set()
    w._tp_size = 1
    w._tp_rank = 0
    w._stream = torch.cuda.Stream()
    return w, layouts


def _one_shot(layouts, block_ids: list[int], num_tokens: int) -> torch.Tensor:
    """What a single pass over the whole sequence would produce."""
    table = torch.tensor(block_ids, dtype=torch.int32, device="cuda")
    total = torch.zeros(num_tokens, dtype=torch.float32, device="cuda")
    for layout in layouts:
        total += token_scores(layout.k_view, layout.v_view, table, num_tokens)
    return total / len(layouts)


def _feed(w, req_id: str, chunks: list[tuple[int, int, list[int]]]) -> torch.Tensor:
    """Drive score_step once per chunk of (already_computed, new, new_blocks)."""
    for computed, new, new_blocks in chunks:
        w.score_step(
            StreamingMetadata(
                scheduled=[
                    StepRequest(
                        request_id=req_id,
                        new_block_ids=(new_blocks,) if new_blocks else (),
                        num_computed_tokens=computed,
                        num_scheduled_tokens=new,
                        prompt_token_ids=[1, 2, 3] if computed == 0 else None,
                    )
                ]
            )
        )
    torch.cuda.synchronize()
    st = w._state[req_id]
    return st.scored[: st.total]


@requires_cuda
@pytest.mark.parametrize("num_layers", [1, 4, 12])
def test_incremental_equals_one_shot(num_layers):
    """The core property, across layer counts.

    A single layer would hide a per-layer accumulation bug, so several are
    required for this to have teeth.
    """
    w, layouts = _make_worker(num_layers, heads=4, head_size=32)
    blocks = [7, 2, 9, 4]

    # prefill 20 tokens, then decode a token at a time
    chunks = [(0, 20, blocks)]
    for i in range(20, 50):
        chunks.append((i, 1, []))

    got = _feed(w, "r", chunks)
    expected = _one_shot(layouts, blocks, 50)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_chunked_prefill_boundaries():
    """Chunks that do not align to block_size must still stitch together."""
    w, layouts = _make_worker(6, heads=2, head_size=64)
    blocks = [3, 8, 1, 5]

    got = _feed(w, "r", [(0, 7, blocks), (7, 19, []), (26, 5, []), (31, 17, [])])
    expected = _one_shot(layouts, blocks, 48)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_blocks_arriving_incrementally():
    """Block tables grow over time; scores must not depend on when they arrive."""
    w, layouts = _make_worker(4, heads=4, head_size=32)

    # one block handed over per step, as vLLM actually does
    got = _feed(w, "r", [(0, 16, [7]), (16, 16, [2]), (32, 16, [9]), (48, 10, [4])])
    expected = _one_shot(layouts, [7, 2, 9, 4], 58)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_preemption_rescores_from_scratch():
    """A preempted request restarts at 0; the rebuilt scores must still match."""
    w, layouts = _make_worker(4, heads=2, head_size=32)
    blocks = [6, 11]

    # score 20 tokens, then the request is preempted and recomputed from 0
    # into the same blocks, finally reaching 30 tokens
    got = _feed(w, "r", [(0, 20, blocks), (0, 25, []), (25, 5, [])])
    expected = _one_shot(layouts, blocks, 30)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_multiple_requests_do_not_interfere():
    w, layouts = _make_worker(4, heads=2, head_size=32)
    a_blocks, b_blocks = [1, 2], [20, 21]

    for computed, new in ((0, 10), (10, 10), (20, 8)):
        w.score_step(
            StreamingMetadata(
                scheduled=[
                    StepRequest("a", (a_blocks,) if computed == 0 else (),
                                computed, new, [1] if computed == 0 else None),
                    StepRequest("b", (b_blocks,) if computed == 0 else (),
                                computed, new, [2] if computed == 0 else None),
                ]
            )
        )
    torch.cuda.synchronize()

    for req_id, blocks in (("a", a_blocks), ("b", b_blocks)):
        st = w._state[req_id]
        torch.testing.assert_close(
            st.scored[: st.total], _one_shot(layouts, blocks, 28), rtol=1e-5, atol=1e-6
        )


@requires_cuda
def test_scores_are_positive_and_finite():
    w, _ = _make_worker(4, heads=2, head_size=32)
    got = _feed(w, "r", [(0, 16, [1]), (16, 16, [2])])
    assert torch.isfinite(got).all()
    assert (got > 0).all()
