"""CUDA graph regression tests for FA2.

FA2 and FA3 expose different graph-facing contracts. FA3 has explicit
seqused_q and max_seqlen_k arguments, so its graph tests can keep oversized
buffers while changing the live query length and an independent graph KV
bound. FA2 has no seqused_q: the last cu_seqlens entry must still describe the
physical packed tensor. Its varlen graph coverage therefore changes legal
sequence boundaries while keeping the packed total fixed.

FA2 also has a training-oriented forward/backward path and derives the KV graph
bound from the cache or block-table capacity. Serving replays are expected to
change device-side cache_seqlens, cache_batch_idx, and block_table values
without changing tensor addresses. These tests cover those FA2-specific
requirements and poison unused cache rows so an out-of-range read fails
deterministically.
"""

import os

import pytest
import torch

from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from test_flash_attn import attention_ref


DISABLE_PAGEDKV = os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"


def capture_graph(fn):
    """Warm up on a side stream, then capture fn into a CUDA graph."""
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    return graph, result


def assert_matches_reference(out, q, k, v, causal):
    """Compare graph output with the existing independent PyTorch reference."""
    out_ref, _ = attention_ref(q, k, v, causal=causal)
    out_pt, _ = attention_ref(q, k, v, causal=causal, upcast=False, reorder_ops=True)
    max_error = (out - out_ref).abs().max().item()
    reference_error = (out_pt - out_ref).abs().max().item()
    rounding_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    tolerance = 2 * reference_error + rounding_atol
    assert torch.isfinite(out).all(), "CUDA graph output contains NaN or Inf"
    assert max_error <= tolerance, (
        f"output error {max_error} exceeds reference-derived tolerance {tolerance}"
    )


@pytest.mark.parametrize("causal", [False, True])
def test_training_fwd_bwd_graph_replay(causal):
    """FA2 training graphs must replay both forward and backward kernels."""
    device = "cuda"
    dtype = torch.float16
    batch_size, seqlen_q, seqlen_k = 2, 64, 96
    nheads, nheads_kv, headdim = 4, 2, 64

    states = []
    for seed in (11, 29):
        torch.random.manual_seed(seed)
        states.append(
            (
                torch.randn(
                    batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype
                ),
                torch.randn(
                    batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype
                ),
                torch.randn(
                    batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype
                ),
                torch.randn(
                    batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype
                ),
            )
        )

    q = torch.empty_like(states[0][0], requires_grad=True)
    k = torch.empty_like(states[0][1], requires_grad=True)
    v = torch.empty_like(states[0][2], requires_grad=True)
    dout = torch.empty_like(states[0][3])

    def load_state(state):
        with torch.no_grad():
            q.copy_(state[0])
            k.copy_(state[1])
            v.copy_(state[2])
            dout.copy_(state[3])

    def run_fa():
        out = flash_attn_func(q, k, v, causal=causal, deterministic=True)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, dq, dk, dv

    load_state(states[0])
    graph, graph_result = capture_graph(run_fa)

    for state_idx in (0, 1, 0):
        state = states[state_idx]
        load_state(state)

        q_eager = state[0].detach().clone().requires_grad_()
        k_eager = state[1].detach().clone().requires_grad_()
        v_eager = state[2].detach().clone().requires_grad_()
        out_eager = flash_attn_func(
            q_eager, k_eager, v_eager, causal=causal, deterministic=True
        )
        eager_result = (
            out_eager,
            *torch.autograd.grad(
                out_eager, (q_eager, k_eager, v_eager), state[3]
            ),
        )

        graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip(graph_result, eager_result):
            assert torch.isfinite(actual).all(), (
                "CUDA graph forward/backward result contains NaN or Inf"
            )
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        assert_matches_reference(
            graph_result[0],
            state[0],
            state[1],
            state[2],
            causal,
        )


@pytest.mark.parametrize("causal", [False, True])
@torch.inference_mode()
def test_varlen_graph_replay_changes_packed_boundaries(causal):
    """Replay legal FA2 cu_seqlens changes without changing packed totals."""
    device = "cuda"
    dtype = torch.float16
    total_q, total_k = 128, 160
    nheads, nheads_kv, headdim = 4, 2, 64

    torch.random.manual_seed(101 + int(causal))
    q = torch.randn(total_q, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(total_k, nheads_kv, headdim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    cu_seqlens_q = torch.tensor([0, 64, total_q], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 80, total_k], device=device, dtype=torch.int32)

    q_boundaries = ((0, 64, total_q), (0, 17, total_q))
    k_boundaries = ((0, 80, total_k), (0, 33, total_k))

    def run_fa():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=total_q,
            max_seqlen_k=total_k,
            causal=causal,
        )

    graph, out_graph = capture_graph(run_fa)

    for state_idx in (0, 1, 0):
        q_offsets = q_boundaries[state_idx]
        k_offsets = k_boundaries[state_idx]
        cu_seqlens_q.copy_(torch.tensor(q_offsets, device=device, dtype=torch.int32))
        cu_seqlens_k.copy_(torch.tensor(k_offsets, device=device, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()

        for batch_idx in range(2):
            q_start, q_end = q_offsets[batch_idx : batch_idx + 2]
            k_start, k_end = k_offsets[batch_idx : batch_idx + 2]
            assert_matches_reference(
                out_graph[q_start:q_end].unsqueeze(0),
                q[q_start:q_end].unsqueeze(0),
                k[k_start:k_end].unsqueeze(0),
                v[k_start:k_end].unsqueeze(0),
                causal,
            )


def poison_unpaged_tail(v_cache, cache_batch_idx, cache_seqlens):
    """Fill logically unused cache rows selected by each active batch with NaNs."""
    for slot, live_seqlen in zip(cache_batch_idx, cache_seqlens):
        v_cache[slot, live_seqlen:].fill_(float("nan"))


@torch.inference_mode()
def test_kvcache_graph_replay_changes_lengths_and_cache_slots():
    """FA2 decode graphs must consume replay-time lengths and cache slot indices."""
    device = "cuda"
    dtype = torch.float16
    batch_size, cache_slots, cache_capacity = 2, 4, 256
    nheads, nheads_kv, headdim = 4, 2, 64

    torch.random.manual_seed(211)
    q = torch.randn(batch_size, 1, nheads, headdim, device=device, dtype=dtype)
    k_cache = torch.randn(
        cache_slots, cache_capacity, nheads_kv, headdim, device=device, dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    v_cache_saved = v_cache.clone()
    cache_seqlens = torch.tensor(
        [cache_capacity, 193], device=device, dtype=torch.int32
    )
    cache_batch_idx = torch.tensor([0, 2], device=device, dtype=torch.int32)

    states = (
        ((cache_capacity, 193), (0, 2)),
        ((37, 129), (3, 1)),
    )

    def run_fa():
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            causal=True,
            num_splits=1,
        )

    graph, out_graph = capture_graph(run_fa)

    for state_idx in (0, 1, 0):
        live_lengths, cache_slots_for_batch = states[state_idx]
        v_cache.copy_(v_cache_saved)
        poison_unpaged_tail(v_cache, cache_slots_for_batch, live_lengths)
        cache_seqlens.copy_(
            torch.tensor(live_lengths, device=device, dtype=torch.int32)
        )
        cache_batch_idx.copy_(
            torch.tensor(cache_slots_for_batch, device=device, dtype=torch.int32)
        )
        graph.replay()
        torch.cuda.synchronize()

        for batch_idx, (slot, live_seqlen) in enumerate(
            zip(cache_slots_for_batch, live_lengths)
        ):
            assert_matches_reference(
                out_graph[batch_idx : batch_idx + 1],
                q[batch_idx : batch_idx + 1],
                k_cache[slot : slot + 1, :live_seqlen],
                v_cache[slot : slot + 1, :live_seqlen],
                causal=True,
            )


def gather_paged_sequence(cache, page_ids, live_seqlen):
    """Materialize one logical paged-cache prefix for the reference."""
    return cache[page_ids.long()].flatten(0, 1)[:live_seqlen].unsqueeze(0)


def poison_paged_tail(v_cache, block_table, live_lengths):
    """Fill graph-visible but logically unused paged-cache rows with NaNs."""
    page_size = v_cache.shape[1]
    for batch_idx, live_seqlen in enumerate(live_lengths):
        first_page = live_seqlen // page_size
        first_offset = live_seqlen % page_size
        for logical_page in range(first_page, block_table.shape[1]):
            physical_page = int(block_table[batch_idx, logical_page])
            begin = first_offset if logical_page == first_page else 0
            v_cache[physical_page, begin:].fill_(float("nan"))


@pytest.mark.skipif(DISABLE_PAGEDKV, reason="paged KV disabled in this build")
@torch.inference_mode()
def test_paged_kvcache_graph_replay_changes_lengths_and_block_table():
    """FA2 paged decode uses table capacity as its static CUDA graph bound."""
    device = "cuda"
    dtype = torch.float16
    batch_size, page_size, pages_per_seq, num_pages = 2, 256, 2, 6
    cache_capacity = page_size * pages_per_seq
    nheads, nheads_kv, headdim = 4, 2, 64

    torch.random.manual_seed(307)
    q = torch.randn(batch_size, 1, nheads, headdim, device=device, dtype=dtype)
    k_cache = torch.randn(
        num_pages, page_size, nheads_kv, headdim, device=device, dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    v_cache_saved = v_cache.clone()

    table_states = (
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        torch.tensor([[4, 5], [0, 1]], dtype=torch.int32),
    )
    length_states = (
        (cache_capacity, 300),
        (37, 400),
    )
    block_table = table_states[0].to(device)
    cache_seqlens = torch.tensor(length_states[0], device=device, dtype=torch.int32)

    def run_fa():
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True,
            num_splits=1,
        )

    graph, out_graph = capture_graph(run_fa)

    for state_idx in (0, 1, 0):
        table_cpu = table_states[state_idx]
        live_lengths = length_states[state_idx]
        v_cache.copy_(v_cache_saved)
        poison_paged_tail(v_cache, table_cpu, live_lengths)
        block_table.copy_(table_cpu)
        cache_seqlens.copy_(
            torch.tensor(live_lengths, device=device, dtype=torch.int32)
        )
        graph.replay()
        torch.cuda.synchronize()

        for batch_idx, live_seqlen in enumerate(live_lengths):
            page_ids = block_table[batch_idx]
            assert_matches_reference(
                out_graph[batch_idx : batch_idx + 1],
                q[batch_idx : batch_idx + 1],
                gather_paged_sequence(k_cache, page_ids, live_seqlen),
                gather_paged_sequence(v_cache, page_ids, live_seqlen),
                causal=True,
            )
