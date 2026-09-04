"""CUDA graph regression tests for FA3.

A framework such as vLLM captures a CUDA graph once and replays it for later
steps.  Replaying freezes the kernel instance and its launch configuration while
the runtime metadata (effective sequence lengths, page tables) keeps changing, so
FA3 must derive every memory bound from that runtime metadata alone.  When it
does not, surplus work tiles turn into out-of-range indices and the kernel reads
or writes outside the live region, which shows up as silently corrupted output
rather than a hard failure.
"""

import os

import pytest
import torch

from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from test_util import attention_ref

DISABLE_PAGEDKV = os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"


def capture_graph(fn):
    """Warm up on a side stream, then capture ``fn`` into a CUDA graph."""
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def assert_matches_reference(out, q, k, v, causal):
    """Check graph output against an independent PyTorch attention reference."""
    out_ref, _ = attention_ref(q, k, v, causal=causal)
    out_pt, _ = attention_ref(q, k, v, causal=causal, upcast=False, reorder_ops=True)
    max_error = (out - out_ref).abs().max().item()
    reference_error = (out_pt - out_ref).abs().max().item()
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    tolerance = 2 * reference_error + fwd_atol
    print(f"Output max diff: {max_error}")
    print(f"Reference implementation max diff: {reference_error}")
    assert torch.isfinite(out).all(), "CUDA graph output contains NaN or Inf"
    assert max_error <= tolerance, f"output error {max_error} exceeds tolerance {tolerance}"


def gather_paged_cache(cache, page_table, seqlen):
    """Materialize the logical, unpaged cache prefix used by the reference."""
    page_size = cache.shape[1]
    num_live_pages = (seqlen + page_size - 1) // page_size
    physical_pages = page_table[:, :num_live_pages].long()
    gathered = cache[physical_pages].flatten(1, 2)
    return gathered[:, :seqlen]


def poison_paged_tail(cache, page_table_cpu, live_seqlen, graph_max_seqlen):
    """Fill the graph-visible but logically unused V region with NaNs."""
    page_size = cache.shape[1]
    first_page = live_seqlen // page_size
    last_page = (graph_max_seqlen + page_size - 1) // page_size
    for batch_idx in range(page_table_cpu.shape[0]):
        for logical_page in range(first_page, last_page):
            begin = live_seqlen % page_size if logical_page == first_page else 0
            end = graph_max_seqlen % page_size if logical_page == last_page - 1 else page_size
            if end == 0:
                end = page_size
            physical_page = int(page_table_cpu[batch_idx, logical_page])
            cache[physical_page, begin:end].fill_(float("nan"))


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("replay_seqlen_q", [1, 65])
@pytest.mark.parametrize("capacity_seqlen_q", [512])
@torch.inference_mode()
def test_varlen_graph_replay_shorter_seqused_q(
        capacity_seqlen_q, replay_seqlen_q, d, causal, dtype
):
    """Capture with a long Q, replay with a short one over the same buffers."""
    device = "cuda"
    torch.random.manual_seed(capacity_seqlen_q + replay_seqlen_q + int(causal))
    seqlen_k = 512
    nheads, nheads_kv = 6, 2

    q = torch.randn(capacity_seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    cu_seqlens_q = torch.tensor([0, capacity_seqlen_q], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, seqlen_k], device=device, dtype=torch.int32)
    # The only tensor mutated between capture and replay, mirroring the static
    # buffers a serving framework reuses across steps.
    seqused_q = torch.tensor([capacity_seqlen_q], device=device, dtype=torch.int32)

    def run_fa():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            capacity_seqlen_q,
            seqlen_k,
            seqused_q=seqused_q,
            causal=causal,
            num_splits=1,
        )

    graph, out_graph = capture_graph(run_fa)

    # A -> B -> A catches both the shorter replay and stale scheduler state.
    for active_seqlen_q in (capacity_seqlen_q, replay_seqlen_q, capacity_seqlen_q):
        seqused_q.fill_(active_seqlen_q)
        graph.replay()
        torch.cuda.synchronize()  # surface asynchronous invalid accesses

        assert_matches_reference(
            out_graph[:active_seqlen_q].unsqueeze(0),
            q[:active_seqlen_q].unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            causal,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("replay_seqlen_k", [37, 300])
@pytest.mark.parametrize("graph_max_seqlen_k", [512])
@pytest.mark.parametrize("table_capacity_seqlen_k", [2048])
@torch.inference_mode()
def test_paged_kvcache_graph_replay_shorter_cache_seqlen(
        table_capacity_seqlen_k, graph_max_seqlen_k,
        replay_seqlen_k, page_size, d, causal, dtype
):
    """Table capacity, graph bound, and live KV length must stay distinct.

    The page table is intentionally wider than ``max_seqlen_k``. During replay,
    V rows after ``cache_seqlens`` are poisoned so an unmasked trailing load
    becomes a deterministic NaN instead of a silent finite-value overread.
    """
    if DISABLE_PAGEDKV:
        pytest.skip("paged KV disabled in this build")
    device = "cuda"
    assert replay_seqlen_k < graph_max_seqlen_k < table_capacity_seqlen_k
    torch.random.manual_seed(
        table_capacity_seqlen_k + graph_max_seqlen_k
        + replay_seqlen_k + page_size + int(causal)
    )
    batch_size, seqlen_q = 2, 1
    nheads, nheads_kv = 6, 2

    pages_per_seq = table_capacity_seqlen_k // page_size
    num_pages = pages_per_seq * batch_size
    k_cache = torch.randn(num_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    v_cache = torch.randn(num_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    v_cache_saved = v_cache.clone()
    page_table_cpu = torch.randperm(num_pages, dtype=torch.int64, device="cpu").reshape(
        batch_size, pages_per_seq
    )
    page_table = page_table_cpu.to(device=device, dtype=torch.int32)
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    cache_seqlens = torch.full(
        (batch_size,), graph_max_seqlen_k, device=device, dtype=torch.int32
    )

    def run_fa():
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            max_seqlen_k=graph_max_seqlen_k,
            causal=causal,
            num_splits=1,
        )

    graph, out_graph = capture_graph(run_fa)

    for live_seqlen_k in (graph_max_seqlen_k, replay_seqlen_k, graph_max_seqlen_k):
        v_cache.copy_(v_cache_saved)
        if live_seqlen_k < graph_max_seqlen_k:
            poison_paged_tail(
                v_cache, page_table_cpu, live_seqlen_k, graph_max_seqlen_k
            )
        cache_seqlens.fill_(live_seqlen_k)
        graph.replay()
        torch.cuda.synchronize()

        k_ref = gather_paged_cache(k_cache, page_table, live_seqlen_k)
        v_ref = gather_paged_cache(v_cache, page_table, live_seqlen_k)
        assert_matches_reference(out_graph, q, k_ref, v_ref, causal)
