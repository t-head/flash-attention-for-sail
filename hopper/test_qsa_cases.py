"""QSA forward correctness and optional timing, with automatic SplitKV.

Run pytest hopper/test_qsa_cases.py, or use --list / --case / --benchmark directly.
FA3_QSA source configurations are recorded below; no archived test modules or
subprocess adapters are needed. The shared sparse reference checks every output.
Input generation is consolidated, so timings are not historical A/B samples.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate
import math
from pathlib import Path
import statistics
import sys

import pytest
import torch


@dataclass(frozen=True)
class Case:
    name: str
    sq: int
    sk: int
    dim: int
    topk: int
    page: int
    batch: int
    heads: int
    kv_heads: int
    mode: str
    error: str = ""
    source: str = ""
    cache_length: str = "full"
    allow_aiu: bool = False
    mismatch: bool = False


# FA3_QSA provenance (commit / path); repeated source configurations retain IDs.
SOURCES = {
    "decode": "df2b389e4a7ef17dcbc3a815301b45f2fa90f232 / hopper/test_flash_attn_decode_perf.py",
    "decode_hopper": "4202330db0c59bbefa790044ffda2860040e3ff3 / hopper/test_flash_attn_decode_perf.py",
    "decode_release": "4d6b23610b8c059596e87027e20c655270b19e57 / hopper/test_flash_attn_decode_perf.py",
    "functional": "3ac92fbe896bca992ce0def9e0e14256448acba6 / hopper/test_flash_attn.py",
    "functional_5090": "4d6b23610b8c059596e87027e20c655270b19e57 / hopper/test_flash_attn_5090.py",
    "functional_hopper": "4202330db0c59bbefa790044ffda2860040e3ff3 / hopper/test_flash_attn.py",
    "functional_release": "4d6b23610b8c059596e87027e20c655270b19e57 / hopper/test_flash_attn.py",
    "functional_ut": "420b6f0cf80b34149d19e9bba165a1aaca440470 / hopper/test_flash_attn.py",
    "prefill_hopper": "4202330db0c59bbefa790044ffda2860040e3ff3 / hopper/test_flash_attn_prefill_perf.py",
    "prefill_release": "4d6b23610b8c059596e87027e20c655270b19e57 / hopper/test_flash_attn_prefill_perf.py",
    "prefill_ut": "420b6f0cf80b34149d19e9bba165a1aaca440470 / hopper/test_flash_attn_prefill_perf.py",
}

# name, sq, sk, dim, topk, page, batch, heads, kv_heads, index mode
CASES = [
    Case("original:decode",           4,   8192, 256, 2048, 544, 128, 32, 1, "structured", source="decode", cache_length="topk"),
    Case("original:decode_hopper",    4,  32768, 256, 2048, 544,  16, 16, 2, "physical", source="decode_hopper"),
    Case("original:decode_release",   4,   8192, 256, 2048, 544, 128, 16, 2, "physical", source="decode_release"),
    Case("original:functional",     960,   1024, 256,  128, 544,   1, 16, 2, "structured", source="functional"),
    Case("original:functional_5090",  1,    256, 256,  256, 128,  32, 16, 2, "structured", source="functional_5090", cache_length="random"),
    Case("original:functional_hopper",960,  1024, 256,  128, 544,   2, 16, 2, "structured", source="functional_hopper"),
    Case("original:functional_release",960, 1024, 256,  128, 544,   2, 16, 2, "structured", source="functional_release"),
    Case("original:functional_ut",   16,    128, 256,  128, 128,  32, 16, 2, "structured", source="functional_ut"),
    Case("original:prefill_hopper",16384, 131072, 256, 2048, 544,   1, 16, 1, "physical", source="prefill_hopper"),
    Case("original:prefill_release",16384,131072, 256, 2048, 544,   1, 16, 1, "physical", source="prefill_release"),
    Case("original:prefill_ut",    16384, 131072, 256, 2048, 544,   1, 16, 1, "physical", source="prefill_ut"),
    Case("fa3_precision",           960,   1024, 256,  128, 544,   1, 16, 2, "structured"),
    Case("fa3_precision_rand",      960,   1024, 256,  128, 544,   1, 16, 2, "random"),
    Case("probe_rand_noaiu",        960,   1024, 256,  128, 136,   1, 16, 2, "random"),
    Case("probe_block16",           960,   1024, 256,  128, 544,   1, 16, 2, "block16"),
    Case("hist_topk64",               2,   1024, 256,   64, 128,   1, 16, 2, "structured"),
    Case("hist_topk64_block64",       2,   1024, 256,   64, 128,   1, 16, 2, "block64"),
    Case("fa3_decode",                4,   8192, 256, 2048, 544, 128, 16, 2, "block16"),
    Case("fa3_decode_h32_mqa",        4,   8192, 256, 2048, 544, 128, 32, 1, "block16"),
    Case("fa3_decode_d128",           4,   8192, 128, 2048, 544, 128, 16, 2, "block16", error="QSA only supports head dim 256"),
    Case("fa3_prefill",           16384, 131072, 256, 2048, 544,   1, 16, 1, "block16"),
    Case("sq1_decode",               1,   1024, 256,  128, 544,   4, 16, 2, "structured"),
    Case("sq1_decode_h32_mqa",       1,   1024, 256,  128, 544,   4, 32, 1, "structured"),
    Case("sq1_rand",                 1,   8192, 256,  128, 544,   4, 16, 2, "random"),
    Case("sq1_block16",              1,   8192, 256,  128, 544,   4, 16, 2, "block16"),
    # Four-warp decode: independent rows, partial score tiles, and both KV strides.
    Case("sq1_g3_b2_fourwarp",       1,   4893, 256, 2051,  16,   2,  3, 1, "random", cache_length="topk"),
    Case("sq1_g3_b3_fourwarp",       1,   4893, 256, 2051,  16,   3,  3, 1, "random", cache_length="topk"),
    Case("sq1_g6_fourwarp",          1,   4893, 256, 2051,  16,   2,  6, 1, "random", cache_length="topk"),
    Case("sq1_g12_fourwarp",         1,   4893, 256, 2051,  16,   1, 12, 1, "random", cache_length="topk"),
    Case("sq1_g12_two_kv_fourwarp",  1,   4893, 256, 2051,  16,   2, 24, 2, "random", cache_length="topk"),
    Case("sq1_g64_reject",           1,   1024, 256,  128, 544,   2, 64, 1, "structured", error="QSA supports at most 32 query heads per KV head"),
    Case("tail_topk100",            960,   1024, 256,  100, 544,   1, 16, 2, "block16tail"),
    Case("tail_topk100_ps576",      960,   1024, 256,  100, 576,   1, 16, 2, "block64tail"),
    Case("tail_topk100_sq1_g32",      1,   8192, 256,  100, 544,   4, 32, 1, "block16tail"),
    # Preserve the target branch's explicit AIU contract checks.
    Case("run16_allow_aiu",         960,   1024, 256,  128, 544,   1, 16, 2, "block16", allow_aiu=True),
    Case("rand_allow_aiu",          960,   1024, 256,  128, 544,   1, 16, 2, "random", allow_aiu=True, mismatch=True),
]


def build_inputs(case, dtype, varlen=False):
    torch.manual_seed(0)
    device = "cuda"
    lengths = [max(1, case.sq - b) if varlen else case.sq for b in range(case.batch)]
    cu = torch.tensor([0] + list(accumulate(lengths)), device=device, dtype=torch.int32)
    q = torch.randn(sum(lengths), case.heads, case.dim, device=device, dtype=dtype)
    blocks_per_batch = math.ceil(case.sk / case.page) * 3
    blocks = blocks_per_batch * case.batch
    k = torch.randn(blocks, case.page, case.kv_heads, case.dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    # Source cases use shuffled physical pages; auxiliary cases use identity pages.
    pages = (torch.randperm(blocks, device=device) if case.source
             else torch.arange(blocks, device=device)).reshape(case.batch, blocks_per_batch)
    rows = []
    for b, length in enumerate(lengths):
        if case.mode == "structured":
            indices = torch.arange(case.topk, device=device).expand(length, -1)
        elif case.mode in ("random", "physical"):
            indices = torch.randint(case.sk, (length, case.topk), device=device)
        elif case.mode.startswith("block"):
            run = int(case.mode[5:].removesuffix("tail"))
            assert case.sk % run == 0
            assert case.topk % run == 0 or case.mode.endswith("tail")
            starts = torch.randint(case.sk // run, (length, math.ceil(case.topk / run)),
                                   device=device) * run
            indices = (starts[..., None] + torch.arange(run, device=device)).reshape(length, -1)
            indices = indices[:, :case.topk]
        else:
            raise ValueError(f"Unknown index mode: {case.mode}")
        # Legacy random sources specify absolute physical indices directly.
        if case.mode != "physical":
            indices = pages[b, indices // case.page] * case.page + indices % case.page
        rows.append(indices)
    topk = torch.cat(rows).to(torch.int32)
    cache = torch.full((case.batch,), case.topk if case.cache_length == "topk" else case.sk,
                       device=device, dtype=torch.int32)
    if case.cache_length == "random":
        cache = torch.randint(1, case.sk + 1, (case.batch,), device=device, dtype=torch.int32)
    return q, k, v, topk, cu, cache


def sparse_attention_ref(q, k, v, indices, cu, cache, *, source=False, return_lse=False):
    """Gather per query, mask sparse columns, and compare FP32 with low precision."""
    topk = indices.shape[-1]
    heads, dim = q.shape[-2:]
    kv_heads = k.shape[-2]
    group = heads // kv_heads
    k, v = k.flatten(0, 1), v.flatten(0, 1)
    ref, pt = torch.empty_like(q), torch.empty_like(q)
    lse = torch.empty(heads, q.shape[0], device=q.device, dtype=torch.float32) if return_lse else None
    # Bound the gathered FP32 K/V tiles for long prefill sequences.
    chunk = max(1, (128 << 20) // (topk * kv_heads * dim * 4))
    offsets, cache_lengths = cu.tolist(), cache.tolist()
    columns = torch.arange(topk, device=q.device)[None, :]
    scale = 1.0 / math.sqrt(dim)
    for b, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
        for start in range(lo, hi, chunk):
            end = min(start + chunk, hi)
            idx = indices[start:end].long()
            qq = q[start:end].reshape(end - start, kv_heads, group, dim)
            kk, vv = k[idx], v[idx]
            # Causal positions are sparse columns, not gathered physical tokens.
            positions = torch.arange(start - lo, end - lo, device=q.device)
            valid = (columns < cache_lengths[b]) & (
                columns <= positions[:, None] + cache_lengths[b] - (hi - lo))
            valid = valid[:, None, None, :]
            for target, upcast in ((ref, True), (pt, False)):
                q_ = qq.float() if upcast else qq
                k_ = kk.float() if upcast else kk
                v_ = vv.float() if upcast else vv
                if source:
                    # Preserve the source reference's Q/K scaling order.
                    scores = torch.einsum("chgd,cthd->chgt", q_ * scale if upcast else q_,
                                          k_ if upcast else k_ * scale)
                else:
                    scores = torch.einsum("chgd,cthd->chgt", q_, k_) * scale
                scores.masked_fill_(~valid, float("-inf"))
                if upcast and return_lse:
                    lse[:, start:end] = torch.logsumexp(scores, dim=-1).reshape(end - start, heads).T
                weights = torch.softmax(scores, dim=-1).masked_fill(~valid, 0.0)
                out = torch.einsum("chgt,cthd->chgd", weights, v_)
                target[start:end] = out.reshape(end - start, heads, dim)
    return (ref, pt, lse) if return_lse else (ref, pt)


def run_case(case, dtype=torch.bfloat16, *, metadata=True, varlen=False, benchmark=False):
    # Resolve the local FA3 interface without loading unrelated archived tests.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata

    q, k, v, topk, cu, cache = build_inputs(case, dtype, varlen)
    scheduler = None
    if metadata and not case.error:
        scheduler = get_scheduler_metadata(
            case.batch, case.sq, math.ceil(case.sk / case.page) * 3 * case.page,
            case.heads, case.kv_heads, case.dim, cache, dtype, headdim_v=case.dim,
            cu_seqlens_q=cu, max_seqlen_k_new=case.sq, page_size=case.page,
            causal=True, num_splits=0)

    def forward():
        return flash_attn_with_kvcache(
            q, k, v, cache_seqlens=cache, page_table=topk.unsqueeze(1), cu_seqlens_q=cu,
            max_seqlen_q=case.sq, causal=True, scheduler_metadata=scheduler,
            num_splits=0, pack_gqa=True, return_softmax_lse=True,
            qsa_allow_aiu=case.allow_aiu)[0]

    if case.error:
        with pytest.raises(RuntimeError, match=case.error):
            forward()
        return

    ref, pt = sparse_attention_ref(q, k, v, topk, cu, cache, source=bool(case.source))
    assert torch.isfinite(ref).all() and torch.isfinite(pt).all()
    # Reuse precomputed metadata across calls, as in the source tests.
    for _ in range(2 if metadata else 1):
        out = forward()
        assert torch.isfinite(out).all(), case.name
        diff, pt_diff = (out.float() - ref.float()).abs(), (pt.float() - ref.float()).abs()
        if case.mismatch:
            # A scattered list violates the caller-certified AIU bulk-load contract.
            assert diff.max().item() > 2 * pt_diff.max().item() + 1e-5, case.name
            continue
        if case.source == "decode":
            assert diff.max().item() <= 5e-3 and diff.mean().item() <= 1e-3, case.name
        else:
            mean_factor, mean_atol = (1.5, 0.0) if case.source else (2.0, 1e-6)
            assert diff.max().item() <= 2 * pt_diff.max().item() + 1e-5, case.name
            assert diff.mean().item() <= mean_factor * pt_diff.mean().item() + mean_atol, case.name
    if benchmark:
        for _ in range(5):
            forward()
        torch.cuda.synchronize()
        samples = []
        for _ in range(20):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            forward()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        print(f"{case.name}: {statistics.median(samples):.3f} us (automatic SplitKV)", flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_qsa(case, dtype):
    try:
        run_case(case, dtype)
    finally:
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", help="Case name (including original:<name>) or auxiliary index")
    parser.add_argument("--params", help="Ad-hoc configuration: sq,sk,d,topk,ps,b,h,hk,mode")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(case)
        return 0
    if args.params:
        fields = args.params.split(",")
        if len(fields) != 9:
            parser.error("--params requires nine comma-separated values")
        cases = [Case("adhoc", *(int(x) for x in fields[:8]), fields[8])]
    elif args.case and args.case.isdigit():
        auxiliary = [case for case in CASES if not case.source]
        if int(args.case) >= len(auxiliary):
            parser.error(f"Unknown auxiliary index: {args.case}")
        cases = [auxiliary[int(args.case)]]
    elif args.case:
        cases = [case for case in CASES if case.name == args.case]
        if not cases:
            parser.error(f"Unknown case: {args.case}")
    else:
        cases = CASES
    if not torch.cuda.is_available():
        parser.error("QSA requires a CUDA-compatible device")
    failures = []
    for case in cases:
        try:
            run_case(case, torch.bfloat16 if args.dtype == "bf16" else torch.float16,
                     metadata=not args.no_metadata, varlen=args.varlen, benchmark=args.benchmark)
            print(f"PASS {case.name}", flush=True)
        except Exception as error:
            failures.append(case.name)
            print(f"FAIL {case.name}: {type(error).__name__}: {error}", flush=True)
        finally:
            torch.cuda.empty_cache()
    print(f"{len(cases) - len(failures)}/{len(cases)} passed", flush=True)
    return bool(failures)


# ---------------------------------------------------------------------------
# fwd-direct path: torch.ops.flash_attn_3.fwd called with a 3-D QSA page_table
# ---------------------------------------------------------------------------

def _fwd_direct(q, k, v, page_table_3d, cu_seqlens_q, seqused_k, max_seqlen_q,
                causal=True, num_splits=1, pack_gqa=True, out=None):
    """Call the C++ fwd op directly, bypassing flash_attn_with_kvcache."""
    import flash_attn_3._C
    scale = q.shape[-1] ** -0.5
    return torch.ops.flash_attn_3.fwd(
        q, k, v,
        None, None, None, out,                     # k_new, v_new, qv, out
        cu_seqlens_q, None, None,                  # cu_seqlens_q/k/k_new
        None, seqused_k,                           # seqused_q, seqused_k
        max_seqlen_q, None,                        # max_seqlen_q/k
        page_table_3d,                             # page_table (3-D triggers QSA)
        None, None,                                # kv_batch_idx, leftpad_k
        None, None, None,                          # rotary
        None, None, None,                          # descale
        scale,                                     # softmax_scale
        causal, -1, -1, 0, 0.0, True,             # causal, window, chunk, softcap, rot_interl
        None, num_splits, pack_gqa, 0,             # scheduler, splits, pack_gqa, sm_margin
        None, False,                               # s_aux, qsa_allow_aiu
    )


def test_qsa_fwd_direct_causal_no_seqused_k_falls_back():
    """Causal QSA without seqused_k silently drops causal (== explicit causal=False)."""
    case = Case("_fwd_fallback", 960, 1024, 256, 128, 544, 1, 16, 2, "structured")
    q, k, v, topk, cu, cache = build_inputs(case, torch.bfloat16)
    pt3d = topk.unsqueeze(1)
    out_causal_no_sk, *_ = _fwd_direct(q, k, v, pt3d, cu, None, case.sq, causal=True)
    out_noncausal, *_ = _fwd_direct(q, k, v, pt3d, cu, None, case.sq, causal=False)
    assert torch.equal(out_causal_no_sk, out_noncausal), \
        f"max diff {(out_causal_no_sk.float() - out_noncausal.float()).abs().max().item()}"


def test_qsa_fwd_direct_causal_matches_kvcache():
    """fwd-direct with seqused_k must produce the same output as flash_attn_with_kvcache."""
    from flash_attn_interface import flash_attn_with_kvcache
    case = Case("_fwd_match", 960, 1024, 256, 128, 544, 1, 16, 2, "structured")
    q, k, v, topk, cu, cache = build_inputs(case, torch.bfloat16)
    pt3d = topk.unsqueeze(1)
    scale = case.dim ** -0.5
    out_kv = flash_attn_with_kvcache(
        q, k, v, cache_seqlens=cache, page_table=pt3d, cu_seqlens_q=cu,
        max_seqlen_q=case.sq, causal=True, softmax_scale=scale,
        num_splits=1, pack_gqa=True)
    out_fwd, *_ = _fwd_direct(q, k, v, pt3d, cu, cache, case.sq)
    assert torch.equal(out_fwd, out_kv), \
        f"max diff {(out_fwd.float() - out_kv.float()).abs().max().item()}"


@pytest.mark.parametrize("queries,selected,cache_tokens", [
    (511, 511, 8192),
    (512, 512, 8192),
    (513, 2051, 32768),
])
def test_qsa_tp1_prefill_split_boundary(queries, selected, cache_tokens):
    """Check TP1 auto and explicit splits."""
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(queries, 24, 256, device=device, dtype=torch.bfloat16)
    k = torch.randn(cache_tokens // 16, 16, 2, 256, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    # Independent queries share one cache and include a top-k tail.
    indices = torch.randint(cache_tokens, (queries, 2051), device=device, dtype=torch.int32)
    cu = torch.arange(queries + 1, device=device, dtype=torch.int32)
    cache = torch.full((queries,), selected, device=device, dtype=torch.int32)
    ref, pt = sparse_attention_ref(q, k, v, indices, cu, cache)
    pt_diff = (pt.float() - ref.float()).abs()
    for splits in (0, 1, 4):
        result = _fwd_direct(q, k, v, indices.unsqueeze(1), cu, cache, 1,
                             num_splits=splits)
        diff = (result[0].float() - ref.float()).abs()
        assert torch.isfinite(result[0]).all()
        assert diff.max().item() <= 2 * pt_diff.max().item() + 1e-5
        assert diff.mean().item() <= 2 * pt_diff.mean().item() + 1e-6
        if splits == 4:
            assert result[2] is not None and result[2].shape[0] == splits


@pytest.mark.parametrize(
    "queries,prefix_selected,tail_selected,heads,kv_heads,num_splits",
    [
        pytest.param(513, 2051, 200, 3, 1, 0, id="long-prefix-short-tail"),
        pytest.param(1025, 200, 2051, 24, 2, 2, id="short-prefix-long-tail"),
    ],
)
def test_qsa_prefill_mixed_selected(queries, prefix_selected, tail_selected,
                                   heads, kv_heads, num_splits):
    """Check all outputs with mixed selected lengths."""
    torch.manual_seed(0)
    device = "cuda"
    cache_tokens = 8192
    q = torch.randn(queries, heads, 256, device=device, dtype=torch.bfloat16)
    k = torch.randn(cache_tokens // 16, 16, kv_heads, 256, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    indices = torch.randint(cache_tokens, (queries, 2051), device=device, dtype=torch.int32)
    cu = torch.arange(queries + 1, device=device, dtype=torch.int32)
    cache = torch.full((queries,), prefix_selected, device=device, dtype=torch.int32)
    # Vary the last quarter; odd Q also exercises the head mapping.
    cache[3 * queries // 4:] = tail_selected
    ref, pt = sparse_attention_ref(q, k, v, indices, cu, cache)
    result = _fwd_direct(q, k, v, indices.unsqueeze(1), cu, cache, 1, num_splits=num_splits)
    diff = (result[0].float() - ref.float()).abs()
    pt_diff = (pt.float() - ref.float()).abs()
    assert torch.isfinite(result[0]).all()
    assert diff.max().item() <= 2 * pt_diff.max().item() + 1e-5
    assert diff.mean().item() <= 2 * pt_diff.mean().item() + 1e-6


@pytest.mark.parametrize("heads,kv_heads,k_padding,v_padding,num_splits", [
    pytest.param(6, 1, 32, 64, 0, id="gqa6-unequal-strides"),
    pytest.param(16, 2, 0, 0, 1, id="sm89-gqa8-v-window-tail"),
])
def test_qsa_prefill_padded_kv_strides(heads, kv_heads, k_padding, v_padding, num_splits):
    """Check strided M8 outputs and SM89 GQA8's partial final V tile."""
    if heads == 16 and torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("SM89 GQA8 V index window")
    torch.manual_seed(0)
    queries, cache_tokens, topk = 129, 4096, 2051
    q = torch.randn(queries, heads, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(cache_tokens // 16, 16, kv_heads, 256 + k_padding,
                    device=q.device, dtype=q.dtype)[..., :256]
    v = torch.randn(cache_tokens // 16, 16, kv_heads, 256 + v_padding,
                    device=q.device, dtype=q.dtype)[..., :256]
    assert k.stride(1) == kv_heads * (256 + k_padding)
    assert v.stride(1) == kv_heads * (256 + v_padding)
    if heads == 16:
        assert queries * kv_heads > 4 and k.stride(1) == v.stride(1) == 512
        assert topk % 16 == 3
    indices = torch.randint(cache_tokens, (queries, topk), device=q.device, dtype=torch.int32)
    if heads == 16:
        q.zero_()
        indices[:, -3:] = cache_tokens - 1
        v[-1, -1].fill_(64.0)
    cu = torch.arange(queries + 1, device=q.device, dtype=torch.int32)
    lengths = torch.tensor([0, 1, 200, 511, 512, 2048, 2051], device=q.device, dtype=torch.int32)
    cache = lengths[torch.arange(queries, device=q.device) % lengths.numel()]
    ref, pt, lse_ref = sparse_attention_ref(q, k, v, indices, cu, cache, return_lse=True)
    result = _fwd_direct(q, k, v, indices.unsqueeze(1), cu, cache, 1, num_splits=num_splits)
    diff, pt_diff = (result[0].float() - ref.float()).abs(), (pt.float() - ref.float()).abs()
    assert torch.isfinite(result[0]).all()
    assert diff.max().item() <= 2 * pt_diff.max().item() + 1e-5
    assert diff.mean().item() <= 2 * pt_diff.mean().item() + 1e-6
    torch.testing.assert_close(result[1].float(), lse_ref, rtol=1e-5, atol=1e-4)
    assert result[2] is None and result[3] is None


@pytest.fixture
def fp8_qsa_device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("FP8 QSA requires PPU0015")


def _fp8_qsa_inputs(heads, kv_heads, width, rows=3):
    torch.manual_seed(20261017)
    capacity = 128
    q = torch.randn(rows, heads, 256, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(capacity, 2, kv_heads, 256, device="cuda", dtype=torch.bfloat16)
    k, v = kv[:, 0], kv[:, 1]
    k, v = k.contiguous(), v.contiguous()
    slots = torch.randint(capacity, (rows, width), device="cuda", dtype=torch.int32)
    if rows > 1:
        slots[0] = -1
    if rows > 2:
        slots[1, 1:] = -1
    counts = (slots >= 0).sum(-1, dtype=torch.int32)
    pages = slots.clamp_min(0)[:, None, :]
    cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    q, k, v = ((x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
               for x, scale in ((q, .02), (k, .025), (v, .025)))
    descales = {name: torch.full((rows, kv_heads), scale, device="cuda")
                for name, scale in (("q_descale", .02), ("k_descale", .025), ("v_descale", .025))}
    return q, k.reshape(-1, 64, kv_heads, 256), v.reshape(-1, 64, kv_heads, 256), slots, pages, counts, cu, descales


@pytest.mark.usefixtures("fp8_qsa_device")
@pytest.mark.parametrize("heads,kv_heads", [(6, 1), (24, 2)])
@pytest.mark.parametrize("width", [1, 17, 65, 2051])
@pytest.mark.parametrize("splits", [0, 1, 4, 32, 128])
def test_fp8_single_query_qsa(heads, kv_heads, width, splits):
    from flash_attn_interface import flash_attn_with_kvcache

    q, k, v, slots, pages, counts, cu, descales = _fp8_qsa_inputs(heads, kv_heads, width)
    try:
        out, lse, *_ = flash_attn_with_kvcache(q, k, v, cache_seqlens=counts, cu_seqlens_q=cu,
            max_seqlen_q=1, page_table=pages, softmax_scale=.0625, causal=True,
            num_splits=splits, return_softmax_lse=True, **descales)
    except RuntimeError as error:
        if "QSA FP8 is not compiled in" in str(error):
            pytest.skip(str(error))
        raise
    qf, kf, vf = (x.cpu().double() for x in (q, k.flatten(0, 1), v.flatten(0, 1)))
    qf, kf, vf = qf * .02, kf * .025, vf * .025
    expected = torch.zeros(q.shape, dtype=torch.float64)
    for row in (1, 2):
        ids = slots[row].cpu().long()
        ids = ids[ids >= 0]
        for head in range(heads):
            scores = kf[ids, head // (heads // kv_heads)] @ qf[row, head] * .0625
            expected[row, head] = scores.softmax(0) @ vf[ids, head // (heads // kv_heads)]
            assert abs(float(lse[head, row]) - float(scores.logsumexp(0))) < .1
    assert out.dtype == torch.bfloat16 and out.shape == q.shape
    assert lse.shape == (heads, len(q)) and torch.isfinite(out).all()
    assert torch.count_nonzero(out[0]) == 0 and torch.isneginf(lse[:, 0]).all()
    relative = (out.cpu().double() - expected).norm() / expected.norm().clamp_min(1e-12)
    assert relative <= .1


@pytest.mark.usefixtures("fp8_qsa_device")
@pytest.mark.parametrize("rows,heads,kv_heads", [
    (1, 6, 1), (2, 6, 1), (8, 6, 1), (9, 6, 1),
    (639, 6, 1), (640, 6, 1), (780, 6, 1), (781, 6, 1),
    (896, 6, 1), (936, 6, 1), (937, 6, 1), (1, 24, 2), (4096, 24, 2),
])
def test_fp8_qsa_auto_boundaries(rows, heads, kv_heads):
    from flash_attn_interface import flash_attn_with_kvcache

    q, k, v, slots, pages, counts, cu, descales = _fp8_qsa_inputs(
        heads, kv_heads, 2051, rows=rows
    )
    sample = sorted({0, min(1, rows - 1), rows // 2, rows - 1})
    qf = q[sample].cpu().double().reshape(len(sample), kv_heads, -1, 256) * .02
    kf, vf = (x.flatten(0, 1).cpu().double() * .025 for x in (k, v))
    expected = torch.zeros((len(sample), heads, 256), dtype=torch.float64)
    for index, row in enumerate(sample):
        ids = slots[row].cpu().long()
        ids = ids[ids >= 0]
        if len(ids):
            scores = torch.einsum("hgd,khd->hgk", qf[index], kf[ids]) * .0625
            expected[index] = torch.einsum(
                "hgk,khd->hgd", scores.softmax(-1), vf[ids]
            ).reshape(heads, 256)
    for splits in (0, 1):
        try:
            out = flash_attn_with_kvcache(
                q, k, v, cache_seqlens=counts, cu_seqlens_q=cu, max_seqlen_q=1,
                page_table=pages, softmax_scale=.0625, causal=True,
                num_splits=splits, **descales
            )
        except RuntimeError as error:
            if "QSA FP8 is not compiled in" in str(error):
                pytest.skip(str(error))
            raise
        assert torch.isfinite(out).all()
        assert torch.count_nonzero(out[counts == 0]) == 0
        relative = (out[sample].cpu().double() - expected).norm() / expected.norm().clamp_min(1e-12)
        assert relative <= .1


@pytest.mark.usefixtures("fp8_qsa_device")
@pytest.mark.parametrize("heads,kv_heads", [(6, 1), (24, 2)])
@pytest.mark.parametrize("splits", [0, 1, 4, 128])
def test_fp8_qsa_preserves_constant_value(heads, kv_heads, splits):
    from flash_attn_interface import flash_attn_with_kvcache

    q, k, v, _, pages, counts, cu, descales = _fp8_qsa_inputs(
        heads, kv_heads, 2051
    )
    v = torch.full(v.shape, 2.0, device=v.device).to(v.dtype)
    descales["v_descale"].fill_(.5)
    try:
        out = flash_attn_with_kvcache(
            q, k, v, cache_seqlens=counts, cu_seqlens_q=cu, max_seqlen_q=1,
            page_table=pages, softmax_scale=.0625, causal=True,
            num_splits=splits, **descales
        )
    except RuntimeError as error:
        if "QSA FP8 is not compiled in" in str(error):
            pytest.skip(str(error))
        raise
    assert torch.count_nonzero(out[0]) == 0
    assert torch.equal(out[1:], torch.ones_like(out[1:]))


@pytest.mark.usefixtures("fp8_qsa_device")
def test_fp8_qsa_rejects_unsupported_features():
    from flash_attn_interface import flash_attn_with_kvcache

    q, k, v, _, pages, counts, cu, descales = _fp8_qsa_inputs(6, 1, 65)
    kwargs = dict(cache_seqlens=counts, cu_seqlens_q=cu, max_seqlen_q=1, page_table=pages,
                  causal=True, softmax_scale=.0625, **descales)
    with pytest.raises(RuntimeError, match="FP8 QSA|QSA FP8"):
        flash_attn_with_kvcache(q, k[:0], v[:0], **kwargs)
    for changes in (dict(causal=False), dict(qsa_allow_aiu=True), dict(pack_gqa=False),
                    dict(num_splits=129), dict(q_descale=None), dict(q_descale=descales['q_descale'].half()),
                    dict(softmax_scale=0.0), dict(softmax_scale=-1.0), dict(softmax_scale=float('nan'))):
        with pytest.raises(RuntimeError, match="FP8 QSA|QSA FP8"):
            flash_attn_with_kvcache(q, k, v, **{**kwargs, **changes})


def _run_fp8_general_qsa(group, kv_heads, lengths, page, width, splits, layout="contiguous"):
    """Check native FP8 against dequantized CPU FP64, including every head and LSE."""
    from flash_attn_interface import flash_attn_with_kvcache

    torch.manual_seed(20261021)
    batch, total = len(lengths), sum(lengths)
    heads, pages_count = group * kv_heads, math.ceil(128 / page)
    q = torch.randn(total, heads, 256, device="cuda").to(torch.float8_e4m3fn)
    if layout == "interleaved":
        kv = torch.randn(pages_count, page, 2, kv_heads, 256, device="cuda").to(q.dtype)
        k, v = kv[:, :, 0], kv[:, :, 1]
    else:
        extra = int(layout == "gapped-pages")
        k, v = (torch.randn(pages_count, page + extra, kv_heads, 256, device="cuda")
                .to(q.dtype)[:, :page] for _ in range(2))
    if layout == "strided":
        q = torch.stack((q, q), dim=1)[:, 0]
        k = torch.stack((k, k), dim=2)[:, :, 0]
        v = torch.stack((v, v), dim=2)[:, :, 1]
    cu = torch.tensor([0] + list(accumulate(lengths)), device="cuda", dtype=torch.int32)
    indices = torch.randint(pages_count * page, (total, 1, width), device="cuda", dtype=torch.int32)
    if layout == "gapped-pages" and pages_count > 1:
        indices[..., 0] = page
        indices[..., -1] = pages_count * page - 1
    cache = torch.tensor([0, 1] + [width] * max(0, batch - 2), device="cuda", dtype=torch.int32)[:batch]
    if batch == 1:
        cache.fill_(width)
    base = torch.arange(batch * kv_heads, device="cuda").reshape(batch, kv_heads).float()
    descales = {name: value + base * .001 for name, value in
                (("q_descale", .7), ("k_descale", .8), ("v_descale", .9))}
    try:
        out, lse, *_ = flash_attn_with_kvcache(
            q, k, v, cache_seqlens=cache, cu_seqlens_q=cu, max_seqlen_q=max(lengths),
            page_table=indices, causal=True, num_splits=splits, return_softmax_lse=True,
            **descales)
    except RuntimeError as error:
        if "QSA FP8 is not compiled in" in str(error):
            pytest.skip(str(error))
        raise
    qf, kf, vf = (x.cpu().double() for x in (q, k.flatten(0, 1), v.flatten(0, 1)))
    indices, cache = indices.cpu().long(), cache.cpu().tolist()
    scales = {name: x.cpu().double() for name, x in descales.items()}
    expected = torch.zeros(q.shape, dtype=torch.float64)
    expected_lse = torch.full((heads, total), -torch.inf, dtype=torch.float64)
    row = 0
    for b, length in enumerate(lengths):
        for pos in range(length):
            count = max(0, min(width, cache[b] - length + pos + 1))
            if count:
                ids = indices[row, 0, :count]
                qq = qf[row].reshape(kv_heads, group, 256) * scales["q_descale"][b, :, None, None]
                kk = kf[ids] * scales["k_descale"][b, None, :, None]
                vv = vf[ids] * scales["v_descale"][b, None, :, None]
                scores = torch.einsum("hgd,khd->hgk", qq, kk) * .0625
                expected[row] = torch.einsum("hgk,khd->hgd", scores.softmax(-1), vv).reshape(heads, 256)
                expected_lse[:, row] = scores.logsumexp(-1).reshape(heads)
            row += 1
    assert out.dtype == torch.bfloat16 and out.shape == q.shape and torch.isfinite(out).all()
    relative = (out.cpu().double() - expected).norm() / expected.norm().clamp_min(1e-12)
    assert relative <= .1
    empty = torch.isneginf(expected_lse)
    assert torch.equal(torch.isneginf(lse.cpu()), empty)
    assert torch.count_nonzero(out.cpu().transpose(0, 1)[empty]) == 0
    torch.testing.assert_close(lse.cpu().double()[~empty], expected_lse[~empty], atol=.1, rtol=0)


@pytest.mark.usefixtures("fp8_qsa_device")
@pytest.mark.parametrize("group", range(1, 33))
@pytest.mark.parametrize("kv_heads", [1, 3])
@pytest.mark.parametrize("splits", [0, 1, 4])
def test_fp8_qsa_group_range(group, kv_heads, splits):
    _run_fp8_general_qsa(group, kv_heads, [1, 1, 1], 16, 65, splits)


@pytest.mark.usefixtures("fp8_qsa_device")
@pytest.mark.parametrize("group,kv_heads,lengths,page,width,layout", [
    (1, 1, [4, 4, 4], 1, 17, "contiguous"),
    (3, 1, [16, 9, 1], 16, 33, "contiguous"),
    (6, 1, [8, 3, 0, 1], 64, 65, "contiguous"),
    (8, 2, [16, 7, 1], 128, 31, "interleaved"),
    (12, 2, [4, 2, 1], 64, 65, "interleaved"),
    (16, 4, [2, 4, 1], 63, 32, "strided"),
    (6, 1, [4, 3, 1], 16, 33, "gapped-pages"),
    (17, 3, [1, 4, 2], 16, 33, "gapped-pages"),
    (17, 3, [1, 4, 2], 544, 33, "gapped-pages"),
    (24, 2, [4, 3, 1], 576, 65, "interleaved"),
    (31, 5, [1, 8, 2], 16, 65, "strided"),
    (32, 2, [17, 3, 1], 64, 2051, "contiguous"),
    (1, 1, [1], 64, 65537, "contiguous"),
])
@pytest.mark.parametrize("splits", [0, 1, 4, 128])
def test_fp8_qsa_general_sizes(group, kv_heads, lengths, page, width, layout, splits):
    _run_fp8_general_qsa(group, kv_heads, lengths, page, width, splits, layout)


@pytest.mark.usefixtures("fp8_qsa_device")
@pytest.mark.parametrize("batch,sq,heads", [(65536, 1, 1), (1, 16384, 32)])
@pytest.mark.parametrize("splits", [0, 1])
def test_fp8_qsa_large_axes(batch, sq, heads, splits):
    from flash_attn_interface import flash_attn_with_kvcache

    rows = batch * sq
    q = torch.zeros(rows, heads, 256, device="cuda").to(torch.float8_e4m3fn)
    k = torch.zeros(1, 1, 1, 256, device="cuda").to(q.dtype)
    v = torch.ones(1, 1, 1, 256, device="cuda").to(q.dtype)
    scale = torch.ones(batch, 1, device="cuda")
    out = flash_attn_with_kvcache(
        q, k, v, cache_seqlens=torch.full((batch,), sq, device="cuda", dtype=torch.int32),
        cu_seqlens_q=torch.arange(batch + 1, device="cuda", dtype=torch.int32) * sq, max_seqlen_q=sq,
        page_table=torch.zeros(rows, 1, 1, device="cuda", dtype=torch.int32),
        causal=True, num_splits=splits, q_descale=scale, k_descale=scale, v_descale=scale)
    assert torch.equal(out, torch.ones_like(out))


@pytest.mark.usefixtures("fp8_qsa_device")
@pytest.mark.parametrize("heads,kv_heads,message", [(33, 1, "at most 32"), (7, 2, "must divide")])
def test_fp8_qsa_rejects_invalid_head_counts(heads, kv_heads, message):
    from flash_attn_interface import flash_attn_with_kvcache

    q, k, v, _, pages, counts, cu, descales = _fp8_qsa_inputs(heads, kv_heads, 65)
    with pytest.raises(RuntimeError, match=message):
        flash_attn_with_kvcache(q, k, v, cache_seqlens=counts, cu_seqlens_q=cu,
            max_seqlen_q=1, page_table=pages, causal=True, **descales)


@pytest.mark.usefixtures("fp8_qsa_device")
def test_fp8_qsa_rejects_unaligned_storage():
    from flash_attn_interface import flash_attn_with_kvcache

    q, k, v, _, pages, counts, cu, descales = _fp8_qsa_inputs(6, 1, 65)
    bad_q = torch.empty(q.numel() + 1, dtype=q.dtype, device=q.device)[1:].view(q.shape)
    backing = torch.empty(k.numel() + 1, dtype=k.dtype, device=k.device)
    bad_k = backing.as_strided(k.shape, (k.stride(0) + 1, *k.stride()[1:]))
    for qq, kk in ((bad_q, k), (q, bad_k)):
        with pytest.raises(RuntimeError, match="16-byte"):
            flash_attn_with_kvcache(qq, kk, v, cache_seqlens=counts, cu_seqlens_q=cu,
                max_seqlen_q=1, page_table=pages, causal=True, **descales)


@pytest.mark.usefixtures("fp8_qsa_device")
@pytest.mark.parametrize("causal", [False, True])
def test_fp8_qsa_coexists_with_dense_attention(causal):
    from flash_attn_interface import flash_attn_func

    # Compile and execute ordinary FP8 FA alongside the native QSA path.
    _run_fp8_general_qsa(6, 1, [1, 1, 1], 64, 65, 1)
    torch.manual_seed(20261104)
    q, k, v = (torch.randn(2, length, 6, 256, device="cuda").to(torch.float8_e4m3fn)
               for length in (17, 65, 65))
    out = flash_attn_func(q, k, v, causal=causal)
    qf, kf, vf = (x.cpu().double() for x in (q, k, v))
    scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * .0625
    if causal:
        mask = torch.arange(65)[None, :] > torch.arange(17)[:, None] + 65 - 17
        scores.masked_fill_(mask, -torch.inf)
    expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), vf)
    relative = (out.cpu().double() - expected).norm() / expected.norm()
    assert torch.isfinite(out).all() and relative <= .1


@dataclass(frozen=True)
class QsaBackwardCase:
    name: str
    group: int
    kv_heads: int
    topk: int
    causal: bool = False
    cache: tuple = None
    splits: int = 1
    scale: float = None
    layout: str = ""
    strided_qdo: bool = False


QSA_BACKWARD_CASES = [
    QsaBackwardCase("g1-kv1-singleton", 1, 1, 1),
    QsaBackwardCase("g1-kv2-causal127-kstride", 1, 2, 127,
                    causal=True, cache=(126, 0, 127), layout="k"),
    QsaBackwardCase("g3-kv1-cacheignored128-vstride", 3, 1, 128,
                    cache=(0, 0, 1), layout="v"),
    QsaBackwardCase("g3-kv2-nocache129-auto", 3, 2, 129,
                    causal=True, splits=0, scale=0.13),
    QsaBackwardCase("g6-kv1-causal127-mixedmask", 6, 1, 127,
                    causal=True, cache=(1, 0, 2)),
    QsaBackwardCase("g6-kv2-128-allstrided", 6, 2, 128,
                    layout="kv", strided_qdo=True),
    QsaBackwardCase("g12-kv1-causal129-singlevalid", 12, 1, 129,
                    causal=True, cache=(1, 0, 1), scale=0.13, layout="kv"),
    QsaBackwardCase("g12-kv2-causal2051-auto", 12, 2, 2051,
                    causal=True, cache=(2051, 0, 2049), splits=0, layout="kv"),
    QsaBackwardCase("g32-kv1-cacheignored2051-auto", 32, 1, 2051,
                    cache=(0, 0, 1), splits=0, scale=0.13, strided_qdo=True),
    QsaBackwardCase("g32-kv2-nocache-singleton", 32, 2, 1,
                    causal=True, layout="kv"),
    QsaBackwardCase("g3-kv2-causal129-allmasked", 3, 2, 129,
                    causal=True, cache=(0, 0, 0), layout="kv", strided_qdo=True),
]


def _qsa_training_interface():
    # Reuse the forward tests' local-interface resolution, including isolated runs.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import flash_attn_interface
    return flash_attn_interface


def _qsa_backward_inputs(case, dtype):
    torch.manual_seed(1234)
    device, dim, total_q = "cuda", 256, 5
    heads = case.group * case.kv_heads
    if case.strided_qdo:
        q = torch.randn(total_q * 2, heads * 2, dim, device=device, dtype=dtype)[::2, ::2]
        dout = torch.randn(total_q * 3, heads * 3, dim, device=device, dtype=dtype)[1::3, 2::3]
    else:
        q = torch.randn(total_q, heads, dim, device=device, dtype=dtype)
        dout = torch.randn_like(q)
    if "k" in case.layout:
        k = torch.randn(8, 48, case.kv_heads * 2, dim, device=device, dtype=dtype)[1::2, 1::3, ::2]
    else:
        k = torch.randn(4, 16, case.kv_heads, dim, device=device, dtype=dtype)
    if "v" in case.layout:
        # Independently stride pages, rows, and heads; never borrow K's strides.
        v = torch.randn(12, 32, case.kv_heads * 3, dim, device=device, dtype=dtype)[2::3, ::2, 1::3]
    else:
        v = torch.randn(4, 16, case.kv_heads, dim, device=device, dtype=dtype)
    q, k, v = [x.mul_(0.5).detach().requires_grad_() for x in (q, k, v)]
    dout.mul_(0.5)
    assert all(x.is_leaf and x.stride(-1) == 1 for x in (q, k, v))
    if case.layout == "kv":
        assert k.stride() != v.stride() and not k.is_contiguous() and not v.is_contiguous()
    if case.strided_qdo:
        assert not q.is_contiguous() and not dout.is_contiguous()
    # A bounded subset crosses page boundaries and includes both legal endpoints.
    # The other physical tokens must receive exactly zero dK AND dV.
    pool = torch.tensor([0, 1, 14, 15, 16, 17, 30, 31, 32, 47, 48, 62, 63], device=device)
    indices = pool[torch.randint(len(pool), (total_q, 1, case.topk), device=device)].to(torch.int32)
    indices[:, :, 0] = 63
    if case.topk > 1:
        indices[:, :, 1] = 63  # Duplicate within each row AND across queries/batches.
    if case.topk > 2:
        indices[:, :, 2] = 0
    cu = torch.tensor([0, 2, 2, 5], device=device, dtype=torch.int32)  # lengths [2, 0, 3]
    cache = None if case.cache is None else torch.tensor(case.cache, device=device, dtype=torch.int32)
    return q, k, v, dout, indices, cu, cache


def _qsa_backward_valid(indices, cu, cache, causal):
    columns = torch.arange(indices.shape[-1], device=indices.device)[None, :]
    valid = (columns < indices.shape[-1]).expand(indices.shape[0], -1).clone()
    # Noncausal ignores cache; causal without cache falls back to noncausal.
    if causal and cache is not None:
        offsets, cache_lengths = cu.tolist(), cache.tolist()
        for b, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
            qpos = torch.arange(hi - lo, device=indices.device)[:, None]
            valid[lo:hi] &= (columns < cache_lengths[b]) & (
                columns <= qpos + cache_lengths[b] - (hi - lo))
    return valid


def _qsa_differentiable_sparse_ref(q, k, v, indices, cu, cache, *, scale=None, causal=False):
    assert all(x.is_leaf for x in (q, k, v))
    valid = _qsa_backward_valid(indices, cu, cache, causal)
    group, dim = q.shape[1] // k.shape[2], q.shape[2]
    scale = dim ** -0.5 if scale is None else scale
    idx = indices[:, 0].long()
    kk, vv = k[idx // k.shape[1], idx % k.shape[1]], v[idx // v.shape[1], idx % v.shape[1]]
    qq = q.reshape(q.shape[0], k.shape[2], group, dim)
    scores = torch.einsum("rhgd,rthd->rhgt", qq, kk) * scale
    mask = valid[:, None, None, :]
    nonempty = mask.any(dim=-1, keepdim=True)
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    # Prevent NaNs in the softmax backward itself: fix ALL-masked rows BEFORE
    # normalization, then give those rows zero weights (not sentinel indices).
    safe_scores = torch.where(nonempty, masked_scores, torch.zeros_like(scores))
    weights = torch.softmax(safe_scores.float(), dim=-1).to(q.dtype).masked_fill(~mask, 0)
    out = torch.einsum("rhgt,rthd->rhgd", weights, vv).reshape_as(q)
    lse = torch.logsumexp(safe_scores.float(), dim=-1).masked_fill(~nonempty.squeeze(-1), float("-inf"))
    return out, lse.reshape(q.shape[0], q.shape[1]).transpose(0, 1), valid


@contextmanager
def _qsa_full_precision_matmul():
    precision = torch.get_float32_matmul_precision()
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def _qsa_reference_gradients(q, k, v, dout, indices, cu, cache, case, dtype):
    # Both graphs own original, shaped physical-pool leaves, even for strided KV.
    with _qsa_full_precision_matmul(), torch.autocast(device_type=q.device.type, enabled=False):
        leaves = tuple(x.detach().to(dtype).clone().requires_grad_() for x in (q, k, v))
        out, lse, valid = _qsa_differentiable_sparse_ref(
            *leaves, indices, cu, cache, scale=case.scale, causal=case.causal)
        grads = torch.autograd.grad(out, leaves, dout.to(dtype))
    return out.detach(), lse.detach(), valid, tuple(g.detach() for g in grads)


@torch.no_grad()
def _qsa_mma_rounded_backward(q, k, v, dout, out, lse, indices, cu, cache,
                              *, scale=None, causal=False):
    rows, heads, dim = q.shape
    kv_heads, group = k.shape[2], heads // k.shape[2]
    scale = dim ** -0.5 if scale is None else scale
    dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
    dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
    if rows == 0 or indices.shape[-1] == 0:
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
    with _qsa_full_precision_matmul(), torch.autocast(device_type=q.device.type, enabled=False):
        # Logical coordinates preserve independent K/V strides and physical duplicates.
        columns = torch.arange(indices.shape[-1], device=q.device)[None, :]
        valid = torch.ones((rows, indices.shape[-1]), device=q.device, dtype=torch.bool)
        if causal and cache is not None:
            offsets, lengths = cu.tolist(), cache.tolist()
            for b, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
                qpos = torch.arange(hi - lo, device=q.device)[:, None]
                valid[lo:hi] = (columns < lengths[b]) & (columns <= qpos + lengths[b] - (hi - lo))
        counts = valid.sum(-1)
        saved_lse = lse.transpose(0, 1).reshape(rows, kv_heads, group)
        singleton = (counts[:, None, None] == 1) & torch.isfinite(saved_lse)
        active_head = (counts[:, None, None] > 0) & (saved_lse != float("-inf"))
        qq = torch.zeros((rows, kv_heads, 32, dim), device=q.device, dtype=torch.float32)
        do = torch.zeros_like(qq)
        qq[:, :, :group] = torch.where((active_head & ~singleton)[..., None],
                                       q.float().reshape(rows, kv_heads, group, dim), 0)
        do[:, :, :group] = torch.where(active_head[..., None],
                                       dout.float().reshape(rows, kv_heads, group, dim), 0)
        delta = (dout.float() * out.float()).sum(-1).reshape(rows, kv_heads, group)
        delta = torch.where(active_head & ~singleton, delta, 0)
        head_coords = torch.arange(kv_heads, device=q.device)[None, None, :]
        for start in range(0, indices.shape[-1], 32):
            end = min(start + 32, indices.shape[-1])
            width = end - start
            idx = indices[:, 0, start:end].long()
            kk = torch.zeros((rows, kv_heads, 32, dim), device=q.device, dtype=torch.float32)
            vv = torch.zeros_like(kk)
            tile_mask = valid[:, None, start:end, None]
            kk[:, :, :width] = torch.where(tile_mask,
                k[idx // k.shape[1], idx % k.shape[1]].permute(0, 2, 1, 3).float(), 0)
            vv[:, :, :width] = torch.where(tile_mask,
                v[idx // v.shape[1], idx % v.shape[1]].permute(0, 2, 1, 3).float(), 0)
            scores = (qq @ kk.transpose(-1, -2))[:, :, :group, :width]
            dp = (do @ vv.transpose(-1, -2))[:, :, :group, :width]
            active = valid[:, None, None, start:end] & (saved_lse[..., None] != float("-inf"))
            logits = (scores * scale - saved_lse[..., None]).masked_fill(~active, float("-inf"))
            probability = torch.exp(logits)
            probability = torch.where(singleton[..., None], valid[:, None, None, start:end].float(), probability)
            ds = (probability * (dp - delta[..., None])).masked_fill(~active | singleton[..., None], 0)
            p_mma = torch.zeros((rows, kv_heads, 32, 32), device=q.device, dtype=torch.float32)
            ds_mma = torch.zeros_like(p_mma)
            p_mma[:, :, :group, :width] = probability.to(q.dtype).float()
            ds_mma[:, :, :group, :width] = ds.to(q.dtype).float()
            # Each tile contracts padded GQA heads, then contributes to FP32 pools.
            dq_tile = (ds_mma @ kk)[:, :, :group] * scale
            dq.add_(torch.where((active_head & ~singleton)[..., None], dq_tile, 0).reshape_as(dq))
            dk_tile = (ds_mma.transpose(-1, -2) @ qq)[:, :, :width] * scale
            dv_tile = (p_mma.transpose(-1, -2) @ do)[:, :, :width]
            coords = (idx[..., None] * kv_heads + head_coords).reshape(-1, 1).expand(-1, dim)
            dk.reshape(-1, dim).scatter_add_(0, coords, dk_tile.permute(0, 2, 1, 3).reshape(-1, dim))
            dv.reshape(-1, dim).scatter_add_(0, coords, dv_tile.permute(0, 2, 1, 3).reshape(-1, dim))
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def _qsa_check_error(actual, oracle, baseline, dtype, label, *, mma_rounding=False):
    assert torch.isfinite(actual).all(), f"{label}: nonfinite kernel result"
    assert torch.isfinite(oracle).all() and torch.isfinite(baseline).all(), f"{label}: nonfinite reference"
    error = (actual.float() - oracle.float()).abs()
    baseline_error = (baseline.float() - oracle.float()).abs()
    # Dense-test style: 2x independent low-precision error plus one BF16/FP16
    # ulp near 1/16 (rounded slightly upward), with a ten-times tighter mean bound.
    max_margin, mean_margin = (5e-4, 5e-5) if dtype == torch.bfloat16 else (6e-5, 6e-6)
    maximum, mean = error.max().item(), error.mean().item()
    pt_max, pt_mean = baseline_error.max().item(), baseline_error.mean().item()
    max_limit, mean_limit = 2 * pt_max + max_margin, 2 * pt_mean + mean_margin
    if mma_rounding:
        element_limit = max_margin + torch.finfo(dtype).eps * oracle.float().abs()
        max_limit, mean_limit = element_limit.max().item(), mean_margin
    report = (f"{label}: max/mean={maximum:.6g}/{mean:.6g}, "
              f"lowp={pt_max:.6g}/{pt_mean:.6g}, limits={max_limit:.6g}/{mean_limit:.6g}")
    print(report)
    if mma_rounding:
        # Different FP32 reduction orders can straddle the final dtype rounding boundary.
        assert (error <= element_limit).all() and mean <= mean_margin, report
    else:
        assert maximum <= max_limit and mean <= mean_limit, report


def _qsa_native_backward(dout, q, k, v, out, lse, indices, cu, max_q=3,
                         cache=None, scale=None, causal=False, deterministic=False):
    # Importing the existing Python interface also registers the planned native op.
    _qsa_training_interface()
    return torch.ops.flash_attn_3.qsa_bwd(
        dout.detach(), q.detach(), k.detach(), v.detach(), out.detach(), lse.detach(),
        indices, cu, max_q, cache, scale, causal, deterministic)


def _qsa_check_gradients(grads, oracle, baseline, inputs, indices, valid, dtype, label,
                         *, mma_rounding=False):
    assert isinstance(grads, (tuple, list)) and len(grads) == 3, f"{label}: qsa_bwd must return dq, dk, dv"
    for name, grad, ref, pt, original in zip(("dQ", "dK", "dV"), grads, oracle, baseline, inputs):
        assert grad.shape == original.shape and grad.dtype == original.dtype and grad.device == original.device
        _qsa_check_error(grad, ref, pt, dtype, f"{label}/{name}", mma_rounding=mma_rounding)
    touched = torch.zeros(inputs[1].shape[0] * inputs[1].shape[1], device=indices.device, dtype=torch.bool)
    touched[indices[:, 0][valid].long()] = True
    untouched = (~touched).reshape(inputs[1].shape[:2])
    for name, grad in zip(("dK", "dV"), grads[1:]):
        assert torch.count_nonzero(grad[untouched]).item() == 0, f"{label}/{name}: untouched pool tokens"
    row_counts = valid.sum(-1)
    assert torch.count_nonzero(grads[0][row_counts <= 1]).item() == 0, f"{label}: allmask/singleton dQ must be zero"
    if (row_counts <= 1).all():
        assert torch.count_nonzero(grads[1]).item() == 0, f"{label}: singleton/allmask dK must be zero"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("case", QSA_BACKWARD_CASES, ids=lambda case: case.name)
def test_qsa_backward_sparse_reference(case, dtype):
    interface = _qsa_training_interface()
    q, k, v, dout, indices, cu, cache = _qsa_backward_inputs(case, dtype)
    ref, ref_lse, valid, ref_grads = _qsa_reference_gradients(
        q, k, v, dout, indices, cu, cache, case, torch.float32)
    pt, pt_lse, _, pt_grads = _qsa_reference_gradients(q, k, v, dout, indices, cu, cache, case, dtype)
    kwargs = dict(page_table=indices, cu_seqlens_q=cu, max_seqlen_q=3,
                  cache_seqlens=cache, softmax_scale=case.scale, causal=case.causal,
                  num_splits=case.splits, pack_gqa=True, return_softmax_lse=True)
    with torch.no_grad():
        inference = interface.flash_attn_with_kvcache(q, k, v, **kwargs)
    result = interface.flash_attn_with_kvcache(q, k, v, **kwargs)
    assert isinstance(result, tuple) and len(result) == 4, "preserve public output/LSE/accumulator arity"
    out, lse = result[:2]
    assert out.requires_grad, "QSA kvcache output must support autograd"
    assert torch.equal(out.detach(), inference[0]) and torch.equal(lse.detach(), inference[1])
    _qsa_check_error(out, ref, pt, dtype, f"{case.name}/out")
    assert lse.shape == (q.shape[1], q.shape[0]) and lse.dtype == torch.float32
    nonempty = valid.any(-1)
    if nonempty.any():
        _qsa_check_error(lse[:, nonempty], ref_lse[:, nonempty], pt_lse[:, nonempty],
                         dtype, f"{case.name}/natural-log-LSE")
    # FA forward paths may use either infinity sign as their empty-row sentinel.
    assert torch.isinf(lse[:, ~nonempty]).all() and not torch.isnan(lse).any()
    assert torch.count_nonzero(out[~nonempty]).item() == 0
    rounded = _qsa_mma_rounded_backward(q, k, v, dout, out, lse, indices, cu, cache,
                                         scale=case.scale, causal=case.causal)
    native = _qsa_native_backward(dout, q, k, v, out, lse, indices, cu,
                                  cache=cache, scale=case.scale, causal=case.causal)
    public = torch.autograd.grad(out, (q, k, v), dout)
    for label, grads in (("native", native), ("autograd", public)):
        _qsa_check_gradients(grads, rounded, rounded, (q, k, v), indices, valid,
                             dtype, f"{case.name}/{label}/saved-MMA", mma_rounding=True)
        _qsa_check_gradients(grads, ref_grads, pt_grads, (q, k, v), indices, valid,
                             dtype, f"{case.name}/{label}")
    if case.topk == 1:
        # Also exercise the ordinary output-only return path with real autograd.
        only_out = interface.flash_attn_with_kvcache(q, k, v, **dict(kwargs, return_softmax_lse=False))
        assert isinstance(only_out, torch.Tensor) and torch.equal(only_out.detach(), inference[0])
        only_grads = torch.autograd.grad(only_out, (q, k, v), dout)
        _qsa_check_gradients(only_grads, rounded, rounded, (q, k, v), indices, valid,
                             dtype, f"{case.name}/output-only-autograd/saved-MMA", mma_rounding=True)
        _qsa_check_gradients(only_grads, ref_grads, pt_grads, (q, k, v), indices, valid,
                             dtype, f"{case.name}/output-only-autograd")
    if (case.causal and cache is None) or (not case.causal and cache is not None):
        # Exercise fallback/ignored-cache semantics independently in BOTH directions.
        equivalent = dict(kwargs, causal=False, cache_seqlens=None)
        with torch.no_grad():
            out_equiv, lse_equiv, *_ = interface.flash_attn_with_kvcache(q, k, v, **equivalent)
        assert torch.equal(out.detach(), out_equiv) and torch.equal(lse.detach(), lse_equiv)
        grads_equiv = _qsa_native_backward(dout, q, k, v, out_equiv, lse_equiv, indices, cu,
                                          cache=None, scale=case.scale, causal=False)
        _qsa_check_gradients(grads_equiv, rounded, rounded, (q, k, v), indices, valid,
                             dtype, f"{case.name}/equivalent-native/saved-MMA", mma_rounding=True)
        _qsa_check_gradients(grads_equiv, ref_grads, pt_grads, (q, k, v), indices, valid,
                             dtype, f"{case.name}/equivalent-native")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("bad_index", [-1, 64], ids=["negative-no-sentinel", "pool-upper-bound"])
def test_qsa_backward_rejects_invalid_physical_index(dtype, bad_index):
    case = QsaBackwardCase("invalid-index", 3, 2, 129)
    q, k, v, dout, indices, cu, cache = _qsa_backward_inputs(case, dtype)
    indices[0, 0, -1] = bad_index
    # Never send invalid physical indices to forward; synthetic O/LSE are sufficient.
    out, lse = torch.zeros_like(q), torch.zeros(q.shape[1], q.shape[0], device=q.device)
    with pytest.raises(RuntimeError, match=r"(?i)(index|indices|pool|range|bounds|negative)"):
        _qsa_native_backward(dout, q, k, v, out, lse, indices, cu, cache=cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("bad_cu", ["negative", "nonzero-start", "decreasing", "end-short",
                                   "end-long", "max-too-short", "rank", "dtype"])
def test_qsa_backward_rejects_invalid_cu(dtype, bad_cu):
    case = QsaBackwardCase("invalid-cu", 3, 2, 129)
    q, k, v, dout, indices, cu, cache = _qsa_backward_inputs(case, dtype)
    values = {"negative": [-1, 2, 2, 5], "nonzero-start": [1, 2, 2, 5],
              "decreasing": [0, 3, 2, 5], "end-short": [0, 2, 2, 4], "end-long": [0, 2, 2, 6]}
    max_q = 2 if bad_cu == "max-too-short" else 3
    if bad_cu in values:
        cu = torch.tensor(values[bad_cu], device=q.device, dtype=torch.int32)
    elif bad_cu == "rank":
        cu = cu.reshape(2, 2)
    elif bad_cu == "dtype":
        cu = cu.to(torch.int64)
    # As above, malformed cumulative lengths must NEVER run through forward.
    out, lse = torch.zeros_like(q), torch.zeros(q.shape[1], q.shape[0], device=q.device)
    with pytest.raises(RuntimeError, match=r"(?i)(cu.?seqlens|cuq|cumulative|max.?seqlen|int32|shape|dimension|start|end|monotonic)"):
        _qsa_native_backward(dout, q, k, v, out, lse, indices, cu, max_q, cache=cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("variant", ["softcap", "window", "aiu"])
def test_qsa_backward_rejects_unsupported_training(dtype, variant):
    interface = _qsa_training_interface()
    case = QsaBackwardCase("unsupported-training", 3, 2, 128)
    q, k, v, dout, indices, cu, _ = _qsa_backward_inputs(case, dtype)
    # For AIU this is a truthful 16-token contiguous-run certificate, not a bad gather.
    indices.copy_((torch.arange(case.topk, device=q.device, dtype=torch.int32) % 64)[None, None, :])
    options = {"softcap": {"softcap": 2.0}, "window": {"window_size": (2, 0)},
               "aiu": {"qsa_allow_aiu": True}}[variant]
    training_error = r"(?is)(QSA|query.sparse).*(backward|training|grad)|(backward|training|grad).*(QSA|query.sparse)"
    with pytest.raises((RuntimeError, ValueError, NotImplementedError), match=training_error):
        out = interface.flash_attn_with_kvcache(
            q, k, v, page_table=indices, cu_seqlens_q=cu, max_seqlen_q=3,
            causal=False, num_splits=1, pack_gqa=True, **options)
        torch.autograd.grad(out, (q, k, v), dout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_qsa_backward_rejects_native_deterministic(dtype):
    case = QsaBackwardCase("deterministic", 3, 2, 129)
    q, k, v, dout, indices, cu, cache = _qsa_backward_inputs(case, dtype)
    out, lse = torch.zeros_like(q), torch.zeros(q.shape[1], q.shape[0], device=q.device)
    with pytest.raises(RuntimeError, match=r"(?i)deterministic"):
        _qsa_native_backward(dout, q, k, v, out, lse, indices, cu, cache=cache, deterministic=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_qsa_backward_python_fake_opcheck_and_arity(dtype):
    from torch._subclasses.fake_tensor import FakeTensorMode

    interface = _qsa_training_interface()
    case = QsaBackwardCase("fake-arity", 3, 2, 129, causal=True, cache=(1, 0, 2), layout="kv")
    q, k, v, dout, indices, cu, cache = _qsa_backward_inputs(case, dtype)
    schema = torch.ops.flash_attn_3.qsa_bwd.default._schema
    assert len(schema.arguments) == 13 and len(schema.returns) == 3
    torch.library.opcheck(
        interface._flash_attn_forward, (q, k, v),
        dict(page_table=indices, cu_seqlens_q=cu, max_seqlen_q=3, seqused_k=cache,
             causal=True, num_splits=1, pack_gqa=True),
        test_utils=("test_schema", "test_faketensor"))
    mode = FakeTensorMode()
    with mode:
        fq, fk, fv, fdout, findices, fcu, fcache = [
            mode.from_tensor(x) for x in (q, k, v, dout, indices, cu, cache)]
        kwargs = dict(page_table=findices, cu_seqlens_q=fcu, max_seqlen_q=3,
                      cache_seqlens=fcache, causal=True, num_splits=1, pack_gqa=True)
        result = interface.flash_attn_with_kvcache(fq, fk, fv, return_softmax_lse=True, **kwargs)
        assert isinstance(result, tuple) and len(result) == 4
        fout, flse = result[:2]
        assert fout.shape == fq.shape and fout.dtype == dtype and fout.requires_grad
        assert flse.shape == (fq.shape[1], fq.shape[0]) and flse.dtype == torch.float32
        only_out = interface.flash_attn_with_kvcache(fq, fk, fv, **kwargs)
        assert isinstance(only_out, torch.Tensor) and only_out.shape == fq.shape and only_out.requires_grad
        # Traces the Python autograd route and its backward fake registration,
        # without needing to know the parent's private backward-wrapper name.
        grads = torch.autograd.grad(fout, (fq, fk, fv), fdout)
        for grad, original in zip(grads, (fq, fk, fv)):
            assert grad.shape == original.shape and grad.dtype == original.dtype and grad.device == original.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_qsa_backward_scalar_cache_lengths(dtype):
    interface = _qsa_training_interface()
    case = QsaBackwardCase("scalar-cache", 3, 2, 129, causal=True, cache=(2, 2, 2))
    q, k, v, dout, indices, cu, cache = _qsa_backward_inputs(case, dtype)
    out = interface.flash_attn_with_kvcache(
        q, k, v, page_table=indices, cu_seqlens_q=cu, max_seqlen_q=3,
        cache_seqlens=2, causal=True, num_splits=1, pack_gqa=True,
    )
    grads = torch.autograd.grad(out, (q, k, v), dout)
    _, _, valid, oracle = _qsa_reference_gradients(q, k, v, dout, indices, cu, cache, case, torch.float32)
    _, _, _, baseline = _qsa_reference_gradients(q, k, v, dout, indices, cu, cache, case, dtype)
    _qsa_check_gradients(grads, oracle, baseline, (q, k, v), indices, valid, dtype, "scalar-cache")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
def test_qsa_backward_saved_index_version_and_higher_order():
    interface = _qsa_training_interface()
    case = QsaBackwardCase("saved-index", 3, 1, 129)
    q, k, v, dout, indices, cu, cache = _qsa_backward_inputs(case, torch.bfloat16)
    kwargs = dict(page_table=indices, cu_seqlens_q=cu, max_seqlen_q=3,
                  causal=False, num_splits=1, pack_gqa=True)
    out = interface.flash_attn_with_kvcache(q, k, v, **kwargs)
    with pytest.raises(RuntimeError, match="higher-order"):
        torch.autograd.grad(out, (q, k, v), dout, create_graph=True)
    out = interface.flash_attn_with_kvcache(q, k, v, **kwargs)
    indices.add_(0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(out, (q, k, v), dout)


QSA_BACKWARD_MMA_CASES = [
    QsaBackwardCase("mma-g17-topk31", 17, 2, 31, causal=True, cache=(31, 0, 30),
                    layout="kv", strided_qdo=True),
    QsaBackwardCase("mma-g17-topk32", 17, 2, 32, cache=(0, 0, 1),
                    scale=0.13, layout="kv", strided_qdo=True),
    QsaBackwardCase("mma-g17-topk33", 17, 2, 33, causal=True,
                    layout="kv", strided_qdo=True),
    QsaBackwardCase("mma-g17-topk63", 17, 2, 63, scale=0.13,
                    layout="kv", strided_qdo=True),
    QsaBackwardCase("mma-g17-topk64", 17, 2, 64, layout="kv", strided_qdo=True),
    QsaBackwardCase("mma-g17-topk65", 17, 2, 65, causal=True, cache=(34, 0, 65),
                    scale=0.13, layout="kv", strided_qdo=True),
]


def _qsa_coordinate_backward_inputs(case, dtype):
    q, k, v, dout, indices, cu, cache = _qsa_backward_inputs(case, dtype)
    channel = torch.arange(q.shape[-1], device=q.device)
    row = torch.arange(q.shape[0], device=q.device)[:, None, None]
    head = torch.arange(q.shape[1], device=q.device)[None, :, None]
    token = torch.arange(k.shape[0] * k.shape[1], device=q.device).reshape(*k.shape[:2], 1, 1)
    kv_head = torch.arange(k.shape[2], device=q.device)[None, None, :, None]
    with torch.no_grad():
        q.copy_(((row * 7 + head * 11 + channel * 3 + head * channel) % 29 - 14) / 64)
        dout.copy_(((row * 13 + head * 5 + channel * 7 + row * channel) % 31 - 15) / 64)
        k.copy_(((token * 7 + kv_head * 11 + channel * 3 + token * channel) % 31 - 15) / 64)
        v.copy_(((token * 13 + kv_head * 17 + channel * 5 + kv_head * channel) % 37 - 18) / 64)
        col = torch.arange(case.topk, device=q.device)[None, :]
        physical = (row[:, 0] * 19 + col * 7 + (col // 32) * 3) % 64
        indices.copy_(physical[:, None, :])
        indices[:, :, :min(2, case.topk)] = 63
        if case.topk > 2:
            indices[:, :, 2] = 0
        for boundary in (32, 64):
            if case.topk > boundary:
                indices[:, :, boundary] = 63
    return q, k, v, dout, indices, cu, cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("case", QSA_BACKWARD_MMA_CASES, ids=lambda case: case.name)
def test_qsa_backward_mma_tile_boundaries(case, dtype, repeats=1):
    interface = _qsa_training_interface()
    q, k, v, dout, indices, cu, cache = _qsa_coordinate_backward_inputs(case, dtype)
    if repeats > 1:
        q = q.repeat_interleave(repeats, dim=0).detach().requires_grad_()
        dout = dout.repeat_interleave(repeats, dim=0)
        indices = indices.repeat_interleave(repeats, dim=0)
        cu = cu * repeats
    max_q = 3 * repeats
    ref, ref_lse, valid, oracle = _qsa_reference_gradients(
        q, k, v, dout, indices, cu, cache, case, torch.float32)
    pt, pt_lse, _, baseline = _qsa_reference_gradients(q, k, v, dout, indices, cu, cache, case, dtype)
    out, lse, *_ = interface.flash_attn_with_kvcache(
        q, k, v, page_table=indices, cu_seqlens_q=cu, max_seqlen_q=max_q,
        cache_seqlens=cache, softmax_scale=case.scale, causal=case.causal,
        num_splits=case.splits, pack_gqa=True, return_softmax_lse=True)
    _qsa_check_error(out, ref, pt, dtype, f"{case.name}/out")
    _qsa_check_error(lse, ref_lse, pt_lse, dtype, f"{case.name}/natural-log-LSE")
    rounded = _qsa_mma_rounded_backward(q, k, v, dout, out, lse, indices, cu, cache,
                                         scale=case.scale, causal=case.causal)
    native = _qsa_native_backward(dout, q, k, v, out, lse, indices, cu, max_q,
                                  cache=cache, scale=case.scale, causal=case.causal)
    public = torch.autograd.grad(out, (q, k, v), dout)
    for label, grads in (("native", native), ("autograd", public)):
        _qsa_check_gradients(grads, rounded, rounded, (q, k, v), indices, valid,
                             dtype, f"{case.name}/{label}/saved-MMA", mma_rounding=True)
        _qsa_check_gradients(grads, oracle, baseline, (q, k, v), indices, valid,
                             dtype, f"{case.name}/{label}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("singleton", [False, True], ids=["mixed-heads", "singletons"])
@pytest.mark.parametrize("topk", [33, 1537])
def test_qsa_backward_mma_synthetic_saved_state(dtype, singleton, topk):
    case = QsaBackwardCase("mma-synthetic", 17, 2, topk, scale=0.13, layout="kv", strided_qdo=True,
                           causal=singleton, cache=(1, 0, 1) if singleton else None)
    q, k, v, dout, indices, cu, cache = _qsa_coordinate_backward_inputs(case, dtype)
    repeats = 16 if topk > 64 else 1
    if repeats > 1:
        q = q.repeat_interleave(repeats, dim=0).detach().requires_grad_()
        dout = dout.repeat_interleave(repeats, dim=0)
        indices = indices.repeat_interleave(repeats, dim=0)
        cu = cu * repeats
    with torch.no_grad(), _qsa_full_precision_matmul(), torch.autocast(device_type=q.device.type, enabled=False):
        saved_out, lse, valid = _qsa_differentiable_sparse_ref(
            q.detach().float(), k.detach().float(), v.detach().float(), indices, cu, cache,
            scale=case.scale, causal=case.causal)
        out = (saved_out + 0.015625).to(dtype)
        lse = (lse + 0.03125).contiguous()
        head = torch.arange(q.shape[1], device=q.device)[:, None]
        row = torch.arange(q.shape[0], device=q.device)[None, :]
        inactive = (head < case.group) | ((head * 3 + row) % 5 == 0) | (row == 0)
        lse.masked_fill_(inactive, float("-inf"))
    rounded = _qsa_mma_rounded_backward(q, k, v, dout, out, lse, indices, cu, cache,
                                         scale=case.scale, causal=case.causal)
    grads = _qsa_native_backward(dout, q, k, v, out, lse, indices, cu, 3 * repeats,
                                 cache=cache, scale=case.scale, causal=case.causal)
    _qsa_check_gradients(grads, rounded, rounded, (q, k, v), indices, valid,
                         dtype, f"{case.name}/{singleton}/saved-MMA", mma_rounding=True)
    assert torch.count_nonzero(grads[0][torch.isneginf(lse).transpose(0, 1)]).item() == 0
    for grad in grads[1:]:
        assert torch.count_nonzero(grad[:, :, 0]).item() == 0, "inactive KV head must not receive scatter"
    if singleton:
        assert torch.count_nonzero(grads[0]).item() == 0 and torch.count_nonzero(grads[1]).item() == 0
        expected_dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
        for query in range(q.shape[0]):
            if valid[query].sum().item() != 1:
                continue
            token = indices[query, 0, 0].item()
            for head in range(q.shape[1]):
                if torch.isfinite(lse[head, query]):
                    expected_dv[token // v.shape[1], token % v.shape[1], head // case.group].add_(dout[query, head].float())
        assert torch.equal(grads[2], expected_dv.to(dtype)), "finite singleton dV must equal scattered dO"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("empty", ["topk0", "total_q0"])
def test_qsa_backward_mma_empty_native(dtype, empty):
    case = QsaBackwardCase("mma-empty", 17, 2, 33, causal=True, cache=(33, 0, 33), layout="kv")
    q, k, v, dout, indices, cu, cache = _qsa_coordinate_backward_inputs(case, dtype)
    max_q = 3
    if empty == "topk0":
        indices = indices[:, :, :0].contiguous()
    else:
        q, dout, indices = q[:0], dout[:0], indices[:0]
        cu = torch.zeros_like(cu)
        max_q = 0
    out = torch.zeros_like(q)
    lse = torch.full((q.shape[1], q.shape[0]), float("-inf"), device=q.device, dtype=torch.float32)
    rounded = _qsa_mma_rounded_backward(q, k, v, dout, out, lse, indices, cu, cache,
                                         causal=case.causal)
    grads = _qsa_native_backward(dout, q, k, v, out, lse, indices, cu, max_q,
                                 cache=cache, causal=case.causal)
    for grad, ref, original in zip(grads, rounded, (q, k, v)):
        assert grad.shape == original.shape and grad.dtype == original.dtype and grad.device == original.device
        assert torch.count_nonzero(grad).item() == 0 and torch.equal(grad, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("singleton,topk", [(False, 33), (True, 1), (False, 1537)],
                         ids=["mixed-heads-33", "singletons", "mixed-heads-1537"])
def test_qsa_backward_mma_nonfinite_inactive_operands(dtype, singleton, topk):
    case = QsaBackwardCase("mma-inactive-nan", 17, 2, topk,
                           scale=0.13, layout="kv", strided_qdo=True)
    q, k, v, dout, indices, cu, cache = _qsa_coordinate_backward_inputs(case, dtype)
    repeats = 16 if topk > 64 else 1
    if repeats > 1:
        q = q.repeat_interleave(repeats, dim=0).detach().requires_grad_()
        dout = dout.repeat_interleave(repeats, dim=0)
        indices = indices.repeat_interleave(repeats, dim=0)
        cu = cu * repeats
    out = torch.zeros_like(q)
    lse = torch.zeros(q.shape[1], q.shape[0], device=q.device)
    if topk > 64:
        with torch.no_grad(), _qsa_full_precision_matmul():
            _, lse, _ = _qsa_differentiable_sparse_ref(
                q.detach().float(), k.detach().float(), v.detach().float(), indices, cu, cache,
                scale=case.scale, causal=False)
    inactive = torch.arange(q.shape[1], device=q.device) % 5 == 0
    with torch.no_grad():
        lse[inactive] = float("-inf")
        q[:, inactive] = float("nan")
        dout[:, inactive] = float("nan")
        out[:, inactive] = float("nan")
        if singleton:
            q.fill_(float("nan"))
            k.fill_(float("nan"))
            v.fill_(float("nan"))
    rounded = _qsa_mma_rounded_backward(q, k, v, dout, out, lse, indices, cu, cache, scale=case.scale)
    grads = _qsa_native_backward(dout, q, k, v, out, lse, indices, cu, 3 * repeats, scale=case.scale)
    valid = _qsa_backward_valid(indices, cu, cache, False)
    _qsa_check_gradients(grads, rounded, rounded, (q, k, v), indices, valid, dtype,
                         case.name, mma_rounding=True)
    assert torch.count_nonzero(grads[0][:, inactive]).item() == 0


@torch.no_grad()
def _qsa_unaligned_copy(x, offset, padding, flat_pages=False):
    strides = [1]
    for size in reversed(x.shape[1:]):
        strides.insert(0, (size * strides[0] + 2 * padding + 1) | 1)
    if flat_pages:
        assert x.ndim == 4
        strides[0] = x.shape[1] * strides[1]
    storage = torch.empty(offset + sum((n - 1) * s for n, s in zip(x.shape, strides)) + 1,
                          device=x.device, dtype=x.dtype)
    view = storage.as_strided(x.shape, strides, storage_offset=offset)
    view.copy_(x)
    assert 1 <= offset <= 7 and storage.data_ptr() % 16 == 0
    assert view.storage_offset() == offset and view.data_ptr() == storage.data_ptr() + offset * x.element_size()
    assert view.data_ptr() % 16 == offset * x.element_size() and view.stride(-1) == 1
    assert all(s % 2 == 1 for s in view.stride()[1 if flat_pages else 0:-1])
    addresses = [view.data_ptr()]
    for size, stride in zip(view.shape[:-1], view.stride()[:-1]):
        addresses = [address + i * stride * view.element_size() for address in addresses for i in range(size)]
    addresses.sort()
    assert all(b - a >= x.shape[-1] * x.element_size() for a, b in zip(addresses, addresses[1:]))
    assert 0 in {address % 16 for address in addresses} and any(address % 16 for address in addresses)
    return view.detach().requires_grad_(x.requires_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("layout", ["v", "k", "kv"], ids=["aligned-k", "aligned-v", "both-unaligned"])
@pytest.mark.parametrize("group,offset", [(1, 1), (3, 2), (16, 3), (17, 4), (32, 5), (3, 6), (17, 7)])
@pytest.mark.parametrize("page_layout", ["none", "k", "v", "kv"], ids=["padded-pages", "flat-k-pages", "flat-v-pages", "flat-kv-pages"])
def test_qsa_backward_unaligned_storage(dtype, layout, group, offset, page_layout):
    interface = _qsa_training_interface()
    case = QsaBackwardCase(f"unaligned-g{group}-offset{offset}-{layout}-pages-{page_layout}", group, 2, 1537,
                           causal=True, cache=(129, 0, 1537), scale=0.13)
    q, k, v, dout, indices, cu, cache = _qsa_coordinate_backward_inputs(case, dtype)
    q = q.repeat_interleave(16, dim=0).detach().requires_grad_()
    dout = dout.repeat_interleave(16, dim=0)
    indices = indices.repeat_interleave(16, dim=0)
    cu = cu * 16
    pool = torch.tensor([0, 1, 14, 15, 16, 17, 30, 31, 32, 47, 48, 62, 63], device=q.device)
    indices.copy_(pool[indices.long() % len(pool)])
    indices[:, :, :2], indices[:, :, 2] = 63, 0
    assert all(x.is_contiguous() for x in (q, k, v, dout))
    with torch.no_grad():
        out, lse, *_ = interface.flash_attn_with_kvcache(
            *(x.detach().clone(memory_format=torch.contiguous_format) for x in (q, k, v)),
            page_table=indices, cu_seqlens_q=cu, max_seqlen_q=48, cache_seqlens=cache,
            softmax_scale=case.scale, causal=case.causal, num_splits=1, pack_gqa=True,
            return_softmax_lse=True)
    q, dout, out = [_qsa_unaligned_copy(x, (offset + shift - 1) % 7 + 1, padding)
                   for x, shift, padding in ((q, 0, 1), (dout, 1, 2), (out, 4, 5))]
    k, v = [_qsa_unaligned_copy(x, (offset + shift - 1) % 7 + 1, padding, flat_pages=name in page_layout)
            if name in layout else x for x, name, shift, padding in ((k, "k", 2, 3), (v, "v", 3, 4))]
    assert len({x.stride() for x in (q, dout, out)}) == 3 and k.stride() != v.stride()
    for name, x in (("k", k), ("v", v)):
        assert (x.stride(0) == x.shape[1] * x.stride(1)) == (name not in layout or name in page_layout)
        row_alignment = {(x.data_ptr() + (page * x.stride(0) + row * x.stride(1)) * x.element_size()) % 16
                         for page in range(x.shape[0]) for row in range(x.shape[1])}
        assert 0 in row_alignment and bool(row_alignment - {0}) == (name in layout)
        if name not in layout:
            assert x.data_ptr() % 16 == 0 and all(s * x.element_size() % 16 == 0 for s in x.stride()[:-1])
    ref, ref_lse, valid, oracle = _qsa_reference_gradients(q, k, v, dout, indices, cu, cache, case, torch.float32)
    pt, pt_lse, _, baseline = _qsa_reference_gradients(q, k, v, dout, indices, cu, cache, case, dtype)
    _qsa_check_error(out, ref, pt, dtype, f"{case.name}/out")
    _qsa_check_error(lse, ref_lse, pt_lse, dtype, f"{case.name}/natural-log-LSE")
    rounded = _qsa_mma_rounded_backward(q, k, v, dout, out, lse, indices, cu, cache,
                                         scale=case.scale, causal=case.causal)
    grads = _qsa_native_backward(dout, q, k, v, out, lse, indices, cu, 48,
                                 cache=cache, scale=case.scale, causal=case.causal)
    _qsa_check_gradients(grads, rounded, rounded, (q, k, v), indices, valid,
                         dtype, f"{case.name}/saved-MMA", mma_rounding=True)
    _qsa_check_gradients(grads, oracle, baseline, (q, k, v), indices, valid, dtype, case.name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("group", [3, 17], ids=["m16-g3", "m32-g17"])
@pytest.mark.parametrize("topk", [255, 256, 257, 511, 512, 513, 1279, 1280, 1281, 1535, 1536, 1537])
def test_qsa_backward_mma_tile_group_boundaries(dtype, group, topk):
    case = QsaBackwardCase(f"mma-group4-g{group}-topk{topk}", group, 2, topk,
                           causal=True, cache=(129, 0, topk), scale=0.13)
    test_qsa_backward_mma_tile_boundaries(case, dtype, repeats=32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA requires a CUDA-compatible device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_qsa_backward_mma_group_grid_stride(dtype):
    total_q, heads, topk = 8193, 17, 2048
    assert total_q * ((topk + 255) // 256) > 65535
    q = torch.zeros(total_q, heads, 256, device="cuda", dtype=dtype)
    dout, out = torch.ones_like(q), torch.zeros_like(q)
    k = torch.zeros(4, 16, 1, 256, device=q.device, dtype=dtype)
    v = torch.zeros_like(k)
    indices = torch.zeros(total_q, 1, topk, device=q.device, dtype=torch.int32)
    cu = torch.tensor([0, 0, total_q], device=q.device, dtype=torch.int32)
    cache = torch.tensor([0, 1], device=q.device, dtype=torch.int32)
    lse = torch.full((heads, total_q), float("-inf"), device=q.device)
    lse[:, -1] = 0
    grads = _qsa_native_backward(dout, q, k, v, out, lse, indices, cu, total_q,
                                 cache=cache, causal=True)
    assert torch.count_nonzero(grads[0]).item() == 0
    assert torch.count_nonzero(grads[1]).item() == 0
    expected_dv = torch.zeros_like(v)
    expected_dv[0, 0] = heads
    assert torch.equal(grads[2], expected_dv)


if __name__ == "__main__":
    sys.exit(main())
