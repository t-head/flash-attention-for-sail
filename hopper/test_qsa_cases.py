"""QSA forward correctness and optional timing, with automatic SplitKV.

Run pytest hopper/test_qsa_cases.py, or use --list / --case / --benchmark directly.
FA3_QSA source configurations are recorded below; no archived test modules or
subprocess adapters are needed. The shared sparse reference checks every output.
Input generation is consolidated, so timings are not historical A/B samples.
"""

import argparse
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


def sparse_attention_ref(q, k, v, indices, cu, cache, *, source=False):
    """Gather per query, mask sparse columns, and compare FP32 with low precision."""
    topk = indices.shape[-1]
    heads, dim = q.shape[-2:]
    kv_heads = k.shape[-2]
    group = heads // kv_heads
    k, v = k.flatten(0, 1), v.flatten(0, 1)
    ref, pt = torch.empty_like(q), torch.empty_like(q)
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
                weights = torch.softmax(scores, dim=-1).masked_fill(~valid, 0.0)
                out = torch.einsum("chgt,cthd->chgd", weights, v_)
                target[start:end] = out.reshape(end - start, heads, dim)
    return ref, pt


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
                causal=True, num_splits=1, pack_gqa=True):
    """Call the C++ fwd op directly, bypassing flash_attn_with_kvcache."""
    import flash_attn_3._C
    scale = q.shape[-1] ** -0.5
    return torch.ops.flash_attn_3.fwd(
        q, k, v,
        None, None, None, None,                    # k_new, v_new, qv, out
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PPU")
@pytest.mark.parametrize("heads,kv_heads", [(6, 1), (24, 2)])
@pytest.mark.parametrize("width", [1, 17, 65, 2051])
@pytest.mark.parametrize("splits", [0, 1, 4, 32, 128])
def test_fp8_single_query_qsa(heads, kv_heads, width, splits):
    from flash_attn_interface import flash_attn_with_kvcache

    q, k, v, slots, pages, counts, cu, descales = _fp8_qsa_inputs(heads, kv_heads, width)
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("FP8 QSA requires PPU0015")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PPU")
@pytest.mark.parametrize("rows,heads,kv_heads", [
    (1, 6, 1), (2, 6, 1), (8, 6, 1), (9, 6, 1),
    (639, 6, 1), (640, 6, 1), (780, 6, 1), (781, 6, 1),
    (896, 6, 1), (936, 6, 1), (937, 6, 1), (1, 24, 2), (4096, 24, 2),
])
def test_fp8_qsa_auto_boundaries(rows, heads, kv_heads):
    from flash_attn_interface import flash_attn_with_kvcache

    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("FP8 QSA requires PPU0015")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PPU")
@pytest.mark.parametrize("heads,kv_heads", [(6, 1), (24, 2)])
@pytest.mark.parametrize("splits", [0, 1, 4, 128])
def test_fp8_qsa_preserves_constant_value(heads, kv_heads, splits):
    from flash_attn_interface import flash_attn_with_kvcache

    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("FP8 QSA requires PPU0015")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PPU")
def test_fp8_qsa_rejects_unsupported_features():
    from flash_attn_interface import flash_attn_with_kvcache

    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("FP8 QSA requires PPU0015")
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

    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("FP8 QSA requires PPU0015")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PPU")
@pytest.mark.parametrize("group", range(1, 33))
@pytest.mark.parametrize("kv_heads", [1, 3])
@pytest.mark.parametrize("splits", [0, 1, 4])
def test_fp8_qsa_group_range(group, kv_heads, splits):
    _run_fp8_general_qsa(group, kv_heads, [1, 1, 1], 16, 65, splits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PPU")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PPU")
@pytest.mark.parametrize("batch,sq,heads", [(65536, 1, 1), (1, 16384, 32)])
@pytest.mark.parametrize("splits", [0, 1])
def test_fp8_qsa_large_axes(batch, sq, heads, splits):
    from flash_attn_interface import flash_attn_with_kvcache

    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("FP8 QSA requires PPU0015")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PPU")
@pytest.mark.parametrize("heads,kv_heads,message", [(33, 1, "at most 32"), (7, 2, "must divide")])
def test_fp8_qsa_rejects_invalid_head_counts(heads, kv_heads, message):
    from flash_attn_interface import flash_attn_with_kvcache

    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("FP8 QSA requires PPU0015")
    q, k, v, _, pages, counts, cu, descales = _fp8_qsa_inputs(heads, kv_heads, 65)
    with pytest.raises(RuntimeError, match=message):
        flash_attn_with_kvcache(q, k, v, cache_seqlens=counts, cu_seqlens_q=cu,
            max_seqlen_q=1, page_table=pages, causal=True, **descales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires PPU")
def test_fp8_qsa_rejects_unaligned_storage():
    from flash_attn_interface import flash_attn_with_kvcache

    q, k, v, _, pages, counts, cu, descales = _fp8_qsa_inputs(6, 1, 65)
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("FP8 QSA requires PPU0015")
    bad_q = torch.empty(q.numel() + 1, dtype=q.dtype, device=q.device)[1:].view(q.shape)
    backing = torch.empty(k.numel() + 1, dtype=k.dtype, device=k.device)
    bad_k = backing.as_strided(k.shape, (k.stride(0) + 1, *k.stride()[1:]))
    for qq, kk in ((bad_q, k), (q, bad_k)):
        with pytest.raises(RuntimeError, match="16-byte"):
            flash_attn_with_kvcache(qq, kk, v, cache_seqlens=counts, cu_seqlens_q=cu,
                max_seqlen_q=1, page_table=pages, causal=True, **descales)


if __name__ == "__main__":
    sys.exit(main())
