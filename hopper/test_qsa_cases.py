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


if __name__ == "__main__":
    sys.exit(main())
