"""Run all QSA configurations found in FA3_QSA against a per-query-token topk reference.

Shapes are ported from FA3_QSA:
  - main   @3ac92fb  hopper/test_flash_attn.py               (960, 1024) topk=128 ps=544 b=1  h=16 hk=2
  - release          hopper/test_flash_attn_decode_perf.py   (4, 8192)   topk=2048 ps=544 b=128 h=16 hk=2
  - release          hopper/test_flash_attn_prefill_perf.py  (16384,128K) topk=2048 ps=544 b=1  h=16 hk=1
  - decode-dev       env-parameterized decode                (4, 8192)   topk=2048 ps=544 b=128 h=32 hk=1
  - history (1aee7f1)                                        topk=64 ps=128

The FA3_QSA perf tests compare against a dense paged-KV reference, which is meaningless for
random topk lists (they rely on PERF_ONLY to skip the check). Here the reference gathers the
per-query-token topk list, which is valid for both structured and random lists.
"""
import argparse
import math
import sys
import torch
from einops import rearrange

from padding import pad_input, unpad_input
from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata


# QSA takes the AIU paged-KV load path only when the caller passes qsa_allow_aiu=True,
# certifying that every 16-column-aligned group of the top-k list is 16 pool-contiguous
# tokens (the bulk load reads one table entry per group and fetches 16 contiguous tokens
# from it, ignoring the other 15 entries). The default False routes QSA through the
# per-column load, which honors each entry individually, so every list shape -- random
# included -- must match the reference at any page size. qsa_allow_aiu=True cases pin the
# contract from both sides: aligned 16-token runs match, a scattered list diverges.
# QSA is only instantiated for head dim 256, so other head dims must be rejected.
#
# expect: "ok" output must match the reference, "mismatch" the output must diverge as
# documented, "reject" the call must raise.
CASES = [
    # name, seqlen_q, seqlen_k, d, topk, page_size, batch, nheads, nheads_k, topk_mode, expect, allow_aiu
    ("fa3_precision",        960,   1024,   256, 128,  544, 1,   16, 2, "structured", "ok"),
    ("fa3_precision_rand",   960,   1024,   256, 128,  544, 1,   16, 2, "random",     "ok"),
    ("probe_rand_noaiu",     960,   1024,   256, 128,  136, 1,   16, 2, "random",     "ok"),
    ("probe_block16",        960,   1024,   256, 128,  544, 1,   16, 2, "block16",    "ok"),
    ("hist_topk64",          2,     1024,   256, 64,   128, 1,   16, 2, "structured", "ok"),
    ("hist_topk64_block64",  2,     1024,   256, 64,   128, 1,   16, 2, "block64",    "ok"),
    ("fa3_decode",           4,     8192,   256, 2048, 544, 128, 16, 2, "block16",    "ok"),
    ("fa3_decode_h32_mqa",   4,     8192,   256, 2048, 544, 128, 32, 1, "block16",    "ok"),
    ("fa3_decode_d128",      4,     8192,   128, 2048, 544, 128, 16, 2, "block16",    "reject"),
    ("fa3_prefill",          16384, 131072, 256, 2048, 544, 1,   16, 1, "block16",    "ok"),
    ("sq1_decode",           1,     1024,   256, 128,  544, 4,   16, 2, "structured", "ok"),
    ("sq1_decode_h32_mqa",   1,     1024,   256, 128,  544, 4,   32, 1, "structured", "ok"),
    # sq=1 with g<=16 used to sit on the AIU path as well; off AIU, a random list is as
    # valid as an aligned block16 list.
    ("sq1_rand",             1,     8192,   256, 128,  544, 4,   16, 2, "random",     "ok"),
    ("sq1_block16",          1,     8192,   256, 128,  544, 4,   16, 2, "block16",    "ok"),
    ("sq1_g64_reject",       1,     1024,   256, 128,  544, 2,   64, 1, "structured", "reject"),
    # topk not a multiple of 16: the rows past topk must be masked by seqlen_k.
    ("tail_topk100",         960,   1024,   256, 100,  544, 1,   16, 2, "block16tail", "ok"),
    ("tail_topk100_ps576",   960,   1024,   256, 100,  576, 1,   16, 2, "block64tail", "ok"),
    ("tail_topk100_sq1_g32", 1,     8192,   256, 100,  544, 4,   32, 1, "block16tail", "ok"),
    # qsa_allow_aiu=True pins the bulk-load contract from both sides.
    ("run16_allow_aiu",      960,   1024,   256, 128,  544, 1,   16, 2, "block16",    "ok",       True),
    ("rand_allow_aiu",       960,   1024,   256, 128,  544, 1,   16, 2, "random",     "mismatch", True),
]


def build_kvcache(seqlen_k, page_size, batch_size, nheads_k, d, device, dtype):
    """Identity page table: logical token t of batch b lives at page_table[b, t//ps]*ps + t%ps."""
    nblocks_per_b = math.ceil(seqlen_k / page_size) * 3
    num_blocks = nblocks_per_b * batch_size
    k_cache_paged = torch.randn(num_blocks, page_size, nheads_k, d, device=device, dtype=dtype)
    v_cache_paged = torch.randn(num_blocks, page_size, nheads_k, d, device=device, dtype=dtype)
    page_table = rearrange(
        torch.arange(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks", b=batch_size,
    )
    k_cache = rearrange(k_cache_paged[page_table.flatten()],
                        "(b nblocks) ps ... -> b (nblocks ps) ...", b=batch_size)[:, :seqlen_k]
    v_cache = rearrange(v_cache_paged[page_table.flatten()],
                        "(b nblocks) ps ... -> b (nblocks ps) ...", b=batch_size)[:, :seqlen_k]
    return k_cache, v_cache, page_table, k_cache_paged, v_cache_paged


def build_topk(mode, topk, seqlen_k, page_size, page_table, cu_seqlens_q, batch_size, device):
    """Returns (logical, paged): logical indices into k_cache[b], paged indices for the kernel."""
    logical_rows, paged_rows = [], []
    for b in range(batch_size):
        n_q = int(cu_seqlens_q[b + 1] - cu_seqlens_q[b])
        if mode == "structured":
            t = torch.arange(topk, device=device, dtype=torch.int64).unsqueeze(0).expand(n_q, -1)
        elif mode == "random":
            t = torch.randint(0, seqlen_k, (n_q, topk), device=device, dtype=torch.int64)
        elif mode.endswith("tail"):
            # topk that is NOT a multiple of the AIU run length: aligned contiguous runs plus a
            # short final run, so the last n_block has valid rows only up to topk.
            run = int(mode[len("block"):-len("tail")])
            assert topk % run != 0 and seqlen_k % run == 0
            starts = torch.randint(0, seqlen_k // run, (n_q, topk // run + 1),
                                   device=device, dtype=torch.int64) * run
            t = (starts.unsqueeze(-1) + torch.arange(run, device=device, dtype=torch.int64)
                 ).reshape(n_q, -1)[:, :topk]
        elif mode.startswith("block"):
            run = int(mode[len("block"):])
            assert topk % run == 0 and seqlen_k % run == 0
            starts = torch.randint(0, seqlen_k // run, (n_q, topk // run),
                                   device=device, dtype=torch.int64) * run
            t = (starts.unsqueeze(-1) + torch.arange(run, device=device, dtype=torch.int64)).reshape(n_q, topk)
        else:
            raise ValueError(f"unknown topk mode {mode}")
        pages = page_table[b].to(torch.int64)[t // page_size]
        logical_rows.append(t)
        paged_rows.append((pages * page_size + t % page_size).to(torch.int32))
    return torch.cat(logical_rows, dim=0), torch.cat(paged_rows, dim=0)


def reference(q_unpad, k_cache, v_cache, topk_logical, cu_seqlens_q, batch_size, nheads_k, out_dtype):
    """Per-query-token topk gather attention. Returns (fp32-accurate ref, bf16 pytorch impl)."""
    total_q, nheads, d = q_unpad.shape
    g = nheads // nheads_k
    scale = 1.0 / math.sqrt(d)
    topk = topk_logical.shape[1]
    out_ref = torch.empty_like(q_unpad)
    out_pt = torch.empty_like(q_unpad)
    # keep the gathered fp32 K/V tile under ~256MB
    chunk = max(1, min(total_q, (256 << 20) // (topk * nheads_k * d * 4)))
    for b in range(batch_size):
        lo, hi = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        for s in range(lo, hi, chunk):
            e = min(s + chunk, hi)
            idx = topk_logical[s:e]
            k_g = k_cache[b][idx]                       # (c, topk, hk, d)
            v_g = v_cache[b][idx]
            q_c = q_unpad[s:e].view(e - s, nheads_k, g, d)
            for tag, cast in (("ref", torch.float32), ("pt", None)):
                qq = q_c.to(cast) if cast else q_c
                kk = k_g.to(cast) if cast else k_g
                vv = v_g.to(cast) if cast else v_g
                scores = torch.einsum("c k g d, c t k d -> c k g t", qq, kk) * scale
                attn = torch.softmax(scores, dim=-1)
                o = torch.einsum("c k g t, c t k d -> c k g d", attn, vv)
                o = o.reshape(e - s, nheads, d).to(out_dtype)
                (out_ref if tag == "ref" else out_pt)[s:e] = o
    return out_ref, out_pt


NO_METADATA = False
VARLEN = False


def run_case(name, seqlen_q, seqlen_k, d, topk, page_size, batch_size, nheads, nheads_k, mode,
             expect="ok", allow_aiu=False):
    torch.random.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    print(f"=== {name}: sq={seqlen_q} sk={seqlen_k} d={d} topk={topk} ps={page_size} "
          f"b={batch_size} h={nheads} hk={nheads_k} topk_mode={mode} "
          f"allow_aiu={allow_aiu} ===", flush=True)

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    if VARLEN and seqlen_q > 1:
        lengths = torch.full((batch_size, 1), seqlen_q, device=device, dtype=torch.int32)
        lengths[0] = max(1, seqlen_q - 1)  # make lengths non-uniform
        query_padding_mask = torch.arange(seqlen_q, device=device).unsqueeze(0) < lengths
    else:
        query_padding_mask = torch.ones(batch_size, seqlen_q, dtype=torch.bool, device=device)
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, *_ = unpad_input(q, query_padding_mask)
    total_q = q_unpad.shape[0]
    print(f"  total_q={total_q} (kernel detects QSA via the 3-D page_table), "
          f"metadata={'none' if NO_METADATA else 'precomputed'}, "
          f"lengths={'non-uniform' if VARLEN and seqlen_q > 1 else 'uniform'}", flush=True)

    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged = \
        build_kvcache(seqlen_k, page_size, batch_size, nheads_k, d, device, dtype)
    cache_seqlens = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)
    topk_logical, topk_paged = build_topk(mode, topk, seqlen_k, page_size, page_table,
                                          cu_seqlens_q, batch_size, device)

    out_ref, out_pt = reference(q_unpad, k_cache, v_cache, topk_logical, cu_seqlens_q,
                                batch_size, nheads_k, dtype)
    del k_cache, v_cache
    torch.cuda.empty_cache()

    scheduler_metadata = None if NO_METADATA else get_scheduler_metadata(
        batch_size, max_seqlen_q, page_table.shape[1] * page_size,
        nheads, nheads_k, d, cache_seqlens, q.dtype, headdim_v=d,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k_new=None, cache_leftpad=None,
        max_seqlen_k_new=seqlen_q, page_size=page_size,
        causal=True, window_size=(-1, -1), attention_chunk=0, num_splits=1,
    )
    out, lse, *_ = flash_attn_with_kvcache(
        q_unpad, k_cache_paged, v_cache_paged, None, None,
        qv=None, rotary_cos=None, rotary_sin=None,
        cache_seqlens=cache_seqlens, page_table=topk_paged.unsqueeze(1),
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k_new=None, max_seqlen_q=max_seqlen_q,
        causal=True, window_size=(-1, -1), scheduler_metadata=scheduler_metadata,
        num_splits=1, pack_gqa=True, return_softmax_lse=True,
        qsa_allow_aiu=allow_aiu,
    )
    torch.cuda.synchronize()

    max_diff = (out - out_ref).abs().max().item()
    mean_diff = (out - out_ref).abs().mean().item()
    pt_max_diff = (out_pt - out_ref).abs().max().item()
    pt_mean_diff = (out_pt - out_ref).abs().mean().item()
    print(f"  kernel  max/mean diff: {max_diff:.6e} / {mean_diff:.6e}", flush=True)
    print(f"  pytorch max/mean diff: {pt_max_diff:.6e} / {pt_mean_diff:.6e}", flush=True)

    matches = max_diff <= 2 * pt_max_diff + 1e-5 and mean_diff <= 2 * pt_mean_diff + 1e-6
    if expect == "reject":
        print("  FAILED (expected the call to be rejected)", flush=True)
        return False
    if expect == "mismatch":
        # Pins the AIU bulk-load contract: a scattered list riding the AIU path
        # (qsa_allow_aiu=True) must diverge.
        print(f"  {'PASSED (mismatch as documented)' if not matches else 'FAILED (expected a mismatch)'}",
              flush=True)
        return not matches
    print(f"  {'PASSED' if matches else 'FAILED'}", flush=True)
    return matches


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=str, default=None, help="case name or index; default runs all")
    p.add_argument("--params", type=str, default=None,
                   help="ad-hoc config: sq,sk,d,topk,ps,b,h,hk,mode")
    p.add_argument("--no-metadata", action="store_true",
                   help="pass scheduler_metadata=None instead of precomputing it")
    p.add_argument("--varlen", action="store_true",
                   help="use non-uniform query lengths so flash_api sets params.is_varlen_q")
    args = p.parse_args()
    NO_METADATA = args.no_metadata
    VARLEN = args.varlen
    if args.params is not None:
        f = args.params.split(",")
        cases = [("adhoc",) + tuple(int(x) for x in f[:8]) + (f[8],)]
    elif args.case is not None:
        if args.case.isdigit():
            cases = [CASES[int(args.case)]]
        else:
            cases = [c for c in CASES if c[0] == args.case]
            if not cases:
                sys.exit(f"unknown case {args.case}")
    else:
        cases = CASES
    results = []
    for c in cases:
        expect = c[10] if len(c) > 10 else "ok"
        try:
            results.append((c[0], run_case(*c)))
        except Exception as e:
            rejected = expect == "reject"
            print(f"  {'PASSED (rejected as expected)' if rejected else 'ERROR'}: "
                  f"{type(e).__name__}: {e}", flush=True)
            results.append((c[0], rejected))
        finally:
            torch.cuda.empty_cache()
        print(flush=True)
    print("=== summary ===", flush=True)
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if all(ok for _, ok in results) else 1)
