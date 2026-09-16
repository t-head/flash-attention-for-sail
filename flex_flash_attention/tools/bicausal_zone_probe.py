"""Read the kernel's effective per-row visible-column count directly.

With q = k = 0 every score is 0, so softmax runs over the visible set and
LSE = ln(n_visible).  Comparing exp(lse) against the count implied by the slice
geometry localises the wrong rows exactly, with no reference-noise ambiguity.

This is how the BICAUSAL no-mask-zone defect was pinned down: a hand-built
slice whose band left edge goes negative reported 128 visible columns on the
first m_block instead of the band width, proving mask.apply() was skipped for
the whole block.  Keep it around for any future "is the mask really applied?"
question.

    python tools/bicausal_zone_probe.py
"""
import os
import sys

import torch

# The suite next door provides the kernel loader and the layout plumbing; it
# also puts hopper/ on sys.path, which is what makes the package-qualified
# import below resolve.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from test_arb_suite import DEV, _load_kernel, _layout_args        # noqa: E402
from flex_flash_attention.mask_decomp import (SliceInfo,           # noqa: E402
                                        SLICE_BICAUSAL,
                                        build_merge_layout)

itf = _load_kernel()
s, h, d = 128, 4, 64

qi = torch.arange(s, device=DEV).unsqueeze(1)
ki = torch.arange(s, device=DEV).unsqueeze(0)

for bw in (32, 63, 64, 96, 127):
    q = torch.zeros(1, s, h, d, device=DEV, dtype=torch.bfloat16)
    k = torch.zeros(1, s, h, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, s, h, d, device=DEV, dtype=torch.bfloat16)
    layout = build_merge_layout([SliceInfo(0, s, 0, s, SLICE_BICAUSAL, 0, bw)],
                                s, kBlockM=128, device=DEV)
    out, lse = itf.flash_attn_flex_flash_precomputed(q, k, v, *_layout_args(layout))

    got = lse[0, 0].float().exp().round().long()          # per-row visible count
    want = ((ki >= qi - bw) & (ki <= qi)).sum(1)          # band(row) semantics
    bad = (got != want).nonzero().squeeze(1)
    print(f"\nbw={bw}: {bad.numel()} of {s} rows have the wrong visible count")
    if bad.numel():
        for r in bad[:6].tolist():
            print(f"   row {r:3d}: kernel={got[r].item():4d} expected={want[r].item():4d}"
                  f"  (band = [{max(0, r - bw)}, {r}])")
        if bad.numel() > 6:
            print(f"   ... rows {bad.min().item()}..{bad.max().item()}")
