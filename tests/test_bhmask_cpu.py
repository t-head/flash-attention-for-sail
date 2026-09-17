"""CPU-only unit test for P3 layout construction (no GPU kernel needed).

Validates build_bh_layout flat-concatenation semantics (offsets, prefix
sums, mask_bits zero-fill) and _group_bh_masks content dedup.  Runs while
the wheel rebuild occupies the compile farm; touches no CUDA memory.
"""
import os
import sys

import torch

# flex_flash_attention is imported via the package name, so hopper/ must lead
# sys.path (an older installed copy in site-packages may shadow it).
_HOPPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOPPER not in sys.path:
    sys.path.insert(0, _HOPPER)
elif sys.path.index(_HOPPER) != 0:
    sys.path.remove(_HOPPER)
    sys.path.insert(0, _HOPPER)

from flex_flash_attention.mask_decomp import (          # noqa: E402
    build_merge_layout, build_bh_layout, decompose_mask_optimized,
    pack_mask_bits, SliceInfo, SLICE_BITMASK,
)
from flex_flash_attention import interface as itf       # noqa: E402

CPU = torch.device('cpu')
Sq, Sk = 197, 233


def one_layout(mask):
    slices = decompose_mask_optimized(mask, kBlockM=128, tile_align=True)
    t = build_merge_layout(slices, Sq, kBlockM=128, device=CPU)
    t['mask_bits'] = None
    return t


def main():
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(("  PASS: " if cond else "  FAIL: ") + name)
        ok = ok and bool(cond)

    # two algebraic masks + one bitmask-fallback mask
    full = torch.ones(Sq, Sk, dtype=torch.bool)
    causal = torch.tril(torch.ones(Sq, Sk, dtype=torch.bool))
    torch.manual_seed(3)
    rand_m = torch.rand(Sq, Sk) < 0.45
    # random masks hit the BITMASK fallback in the interface; build that
    # layout contract directly:
    bits_t = build_merge_layout(
        [SliceInfo(0, Sq, 0, Sk, SLICE_BITMASK, 0, 0)], Sq,
        kBlockM=128, device=CPU)
    bits_t['mask_bits'] = pack_mask_bits(rand_m)

    gls = [one_layout(full), one_layout(causal), bits_t]
    flat = build_bh_layout(gls, Sq, CPU)

    n0 = gls[0]['q_starts'].shape[0]
    n1 = gls[1]['q_starts'].shape[0]
    n2 = gls[2]['q_starts'].shape[0]
    chk("group_slice_offsets prefix sum",
        torch.equal(flat['group_slice_offsets'],
                    torch.tensor([0, n0, n0 + n1, n0 + n1 + n2],
                                 dtype=torch.int32)))
    v0 = gls[0]['vbatch_to_slice'].shape[0]
    v1 = gls[1]['vbatch_to_slice'].shape[0]
    v2 = gls[2]['vbatch_to_slice'].shape[0]
    chk("group_vb_offsets prefix sum",
        torch.equal(flat['group_vb_offsets'],
                    torch.tensor([0, v0, v0 + v1, v0 + v1 + v2],
                                 dtype=torch.int32)))
    chk("flat slice arrays concat",
        flat['q_starts'].shape[0] == n0 + n1 + n2
        and torch.equal(flat['q_starts'][:n0], gls[0]['q_starts'])
        and torch.equal(flat['q_starts'][n0:n0 + n1], gls[1]['q_starts']))
    chk("row tables flat length 3*Sq",
        flat['row_to_slice'].shape[0] == 3 * Sq
        and flat['row_to_vbatch_start'].shape[0] == 3 * Sq)
    # slice-index rebase: group-1 row entries shifted by n0 (negatives kept)
    r1 = gls[1]['row_to_slice']
    expect = torch.where(r1 < 0, r1, r1 + n0)
    chk("row_to_slice rebased group 1",
        torch.equal(flat['row_to_slice'][Sq:2 * Sq], expect))
    # vbatch rebase
    e_vb = gls[1]['vbatch_to_slice'] + n0
    chk("vbatch_to_slice rebased group 1",
        torch.equal(flat['vbatch_to_slice'][v0:v0 + v1], e_vb))
    e_rs = gls[2]['row_to_vbatch_start'] + v0 + v1
    chk("row_to_vbatch_start rebased group 2",
        torch.equal(flat['row_to_vbatch_start'][2 * Sq:], e_rs))
    # mask_bits: zero rows for algebraic groups, real bits for group 2
    mb = flat['mask_bits']
    stride = bits_t['mask_bits'].shape[1]
    chk("mask_bits shape (3*Sq, stride)", mb.shape == (3 * Sq, stride))
    chk("mask_bits zero rows for algebraic groups",
        mb[:2 * Sq].sum().item() == 0)
    chk("mask_bits group 2 content",
        torch.equal(mb[2 * Sq:], bits_t['mask_bits']))

    # no-bitmask layout set → mask_bits None
    flat2 = build_bh_layout(gls[:2], Sq, CPU)
    chk("mask_bits None when no BITMASK group", flat2['mask_bits'] is None)

    # _group_bh_masks dedup (CPU bool tensor works: reshape+sum+equal)
    mask_bh = torch.stack([
        torch.stack([full, causal, full, rand_m]),
        torch.stack([causal, full, rand_m, full]),
    ])
    uniques, assign = itf._group_bh_masks(mask_bh)
    chk("dedup: 3 unique masks", len(uniques) == 3)
    gid = {assign[0]: 'full', assign[1]: 'causal', assign[3]: 'rand'}
    expect_names = ['full', 'causal', 'full', 'rand',
                    'causal', 'full', 'rand', 'full']
    chk("dedup: assignment consistent",
        all(gid[assign[i]] == expect_names[i] for i in range(8)))

    print("BHMASK CPU RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
