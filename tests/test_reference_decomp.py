"""Reference (golden) decomposer: a straight CPU loop over rows.

The production path in flex_flash_attention/mask_decomp.py is a GPU pipeline (fused
row stats + run-length segmentation on device, only segment boundaries come
back to the host).  This module keeps the original, obviously-correct CPU
implementation so the suite can assert the two agree element for element; it is
NOT a supported runtime path and is deliberately absent from the module's
public surface.
"""

import os
import sys

import torch
from typing import List

# flex_flash_attention is imported via the package name, so hopper/ must lead
# sys.path (an older installed copy in site-packages may shadow it).
_HOPPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOPPER not in sys.path:
    sys.path.insert(0, _HOPPER)
elif sys.path.index(_HOPPER) != 0:
    sys.path.remove(_HOPPER)
    sys.path.insert(0, _HOPPER)

from flex_flash_attention.mask_decomp import (SliceInfo, _reject_holes,          # noqa: E402
                                        SLICE_FULL, SLICE_CAUSAL,
                                        SLICE_INVCAUSAL, SLICE_BICAUSAL)


def row_stats_eager(mask: torch.Tensor):
    """Legacy eager row stats (reference for the fused GPU sweep)."""
    seqlen_q, seqlen_k = mask.shape
    m = mask
    row_any = m.any(dim=1)
    row_sum = m.sum(dim=1)
    ks = m.to(torch.uint8).argmax(dim=1)
    ke = seqlen_k - m.flip(dims=[1]).to(torch.uint8).argmax(dim=1)
    ks = torch.where(row_any, ks, torch.zeros_like(ks))
    ke = torch.where(row_any, ke, torch.zeros_like(ke))
    has_holes = row_any & (row_sum != (ke - ks))
    return (ks.cpu().tolist(), ke.cpu().tolist(), has_holes.cpu().tolist())


def decompose_mask_legacy(mask: torch.Tensor,
                          allow_holes: bool = False) -> List[SliceInfo]:
    """Legacy CPU-loop decomposer (reference implementation; kept for
    equivalence testing against the GPU pipeline)."""
    seqlen_q, seqlen_k = mask.shape
    assert mask.dtype == torch.bool, "mask must be boolean"

    if seqlen_q == 0 or seqlen_k == 0:
        return []

    # Pass 1: per-row bounds + hole check (vectorized)
    row_k_start, row_k_end, row_has_holes = row_stats_eager(mask)
    if not allow_holes and any(row_has_holes):
        _reject_holes(torch.tensor(row_has_holes), seqlen_q)
    row_any = [ks != 0 or ke != 0 for ks, ke in zip(row_k_start, row_k_end)]

    # Pass 2: group rows into slices with cross-row pattern detection
    slices = []
    q_start = 0

    while q_start < seqlen_q:
        # Handle empty rows (k_start == k_end == 0 and all False)
        if not row_any[q_start]:
            # Find contiguous block of empty rows
            q_end = q_start + 1
            while q_end < seqlen_q and not row_any[q_end]:
                q_end += 1
            slices.append(SliceInfo(q_start, q_end, 0, 0, SLICE_FULL, 0, 0))
            q_start = q_end
            continue

        ks0 = row_k_start[q_start]
        ke0 = row_k_end[q_start]
        has_holes0 = row_has_holes[q_start]

        # Try to extend the slice as far as possible
        q_end = q_start + 1

        # Detect pattern from first two rows (if available)
        if q_end < seqlen_q:
            ks1 = row_k_start[q_end]
            ke1 = row_k_end[q_end]
            d_ks = ks1 - ks0  # per-row delta of k_start
            d_ke = ke1 - ke0  # per-row delta of k_end
        else:
            d_ks = 0
            d_ke = 0

        # Extend while the pattern holds
        while q_end < seqlen_q:
            ks_e = row_k_start[q_end]
            ke_e = row_k_end[q_end]
            expected_ks = ks0 + d_ks * (q_end - q_start)
            expected_ke = ke0 + d_ke * (q_end - q_start)
            if ks_e != expected_ks or ke_e != expected_ke:
                break
            # Also check hole pattern is consistent
            q_end += 1

        # Determine mask type from the deltas
        if d_ks == 0 and d_ke == 0:
            # Constant K range → FULL.  Rows reaching here are hole-free (the
            # contiguity gate rejected the rest), so the rectangle is exact.
            if not has_holes0:
                mask_type = SLICE_FULL
            else:
                # Only reachable via allow_holes=True: the rectangle
                # OVER-COVERS the row.  The kernel applies geometric slice
                # masks only — it does NOT mask per element — so these slices
                # must not be used to compute attention.
                mask_type = SLICE_FULL
            diag_offset = 0
            band_width = 0
            k_start = ks0
            k_end = ke0
        elif d_ks == 0 and d_ke == 1:
            # k_end grows with q → CAUSAL (lower triangle)
            # Diagonal: k <= q + offset, where offset = ke0 - q_start - 1
            mask_type = SLICE_CAUSAL
            diag_offset = ke0 - 1 - q_start
            band_width = 0
            k_start = ks0
            k_end = ke0 + (q_end - q_start - 1)  # last row's k_end
        elif d_ks == 1 and d_ke == 0:
            # k_start grows with q → INVCAUSAL (upper triangle)
            mask_type = SLICE_INVCAUSAL
            diag_offset = ks0 - q_start
            band_width = 0
            k_start = ks0
            k_end = ke0
        elif d_ks == 1 and d_ke == 1:
            # Both grow → BICAUSAL (band); right edge pinned at the diagonal
            # (see _classify_segments for the exact-representation rationale).
            mask_type = SLICE_BICAUSAL
            diag_offset = ke0 - 1 - q_start
            band_width = ke0 - 1 - ks0
            k_start = ks0
            k_end = ke0 + (q_end - q_start - 1)
        else:
            # Unknown pattern — treat single row as FULL
            q_end = q_start + 1
            mask_type = SLICE_FULL
            diag_offset = 0
            band_width = 0
            k_start = ks0
            k_end = ke0

        slices.append(SliceInfo(q_start, q_end, k_start, k_end, mask_type,
                                diag_offset, band_width))
        q_start = q_end

    return slices
