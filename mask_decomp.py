"""Mask decomposition: convert arbitrary boolean mask to slice decomposition.

Given an arbitrary boolean mask of shape (seqlen_q, seqlen_k), decompose it
into non-overlapping "slices" where each slice covers a contiguous Q-row range
with a single mask type (FULL, CAUSAL, INVCAUSAL, BICAUSAL).

Masks outside the layered-interval envelope (per-row interval count over the
cap) raise MaskNotSupportedError here; the user-facing entry points fall back
to an exact element-wise bitmask kernel path for those (see pack_mask_bits
and SLICE_BITMASK).

Reference: MagiAttention csrc/flexible_flash_attention/mask.h
"""

import torch
import torch._dynamo
from typing import Tuple, NamedTuple, List
from enum import IntEnum

# torch.compile guards: masks arrive with many different shapes; never let
# dynamo silently fall back to eager past the recompile budget.
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.fail_on_recompile_limit_hit = True

SLICE_FULL = 0
SLICE_CAUSAL = 1
SLICE_INVCAUSAL = 2
SLICE_BICAUSAL = 3
SLICE_BITMASK = 4   # element-wise bitmask fallback (see pack_mask_bits)

_BIT_WEIGHTS = None


def pack_mask_bits(mask: torch.Tensor) -> torch.Tensor:
    """Pack a boolean mask into the kernel's row-major bit layout.

    Returns a uint8 tensor of shape (seqlen_q, ((seqlen_k + 127) // 128) * 16):
    bit k of row q is LSB-first in byte (q, k // 8), i.e.
        visible(q, k) == (bits[q, k >> 3] >> (k & 7)) & 1
    The row stride pads seqlen_k up to a multiple of 128 bits so the last K
    tile (kBlockN <= 128) never reads past the buffer, and the zero padding
    bits double as the seqlen_k tail mask.  This layout mirrors
    AttnSliceParams::mask_bits (attn_slice.h) exactly.
    """
    global _BIT_WEIGHTS
    seqlen_q, seqlen_k = mask.shape
    stride = ((seqlen_k + 127) // 128) * 16     # bytes per row
    m = mask.to(torch.uint8)
    pad = stride * 8 - seqlen_k
    if pad > 0:
        m = torch.nn.functional.pad(m, (0, pad))  # zero padding bits
    if _BIT_WEIGHTS is None or _BIT_WEIGHTS.device != mask.device:
        _BIT_WEIGHTS = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                                    dtype=torch.uint8, device=mask.device)
    # Disjoint bit weights: the uint8 sum cannot overflow (max 255).
    return (m.view(seqlen_q, stride, 8) * _BIT_WEIGHTS).sum(dim=-1).to(torch.uint8)


class SliceInfo(NamedTuple):
    q_start: int
    q_end: int
    k_start: int
    k_end: int
    mask_type: int
    diagonal_offset: int
    band_width: int


def _row_bounds(mask_row: torch.Tensor) -> Tuple[int, int]:
    """Get the [k_start, k_end) range of True values for a row.
    Returns (0, 0) for an all-False row.
    """
    true_indices = torch.where(mask_row)[0]
    if len(true_indices) == 0:
        return (0, 0)
    return (int(true_indices[0]), int(true_indices[-1]) + 1)


def _has_holes(mask_row: torch.Tensor, k_start: int, k_end: int) -> bool:
    """Check if a row has any False values in [k_start, k_end)."""
    return int(mask_row[k_start:k_end].sum()) < (k_end - k_start)


def _row_stats_vectorized(mask: torch.Tensor):
    """Vectorized per-row (k_start, k_end, has_holes) for the whole mask.

    Returns CPU lists row_k_start / row_k_end / row_has_holes, semantically
    identical to calling _row_bounds / _has_holes row by row.

    GPU path: the fused row-stats kernel (torch.compile) computes first/last
    True and the row count in a single sweep — ~14x faster than the legacy
    eager chain (any + sum + to(uint8) + argmax + flip + argmax launched 6-8
    memory-bound passes over the full mask; measured 42ms -> 3ms at
    s=25286).  The three arrays leave the device packed into ONE tensor
    (single D2H) and are converted via numpy (tolist on torch CPU tensors
    is ~3x slower per element).
    """
    seqlen_q, seqlen_k = mask.shape
    ks_t, ke_t, hh_t = _row_stats_gpu(mask)
    # Single D2H: (3, seqlen_q) int32 -> numpy -> python lists.
    packed = torch.stack((ks_t, ke_t, hh_t.to(torch.int32))).cpu().numpy()
    return packed[0].tolist(), packed[1].tolist(), packed[2].astype(bool).tolist()


# Fused per-row stats: first True, last True (+1), and hole detection in one
# elementwise+reduction sweep.  torch.compile collapses the where/min/max/
# sum chain into a handful of kernels (no flip copy, no uint8 argmax passes).
# This compiled version is the FALLBACK; wheels built with
# csrc/mask_decomp_kernels.cu serve the same semantics from a hand-written
# CUDA kernel (~2.7x faster at s=25286, no dynamo dependency) via
# torch.ops.flex_flash_attention.row_stats (see _row_stats_gpu).
@torch.compile
def _row_stats_compiled(m: torch.Tensor, idx: torch.Tensor):
    seqlen_k = m.shape[1]
    first = torch.where(m, idx, seqlen_k).min(dim=1).values
    last = torch.where(m, idx, -1).max(dim=1).values
    row_any = last >= 0
    ks = torch.where(row_any, first, 0)
    ke = torch.where(row_any, last + 1, 0)
    cnt = m.sum(dim=1)
    has_holes = row_any & (cnt != (ke - ks))
    return ks.to(torch.int32), ke.to(torch.int32), has_holes


# Hand-written CUDA kernels for the two fused decomposition sweeps, built
# into the flex_flash_attention extension (csrc/mask_decomp_kernels.cu).  Resolved
# lazily on first use: importing this module must not require the extension
# to be loaded yet (older wheels fall back to the compiled versions below).
_DECOMP_OPS_RESOLVED = False
_ROW_STATS_OP = None
_PEEL_OP = None


def _resolve_decomp_ops():
    global _DECOMP_OPS_RESOLVED, _ROW_STATS_OP, _PEEL_OP
    if not _DECOMP_OPS_RESOLVED:
        try:
            _ROW_STATS_OP = torch.ops.flex_flash_attention.row_stats
            _PEEL_OP = torch.ops.flex_flash_attention.peel_first_interval
        except AttributeError:
            pass  # wheel without the fused-decomp kernels: compiled fallback
        _DECOMP_OPS_RESOLVED = True


def _row_stats_gpu(mask: torch.Tensor):
    """Device-side row stats: (ks, ke, has_holes) int32/bool GPU tensors."""
    _resolve_decomp_ops()
    if _ROW_STATS_OP is not None and mask.is_cuda:
        return _ROW_STATS_OP(mask.contiguous())
    seqlen_k = mask.shape[1]
    idx = torch.arange(seqlen_k, device=mask.device).unsqueeze(0)
    return _row_stats_compiled(mask, idx)


class MaskNotSupportedError(ValueError):
    """The mask is outside this kernel's decomposition envelope.

    Slices describe a CONTIGUOUS [k_start, k_end) column range per Q row
    (FULL / CAUSAL / INVCAUSAL / BICAUSAL are all single-interval shapes,
    see attn_slice.h).  Rows whose visible set splits into several
    intervals ('holes') are handled by decomposing ONE interval layer per
    pass (see _decompose_with_holes); this error is raised only when the
    mask needs more layers than the cap, i.e. its per-row interval count
    is so high that the slice description degenerates (random sparsity,
    runtime-generated block selection).  The user-facing entry points
    (flash_attn_flex_flash / _bwd) catch this error and fall back to the
    exact element-wise bitmask kernel path (SLICE_BITMASK), so such masks
    still compute correctly — just without the slice-based optimizations.

    Practical masks that fit the layered scheme: attention sink /
    StreamingLLM (sink columns + causal window), Longformer (local window
    + global columns), BigBird-style window+global mixtures.
    """


def _reject_holes(has_holes: torch.Tensor, seqlen_q: int):
    """Raise MaskNotSupportedError if any row's visible set is not a single
    contiguous interval.  `has_holes` comes free with the row-stats sweep."""
    n = int(has_holes.sum().item())
    if n == 0:
        return
    first = int(has_holes.to(torch.uint8).argmax().item())
    raise MaskNotSupportedError(
        f"mask has {n} of {seqlen_q} row(s) whose visible columns are NOT a "
        f"single contiguous interval (first such row: {first}). This kernel "
        f"decomposes a mask into per-row contiguous slices "
        f"(FULL/CAUSAL/INVCAUSAL/BICAUSAL) and cannot represent row-internal "
        f"holes; accepting it would silently return wrong results. Pass "
        f"allow_holes=True only to inspect the (approximate, over-covering) "
        f"decomposition — never to compute attention.")


# Upper bound on per-row interval layers for the hole decomposition.  Every
# peel removes one interval per row, so the cap bounds the worst-case row
# interval count we accept; beyond it the slice list degenerates toward one
# slice per cell and a bitmask path is the right tool.
_MAX_HOLE_LAYERS = 16


@torch.compile
def _peel_first_interval_compiled(blk: torch.Tensor, idx: torch.Tensor):
    """One row-chunk peel: locate each row's FIRST visible interval and
    clear it from the block.  Same where/min idiom as _row_stats_compiled
    (one fused sweep, no uint8 argmax passes)."""
    seqlen_k = blk.shape[1]
    ks = torch.where(blk, idx, seqlen_k).min(dim=1).values   # first True
    anyr = ks < seqlen_k
    after = idx >= ks.unsqueeze(1)
    gap = (~blk) & after & anyr.unsqueeze(1)                 # False after ks
    ke = torch.where(gap, idx, seqlen_k).min(dim=1).values   # first False
    ks = torch.where(anyr, ks, 0)
    ke = torch.where(anyr, ke, 0)
    peeled = blk & ~(after & (idx < ke.unsqueeze(1)))        # clear [ks, ke)
    return ks.to(torch.int32), ke.to(torch.int32), peeled


def _peel_first_interval(R: torch.Tensor, chunk: int = 2048):
    """Extract every row's first visible interval from R and clear those
    cells in place.  Returns (ks_t, ke_t) int32 GPU tensors; rows left
    empty get ks == ke == 0 (the segmenter's empty-row convention).

    CUDA-op path: flex_flash_attention::peel_first_interval processes the whole
    residual in one launch (one block per row, no chunk-sized temporaries).
    Compiled fallback: row-chunked so the int64 where/min temporaries stay
    chunk x seqlen_k instead of seqlen_q x seqlen_k."""
    seqlen_q, seqlen_k = R.shape
    dev = R.device
    _resolve_decomp_ops()
    if _PEEL_OP is not None and R.is_cuda:
        Rc = R.contiguous()
        ks_t, ke_t, _ = _PEEL_OP(Rc)
        if Rc is not R:
            R.copy_(Rc)
        return ks_t, ke_t
    ks_t = torch.zeros(seqlen_q, dtype=torch.int32, device=dev)
    ke_t = torch.zeros(seqlen_q, dtype=torch.int32, device=dev)
    idx = torch.arange(seqlen_k, device=dev).unsqueeze(0)
    for r0 in range(0, seqlen_q, chunk):
        r1 = min(r0 + chunk, seqlen_q)
        ks, ke, peeled = _peel_first_interval_compiled(R[r0:r1], idx)
        ks_t[r0:r1] = ks
        ke_t[r0:r1] = ke
        R[r0:r1] = peeled
    return ks_t, ke_t


def _decompose_with_holes(mask: torch.Tensor, seqlen_k: int) -> List[SliceInfo]:
    """Layered decomposition for masks with row-internal holes.

    Each pass peels every row's FIRST visible interval off a residual copy
    of the mask and decomposes that layer with the exact same run-length
    segmentation + classification as the hole-free path.  Because each
    layer's cells are removed before the next peel, layers are pairwise
    cell-disjoint and their union equals the input mask exactly — the
    RangeMerge (vbatch/LSE-merge) kernel path then combines them without
    double counting.

    Layer 0 keeps zero-K FULL slices for fully-masked rows (the kernel's
    zero-output contract); deeper layers drop them — those rows are simply
    not covered by that layer.
    """
    R = mask.clone()
    slices: List[SliceInfo] = []
    for layer in range(_MAX_HOLE_LAYERS):
        ks_l, ke_l = _peel_first_interval(R)
        if not bool((ks_l != ke_l).any()):
            return slices                      # residual empty: done
        layer_slices = _segment_and_classify(ks_l, ke_l, seqlen_k)
        if layer > 0:
            layer_slices = [s for s in layer_slices if s.k_end > s.k_start]
        slices.extend(layer_slices)
    raise MaskNotSupportedError(
        f"mask still has visible cells after {_MAX_HOLE_LAYERS} interval "
        f"layers; its per-row interval count exceeds the layered-decompose "
        f"cap (structured sparsity like sink+window fits; random sparsity "
        f"does not).")


def decompose_mask(mask: torch.Tensor, allow_holes: bool = False) -> List[SliceInfo]:
    """Decompose an arbitrary boolean mask into slices.

    Two-pass algorithm:
      1. Compute per-row (k_start, k_end) bounds (fused GPU kernel).
      2. Group consecutive Q rows into slices, detecting cross-row patterns:
         - CAUSAL:    k_end grows linearly with q  (k_end = q + offset + 1)
         - INVCAUSAL: k_start grows linearly with q (k_start = q + offset)
         - BICAUSAL:  both k_start and k_end change
         - FULL:      k_start and k_end are constant across the group

    Every slice covers a CONTIGUOUS column range per row.  Masks whose rows
    contain several disjoint intervals (holes) are decomposed by iteratively
    peeling each row's FIRST interval as its own layer of slices until the
    residual mask is empty; layers are cell-disjoint by construction, so the
    slice union reproduces the mask exactly (see _decompose_with_holes).

    Args:
        mask: Boolean tensor of shape (seqlen_q, seqlen_k). True = attend, False = mask out.
        allow_holes: legacy inspection mode: skip the layered peel and
            return the single-layer over-covering approximation (the slices
            do NOT reproduce the mask).  Production callers should leave
            this False.

    Returns:
        List of SliceInfo describing the decomposition.
    """
    seqlen_q, seqlen_k = mask.shape
    assert mask.dtype == torch.bool, "mask must be boolean"

    if seqlen_q == 0 or seqlen_k == 0:
        return []

    # Pass 1: per-row bounds (fused GPU sweep).  has_holes comes free with the
    # same sweep and selects between the single-layer and layered paths.
    ks_t, ke_t, hh_t = _row_stats_gpu(mask)
    if allow_holes or not bool(hh_t.any()):
        return _segment_and_classify(ks_t, ke_t, seqlen_k)

    return _decompose_with_holes(mask, seqlen_k)


def _segment_rows_gpu(ks_t: torch.Tensor, ke_t: torch.Tensor, seqlen_k: int):
    """Run-length segmentation of rows by their (Δk_start, Δk_end) slope,
    equivalent to the legacy greedy extender: the prefix-sum extrapolation
    check ks[q] == ks0 + d*(q-s) holds at EVERY intermediate q, so a segment
    extends exactly while all deltas equal its first one — i.e. segments are
    the runs of the delta sequence.  Empty rows (ks == ke) get a poison
    delta (> seqlen_k, unreachable by any real delta) so they always form
    standalone segments.  All on device; only the boundaries come back.

    One subtlety vs plain run-length encoding: the legacy greedy extender
    anchors a segment at row s with slope d[s] and admits row s+1
    UNCONDITIONALLY (the first extrapolation check is at s+2).  The raw
    adjacency test therefore over-splits whenever a single-delta transition
    run (e.g. a doc/frame boundary jump like (2048, 1)) sits between two
    runs of the same slope: it marks BOTH ends of the 1-delta run, carving
    the next run's first row off into a bogus single-row segment.  The
    greedy anchor rule instead drops the boundary right after each segment
    start — kept boundaries are >= 2 rows apart — and re-anchors the slope
    at the run's true first delta (see _segment_and_classify gathering
    d[b[i]]).  The boundaries list is tiny for structured masks, so the
    min-gap filter runs on CPU.

    Two corrections keep empty rows isolated under that min-gap filter:
    (a) row 1 is itself a boundary when emptiness flips between rows 0
    and 1 — the "admit row s+1 unconditionally" rule never applies across
    an empty/non-empty transition; (b) boundaries adjacent to an empty row
    are exempt from the >= 2 gap drop, so the poison deltas always keep
    empty runs as standalone segments.  Without (b) the boundary leaving a
    lone empty row is dropped, the following non-empty row is merged into
    the empty segment and silently classified as empty (zero output).

    Returns (boundaries, d_ks, d_ke): boundaries is a CPU list of segment
    start rows (len >= 1, boundaries[0] == 0); d_ks/d_ke are the int64 GPU
    delta tensors (length seqlen_q - 1, poisoned at entries entering an
    empty row).
    """
    seqlen_q = ks_t.shape[0]
    if seqlen_q == 1:
        return [0], None, None

    empty = ks_t == ke_t
    # Poison delta for empty-row entries: must be unreachable by any real
    # delta (|delta| <= seqlen_k) AND by any empty row's outgoing delta
    # (ks of the next row, <= seqlen_k).
    poison = seqlen_k + ks_t.shape[0] + 1
    d_ks = torch.diff(ks_t.to(torch.int64))
    d_ke = torch.diff(ke_t.to(torch.int64))
    d_ks = torch.where(empty[1:], poison, d_ks)
    d_ke = torch.where(empty[1:], poison, d_ke)
    # Candidate segment start at row i when the slope changed entering it.
    # Row 1 is normally admitted unconditionally by the greedy anchor rule,
    # EXCEPT when emptiness flips between rows 0 and 1: an empty anchor row
    # must never extrapolate its (0, 0) slope over a non-empty row (or vice
    # versa), so that transition is always a boundary.
    changed = torch.zeros(seqlen_q, dtype=torch.bool, device=ks_t.device)
    changed[0] = True
    changed[1] = empty[0] ^ empty[1]
    changed[2:] = (d_ks[1:] != d_ks[:-1]) | (d_ke[1:] != d_ke[:-1])
    b = torch.nonzero(changed, as_tuple=False).squeeze(1).cpu().tolist()
    # Greedy anchor filter: after a segment start at s the extender only
    # checks rows >= s+2, so the next boundary must be >= s + 2.  Drop the
    # adjacency right after every kept boundary (single-delta transition
    # runs are re-anchored, not split off).  Boundaries adjacent to an
    # empty row are EXEMPT from the drop: poison deltas must keep empty
    # runs as standalone segments, otherwise the row right after a lone
    # empty row gets merged into the empty segment and is silently
    # classified as empty (see docstring).
    near_empty = (empty[:-1] | empty[1:]).cpu().tolist()
    if len(b) >= 2:
        kept = [b[0]]
        for x in b[1:]:
            if x >= kept[-1] + 2 or near_empty[x - 1]:
                kept.append(x)
        b = kept
    return b, d_ks, d_ke


def _segment_and_classify(ks_t: torch.Tensor, ke_t: torch.Tensor, seqlen_k: int) -> List[SliceInfo]:
    """GPU segmentation + tiny gathers + CPU slice classification."""
    global _FALLBACK_KS, _FALLBACK_KE
    _FALLBACK_KS, _FALLBACK_KE = ks_t, ke_t
    seqlen_q = ks_t.shape[0]
    b, d_ks, d_ke = _segment_rows_gpu(ks_t, ke_t, seqlen_k)
    ends = b[1:] + [seqlen_q]

    # First-row (ks, ke) of every segment; slopes from the segment's first
    # row-pair (single-row segments get (0, 0) like the legacy greedy path).
    idx_b = torch.tensor(b, device=ks_t.device, dtype=torch.long)
    ks0 = ks_t[idx_b].cpu().tolist()
    ke0 = ke_t[idx_b].cpu().tolist()
    multi = [i for i, (s, e) in enumerate(zip(b, ends)) if e - s >= 2]
    d_ks_l = [0] * len(b)
    d_ke_l = [0] * len(b)
    if multi and d_ks is not None:
        idx_m = torch.tensor([b[i] for i in multi], device=ks_t.device, dtype=torch.long)
        gm = d_ks[idx_m].cpu().tolist()
        gk = d_ke[idx_m].cpu().tolist()
        for i, s_ in zip(multi, gm):
            d_ks_l[i] = s_
        for i, s_ in zip(multi, gk):
            d_ke_l[i] = s_

    segs = [(s, e, d_ks_l[i], d_ke_l[i]) for i, (s, e) in enumerate(zip(b, ends))]
    return _classify_segments(segs, ks0, ke0)


def _classify_segments(segs, ks0_l, ke0_l) -> List[SliceInfo]:
    """Map (q_start, q_end, d_ks, d_ke) segments to SliceInfo, with the
    same type/offset/band rules as the legacy greedy decomposer."""
    slices = []
    for (q_start, q_end, d_ks, d_ke), ks0, ke0 in zip(segs, ks0_l, ke0_l):
        empty_row = ks0 == 0 and ke0 == 0

        if empty_row:
            # Empty row(s): standalone zero-K FULL slice.
            slices.append(SliceInfo(q_start, q_end, 0, 0, SLICE_FULL, 0, 0))
            continue

        if d_ks == 0 and d_ke == 0:
            # Constant K range → FULL.  With the contiguity gate in place the
            # rectangle is exact; under allow_holes=True it over-covers.
            mask_type = SLICE_FULL
            diag_offset = 0
            band_width = 0
            k_start = ks0
            k_end = ke0
        elif d_ks == 0 and d_ke == 1:
            # k_end grows with q → CAUSAL (lower triangle)
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
            # Both grow → BICAUSAL (band).  Semantics: the band's right edge
            # is pinned at the diagonal (k in [q+off-bw, q+off]), so every
            # constant-width sloped window is represented EXACTLY — including
            # even widths (the old symmetric-band form over-covered one
            # column per row for asymmetric/even windows, e.g. causal
            # sliding windows).
            mask_type = SLICE_BICAUSAL
            diag_offset = ke0 - 1 - q_start
            band_width = ke0 - 1 - ks0
            k_start = ks0
            k_end = ke0 + (q_end - q_start - 1)
        else:
            # Unknown pattern — the legacy greedy path truncates to one row
            # and restarts; with constant-delta segments that is equivalent
            # to emitting single-row FULL slices for the whole segment, EXCEPT
            # the first row's k-bounds must come from the per-row stats (they
            # are needed on CPU — fetch lazily from the device via closure).
            slices.extend(_single_row_fallback(q_start, q_end))
            continue

        slices.append(SliceInfo(q_start, q_end, k_start, k_end, mask_type, diag_offset, band_width))
    return slices


# Per-row bounds store for the unknown-pattern fallback (set by
# decompose_mask before classification; CPU lists would cost a full D2H,
# so the fallback gathers just the rows it needs).
_FALLBACK_KS = None
_FALLBACK_KE = None


def _single_row_fallback(q_start, q_end):
    ks_t, ke_t = _FALLBACK_KS, _FALLBACK_KE
    idx = torch.arange(q_start, q_end, device=ks_t.device)
    ks_l = ks_t[idx].cpu().tolist()
    ke_l = ke_t[idx].cpu().tolist()
    return [SliceInfo(q, q + 1, ks, ke, SLICE_FULL, 0, 0)
            for q, ks, ke in zip(range(q_start, q_end), ks_l, ke_l)]


def slices_to_tensors(slices: List[SliceInfo], seqlen_q: int,
                      device: torch.device = torch.device('cpu')) -> dict:
    """Convert slice list to GPU-ready tensors for the kernel.

    Returns a dict with keys: q_starts, q_ends, k_starts, k_ends, mask_types,
    row_to_slice, diagonal_offsets, band_widths.

    Note: row_to_slice maps each row to its FIRST covering slice; this is the
    legacy non-overlapping dispatch table.  For overlapping slices use
    build_merge_layout(), which additionally produces the RangeMerge layout.
    """
    num_slices = len(slices)
    # Everything is built directly on the target device.  (The earlier
    # CPU-first variant paid hundreds of ms at seqlen_q ~ 25k for the
    # row_to_slice boolean/fancy indexing — a torch CPU slow path that
    # also inflates >10x under host load; see bench_decomp/RESULT.md.)
    dev = torch.device(device)
    if num_slices == 0:
        empty = lambda: torch.zeros(0, dtype=torch.int32, device=dev)
        return {
            'q_starts': empty(), 'q_ends': empty(), 'k_starts': empty(),
            'k_ends': empty(), 'mask_types': empty(), 'row_to_slice':
            torch.full((seqlen_q,), -1, dtype=torch.int32, device=dev),
            'diagonal_offsets': empty(), 'band_widths': empty(),
        }

    q_starts = torch.tensor([s.q_start for s in slices], dtype=torch.int32,
                            device=dev)
    q_ends = torch.tensor([s.q_end for s in slices], dtype=torch.int32,
                          device=dev)
    k_starts = torch.tensor([s.k_start for s in slices], dtype=torch.int32,
                            device=dev)
    k_ends = torch.tensor([s.k_end for s in slices], dtype=torch.int32,
                          device=dev)
    mask_types = torch.tensor([s.mask_type for s in slices],
                              dtype=torch.int32, device=dev)
    diagonal_offsets = torch.tensor([s.diagonal_offset for s in slices],
                                    dtype=torch.int32, device=dev)
    band_widths = torch.tensor([s.band_width for s in slices],
                               dtype=torch.int32, device=dev)

    # row_to_slice: first covering slice wins.  Slices arrive sorted by
    # q_start (decompose / align_slices_to_tiles preserve row order), so
    # for each row the first slice with q_end > row is THE covering slice
    # iff its q_start <= row; later slices cannot cover a row an earlier
    # sorted slice missed (their q_start only grows).
    rows = torch.arange(seqlen_q, dtype=torch.int32, device=dev)
    i = torch.searchsorted(q_ends, rows, right=True).to(torch.int32)
    valid = i < num_slices
    # Clamp the out-of-range indices so the gather below is always safe;
    # those rows are discarded via `valid` anyway.  This elementwise form
    # avoids the boolean-masked assignment `covered[valid] = ...`, which
    # is a variable-length fancy-indexing slow path (~2x cost here).
    ic = torch.clamp(i, max=num_slices - 1)
    covered = valid & (q_starts[ic] <= rows)
    row_to_slice = torch.where(
        covered, i, torch.full((seqlen_q,), -1, dtype=torch.int32,
                               device=dev))

    return {
        'q_starts': q_starts,
        'q_ends': q_ends,
        'k_starts': k_starts,
        'k_ends': k_ends,
        'mask_types': mask_types,
        'row_to_slice': row_to_slice,
        'diagonal_offsets': diagonal_offsets,
        'band_widths': band_widths,
    }


# ============================================================================
# RangeMerge layout: slices may overlap in Q (per MagiAttention FFA)
# ============================================================================

def build_merge_layout(slices: List[SliceInfo], seqlen_q: int,
                       kBlockM: int = 128,
                       device: torch.device = torch.device('cpu')) -> dict:
    """Build the RangeMerge (virtual batch) layout for possibly-overlapping
    slices, plus all per-slice tensors.

    A "vbatch" (virtual batch, MagiAttention terminology) is one
    (m_block, covering slice) work entry.  The kernel loops over all vbatches
    covering an m_block with a persistent online-softmax state, merging the
    per-slice partial results via LSE.

    Slices are first re-split at kBlockM boundaries (align_slices_to_tiles)
    so that slice coverage is constant within each m_block; this keeps
    diagonal/band parameters valid because they are per-row formulas in
    global Q coordinates.

    Returns a dict containing all slices_to_tensors() keys plus:
      vbatch_to_slice      [num_vbatches] int32 — vbatch → slice index
      row_to_vbatch_start  [seqlen_q]     int32 — first covering vbatch
      row_to_vbatch_end    [seqlen_q]     int32 — one past last covering
      slices               the (re-aligned) SliceInfo list
    Rows not covered by any slice get start == end (kernel writes zeros).
    """
    slices = align_slices_to_tiles(slices, kBlockM) if kBlockM > 0 else slices

    tensors = slices_to_tensors(slices, seqlen_q, device=device)

    num_m_blocks = (seqlen_q + kBlockM - 1) // kBlockM if kBlockM > 0 else 0
    dev = torch.device(device)
    if num_m_blocks == 0 or len(slices) == 0:
        tensors.update({
            'vbatch_to_slice': torch.zeros(0, dtype=torch.int32, device=dev),
            'row_to_vbatch_start': torch.zeros(seqlen_q, dtype=torch.int32, device=dev),
            'row_to_vbatch_end': torch.zeros(seqlen_q, dtype=torch.int32, device=dev),
            'slices': slices,
        })
        return tensors

    # Covering computation, vectorised over an (m_block x slice) coverage
    # matrix.  Order-independent: any slice may overlap any other in Q
    # (align_slices_to_tiles emits fragments grouped by source slice, NOT
    # q_start-sorted, and raw user layouts may overlap arbitrarily).
    # An earlier searchsorted formulation silently assumed q_start order and
    # dropped all but one slice on multi-block Q-overlapping layouts.
    #   cov[m, s]   = slice s covers any row of m_block m
    #   counts[m]   = covering slices per m_block
    #   pos[m, s]   = rank of slice s within block m's vbatch list
    qs = tensors['q_starts'].long()
    qe = tensors['q_ends'].long()
    m_lo = torch.arange(num_m_blocks, dtype=torch.int64, device=dev) * kBlockM
    m_hi = torch.minimum(m_lo + kBlockM,
                         torch.full_like(m_lo, int(seqlen_q)))
    cov = ((qe.view(1, -1) > m_lo.view(-1, 1))
           & (qs.view(1, -1) < m_hi.view(-1, 1)))
    counts = cov.sum(dim=1)
    vb_first = (counts.cumsum(0) - counts).to(torch.int32)
    total = int(counts.sum())

    if total > 0:
        pos = cov.long().cumsum(dim=1) - 1          # rank among covering
        m_idx, s_idx = cov.nonzero(as_tuple=True)   # row-major: m then s
        # cov is True-contiguous per row in (m, s) order, so the flattened
        # vbatch id of entry (m, s) is vb_first[m] + pos[m, s].
        vb_ids = vb_first[m_idx] + pos[m_idx, s_idx].to(torch.int32)
        vb_tensor = torch.zeros(total, dtype=torch.int32, device=dev)
        vb_tensor[vb_ids] = s_idx.to(torch.int32)
    else:
        vb_tensor = torch.zeros(0, dtype=torch.int32, device=dev)

    # Per-row vbatch range: broadcast the per-block bounds to rows.
    cum = (vb_first + counts.to(torch.int32)).to(torch.int32)
    row_block = torch.arange(seqlen_q, dtype=torch.int32, device=dev) // kBlockM
    row_block = torch.minimum(row_block, torch.full_like(row_block, num_m_blocks - 1))
    vb_lo_per_row = vb_first[row_block]
    vb_hi_per_row = cum[row_block]

    tensors.update({
        'vbatch_to_slice': vb_tensor,
        'row_to_vbatch_start': vb_lo_per_row,
        'row_to_vbatch_end': vb_hi_per_row,
        'slices': slices,
    })
    return tensors


# ============================================================================
# P4 Optimization: tile-aligned slice merging
# ============================================================================

def merge_adjacent_slices(slices: List[SliceInfo]) -> List[SliceInfo]:
    """Merge consecutive slices that have identical mask_type, k_range, and
    diagonal parameters.  This reduces the number of kernel-side lookups.

    Two adjacent slices can be merged when:
      - They share the same mask_type
      - They have the same k_start, k_end, diagonal_offset, band_width
      - The first slice's q_end == the second slice's q_start (contiguous)
    """
    if len(slices) <= 1:
        return slices

    merged = [slices[0]]
    for s in slices[1:]:
        prev = merged[-1]
        if (prev.q_end == s.q_start
            and prev.mask_type == s.mask_type
            and prev.k_start == s.k_start
            and prev.k_end == s.k_end
            and prev.diagonal_offset == s.diagonal_offset
            and prev.band_width == s.band_width):
            # Extend previous slice
            merged[-1] = SliceInfo(
                prev.q_start, s.q_end,
                prev.k_start, prev.k_end,
                prev.mask_type, prev.diagonal_offset, prev.band_width)
        else:
            merged.append(s)
    return merged


def align_slices_to_tiles(slices: List[SliceInfo], kBlockM: int = 128) -> List[SliceInfo]:
    """Split slices at every kBlockM Q-boundary so that each resulting slice
    spans at most ONE kBlockM row-tile.  This minimises wasted tiles in the
    kernel and makes the per-tile covering-slice set constant.

    For example, with kBlockM=128:
      - A slice covering q=[0, 200) is split into [0, 128) and [128, 200).
      - A slice covering q=[50, 300) is split into [50, 128), [128, 256),
        [256, 300) — note that even tile-aligned boundaries (128, 256) cut
        the slice; "aligned" inputs are NOT passed through whole.
      - A slice covering q=[0, 256) is split into [0, 128) and [128, 256).

    Only slices that already fit inside a single tile pass through
    unchanged.  K bounds / offset / band_width are copied verbatim (no
    K-side alignment).
    """
    if kBlockM <= 0 or not slices:
        return slices

    aligned = []
    for s in slices:
        q = s.q_start
        while q < s.q_end:
            # Next tile boundary or slice end, whichever comes first
            next_boundary = ((q // kBlockM) + 1) * kBlockM
            q_end = min(next_boundary, s.q_end)

            if q == s.q_start and q_end == s.q_end:
                # Entire slice fits — no split needed
                aligned.append(s)
            else:
                aligned.append(SliceInfo(
                    q, q_end,
                    s.k_start, s.k_end,
                    s.mask_type, s.diagonal_offset, s.band_width))
            q = q_end
    return aligned


def decompose_mask_optimized(
    mask: torch.Tensor,
    kBlockM: int = 128,
    tile_align: bool = True,
    allow_holes: bool = False,
) -> List[SliceInfo]:
    """Full pipeline: decompose → merge → tile-align.

    This is the recommended entry point for production use.

    Args:
        mask: Boolean tensor of shape (seqlen_q, seqlen_k).  Rows whose
            visible columns form several disjoint intervals (holes) are
            decomposed into one slice layer per interval; only masks whose
            per-row interval count exceeds the layer cap raise
            MaskNotSupportedError.
        kBlockM: Tile size in the Q dimension (must match the kernel).
        tile_align: If True, align slice Q-boundaries to kBlockM.
        allow_holes: legacy inspection mode: single-layer over-covering
            approximation instead of the exact layered decomposition.

    Returns:
        Optimized list of SliceInfo.
    """
    slices = decompose_mask(mask, allow_holes=allow_holes)
    slices = merge_adjacent_slices(slices)
    if tile_align:
        slices = align_slices_to_tiles(slices, kBlockM)
    return slices


def stair_slices(seqlen_q: int, frame_size: int, context_frames: int,
                 kBlockM: int = 128) -> List[SliceInfo]:
    """Build the slice layout of a frame-window stair mask directly from
    its parameters, without materialising the (seqlen, seqlen) mask.

    The mask modelled here is the "chunked causal with context" family:
    row r attends to columns [max(0, (r // frame_size - context_frames)
    * frame_size), r] — causal inside the current frame, plus the
    `context_frames` preceding frames in full.

    The output is field-identical to decompose_mask_optimized() on that
    mask (validated in tests/test_stair_ctor.py), so this is a drop-in
    bypass for callers that know the mask shape analytically: O(nframes)
    instead of O(S^2) mask construction + GPU decomposition (~25x
    end-to-end at S=25k; also saves S^2 bytes of mask memory — see
    bench_decomp/RESULT.md).

    Args:
        seqlen_q: Sequence length (square mask, seqlen_q >= 0).
        frame_size: Rows per frame (>= 1).
        context_frames: Extra fully-visible preceding frames (>= 0).
        kBlockM: Tile size in the Q dimension (must match the kernel).

    Segment structure (matches the decomposer's output exactly):
      seg0: rows [0, min(S, (C+1)*P))            k_start=0  (clamp region)
      seg_f (f = C+1 .. nframes-1): rows [fP, (f+1)P), k_start=(f-C)*P
      tail: rows [nframes*P, S) when non-empty AND nframes > C,
            k_start=(nframes-C)*P
    Every multi-row segment is CAUSAL (offset 0, k_end=q_end envelope);
    a single-row tail degrades to FULL.
    """
    if seqlen_q < 0:
        raise ValueError(f"seqlen_q must be >= 0, got {seqlen_q}")
    if frame_size < 1:
        raise ValueError(f"frame_size must be >= 1, got {frame_size}")
    if context_frames < 0:
        raise ValueError(
            f"context_frames must be >= 0, got {context_frames}")

    nframes = seqlen_q // frame_size
    tail = seqlen_q - nframes * frame_size
    slices = []

    def emit(q0, q1, ks):
        L = q1 - q0
        if L <= 0:
            return
        if L == 1:
            slices.append(SliceInfo(q0, q1, ks, ks + 1, SLICE_FULL, 0, 0))
        else:
            slices.append(SliceInfo(q0, q1, ks, q1, SLICE_CAUSAL, 0, 0))

    emit(0, min(seqlen_q, (context_frames + 1) * frame_size), 0)
    for f in range(context_frames + 1, nframes):
        emit(f * frame_size, (f + 1) * frame_size,
             (f - context_frames) * frame_size)
    # The tail is already inside seg0 when nframes <= context_frames.
    if tail and nframes >= context_frames + 1:
        emit(nframes * frame_size, seqlen_q,
             (nframes - context_frames) * frame_size)

    # Same post-processing as decompose_mask_optimized so the output is
    # field-identical (merge is a no-op here today but keeps parity).
    return align_slices_to_tiles(merge_adjacent_slices(slices), kBlockM)


def build_bh_layout(group_layouts: List[dict], seqlen_q: int,
                    device: torch.device) -> dict:
    """Flat-concatenate per-group layouts into one heterogeneous (P3) layout.

    Each entry of `group_layouts` is a build_merge_layout() dict (plus an
    optional 'mask_bits' key from pack_mask_bits) describing one unique
    mask shared by a set of (batch, head) pairs.  The result holds the
    FLAT concatenation of all per-slice and RangeMerge tables plus the
    group prefix sums consumed by the kernel's rebase_slices():

      group_slice_offsets  [num_groups + 1] int32 — slice-array offsets
      group_vb_offsets     [num_groups + 1] int32 — vbatch-array offsets
      row tables           num_groups * seqlen_q entries each
      mask_bits            num_groups * seqlen_q rows (zero rows for
                           groups whose masks decompose into algebraic
                           slices; those rows are never read by the kernel)

    Group-local slice/vbatch indices in row_to_slice / vbatch_to_slice /
    row_to_vbatch_* are rebased to flat indices; uncovered rows keep their
    negative / start==end semantics (negative entries are NOT shifted).
    """
    dev = torch.device(device)
    keys_flat = ('q_starts', 'q_ends', 'k_starts', 'k_ends', 'mask_types',
                 'diagonal_offsets', 'band_widths')
    parts = {kk: [] for kk in keys_flat}
    parts['row_to_slice'] = []
    parts['vbatch_to_slice'] = []
    parts['row_to_vbatch_start'] = []
    parts['row_to_vbatch_end'] = []
    slice_offsets = [0]
    vb_offsets = [0]
    bits_parts = []
    any_bits = False
    bits_stride = 0
    for gl in group_layouts:
        s_off = slice_offsets[-1]
        v_off = vb_offsets[-1]
        n_slices = gl['q_starts'].shape[0]
        n_vb = gl['vbatch_to_slice'].shape[0]
        for kk in keys_flat:
            parts[kk].append(gl[kk])
        r2s = gl['row_to_slice']
        parts['row_to_slice'].append(
            torch.where(r2s < 0, r2s, r2s + s_off) if s_off else r2s)
        v2s = gl['vbatch_to_slice']
        parts['vbatch_to_slice'].append(v2s + s_off if s_off else v2s)
        parts['row_to_vbatch_start'].append(gl['row_to_vbatch_start'] + v_off)
        parts['row_to_vbatch_end'].append(gl['row_to_vbatch_end'] + v_off)
        gb = gl.get('mask_bits')
        if gb is not None:
            any_bits = True
            bits_stride = gb.shape[1]
        bits_parts.append(gb)
        slice_offsets.append(s_off + n_slices)
        vb_offsets.append(v_off + n_vb)
    out = {kk: torch.cat(v) for kk, v in parts.items()}
    if any_bits:
        filled = [p if p is not None else
                  torch.zeros(seqlen_q, bits_stride,
                              dtype=torch.uint8, device=dev)
                  for p in bits_parts]
        out['mask_bits'] = torch.cat(filled)
    else:
        out['mask_bits'] = None
    out['group_slice_offsets'] = torch.tensor(slice_offsets, dtype=torch.int32,
                                              device=dev)
    out['group_vb_offsets'] = torch.tensor(vb_offsets, dtype=torch.int32,
                                           device=dev)
    out['num_groups'] = len(group_layouts)
    return out
