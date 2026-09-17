"""Equivalence test for stair_slices, the parametric constructor that
bypasses mask materialisation + GPU decomposition.

For every (frame_size, context_frames, seqlen) config, the constructor
output must be FIELD-IDENTICAL to decompose_mask_optimized() applied to
the explicitly-built frame-window stair mask — the constructor is a
drop-in bypass, not an approximation.  Runs on GPU (the golden path
needs it).
"""
import os
import sys
import time

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
    stair_slices, decompose_mask_optimized,
)

DEV = torch.device('cuda')


def frame_window_stair(S, P, C, device=DEV):
    """Row r attends to columns [max(0, (r//P - C)*P), r] — the mask the
    constructor claims to reproduce (same definition as the production
    training mask used in bench_decomp)."""
    row = torch.arange(S, device=device)
    col = torch.arange(S, device=device)
    lo = ((row // P - C) * P).clamp(min=0)
    return (col.view(1, -1) >= lo.view(-1, 1)) & \
           (col.view(1, -1) <= row.view(-1, 1))


# (frame_size, context_frames, seqlen).  The first seven are the bench
# configs; the rest are boundary cases around the seg0/tail rules.
CFGS = [
    (4, 1, 16), (4, 1, 20), (8, 2, 48),
    (2048, 5, 25286), (512, 3, 4096), (64, 0, 320), (128, 7, 1000),
    (128, 0, 128),     # exactly one frame, C=0
    (128, 5, 100),     # S < P: whole sequence inside the clamp region
    (300, 128, 2),     # tail = 44 rows
    (257, 128, 1),     # tail = 1 row -> FULL tail slice
    (64, 0, 63),       # nframes == 0
    (300, 64, 4),      # nframes == C: seg0 already swallows the tail
    (320, 64, 4),      # nframes == C+1: exactly one free frame, no tail
]

passed = 0
failed = 0

print("=" * 74)
print("stair_slices equivalence: ctor vs decompose_mask_optimized")
print("=" * 74)
for P, C, S in CFGS:
    m = frame_window_stair(S, P, C)
    golden = decompose_mask_optimized(m, 128, True)
    ctor = stair_slices(S, P, C, 128)
    ok = ctor == golden
    print(f"  {'PASS' if ok else 'FAIL'}: P={P:<5} C={C} S={S:<6} "
          f"slices={len(golden)}")
    if not ok:
        failed += 1
        for g, c in zip(golden, ctor):
            if g != c:
                print(f"    golden: {g}\n    ctor  : {c}")
                break
        if len(golden) != len(ctor):
            print(f"    slice count differs: golden={len(golden)} "
                  f"ctor={len(ctor)}")
    else:
        passed += 1
    del m

# Input validation must reject nonsense parameters.
for bad in [(-1, 128, 0), (10, 0, 0), (10, 128, -1)]:
    try:
        stair_slices(*bad)
        print(f"  FAIL: no ValueError for {bad}")
        failed += 1
    except ValueError:
        print(f"  PASS: rejects {bad}")
        passed += 1

# Informational: end-to-end speedup on the production config (no assert —
# hardware-dependent; the equivalence above is the contract).
S, P, C = 25286, 2048, 5
torch.cuda.synchronize()
t0 = time.perf_counter()
m = frame_window_stair(S, P, C)
decompose_mask_optimized(m, 128, True)
torch.cuda.synchronize()
t_mask = (time.perf_counter() - t0) * 1000
ts = []
for _ in range(50):
    t0 = time.perf_counter()
    stair_slices(S, P, C, 128)
    ts.append((time.perf_counter() - t0) * 1000)
ts.sort()
print(f"\n  perf (S={S}): mask+decompose {t_mask:.2f} ms vs ctor "
      f"{ts[len(ts)//2]:.3f} ms ({t_mask/ts[len(ts)//2]:.0f}x)")
del m

print(f"\nRESULT: passed={passed} failed={failed}")
sys.exit(1 if failed else 0)
