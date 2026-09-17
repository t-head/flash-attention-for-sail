#!/usr/bin/env python3
# ===========================================================================
# fp32 (TF32 compute) kernel set accuracy guard — P4 Stage A.
#
# The fp32 entry points (flash_attn_flex_flash_f32 / _bwd_f32) run an
# ISOLATED kernel set: inputs float32, MMA on TF32 tensor cores, fp32
# accumulate.  This script guards their numerics against the fp32 dense
# reference shared with test_arb_suite, and calibrates the TF32 tolerance
# (printed per case so it can be tightened once observed).
#
# Expected scale: TF32 has a 10-bit mantissa, so errors sit between bf16
# (7-bit) and true fp32: rms-relative ~1e-3..5e-3 on unit-variance data.
#
# Usage:
#   python test_fp32_guard.py                 # full guard table
#   python test_fp32_guard.py --calibrate     # print metrics, no gating
# ===========================================================================
import argparse
import os
import sys

import torch

_TESTS = os.path.dirname(os.path.abspath(__file__))
_HOPPER = os.path.dirname(os.path.dirname(_TESTS))
for p in (_TESTS, _HOPPER):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_arb_suite import (                        # noqa: E402
    dense_ref, err_metrics, maxdiff, check, section,
)
from test_mask_fixtures import (                    # noqa: E402
    make_causal_mask, make_sliding_window_mask,
)

DEV = torch.device('cuda')
SEEDS = (42, 7)

# TF32 gates — loose at first observation, tighten after calibration runs.
# (max-abs, rms-relative) per tensor, applied with the same noise floor as
# the bf16 suite (abs < 1e-5 waives the relative gate on ~zero refs).
F32_REL_TOL = 5e-3
F32_MAX_TOL = 2e-2


def rel_gate_ok(metrics):
    return all(mx < F32_MAX_TOL and (rel < F32_REL_TOL or mx < 1e-5)
               for mx, _rms, rel in metrics)


def random_bitmask(sq, sk, keep, seed, device):
    g = torch.Generator(device='cpu').manual_seed(seed)
    return torch.rand(sq, sk, generator=g) < keep


def fmt(m):
    mx, rms, rel = m
    return f"max={mx:.5f} rms={rms:.6f} rel={rel:.5f}"


def run_case(name, q, k, v, mask, dout):
    """fwd+bwd on the f32 entry vs fp32 dense ref; returns metrics dict."""
    from flex_flash_attention import interface as itf

    ref = dense_ref(q, k, v, mask, dout=dout)
    o, lse = itf.flash_attn_flex_flash_f32(q, k, v, mask)
    m_out = err_metrics(o, ref['out'])
    # lse lives in log-space; compare absolute, nan==nan on dead rows.
    dl = (lse - ref['lse']).abs()
    dl = torch.where(torch.isnan(lse) & torch.isnan(ref['lse']),
                     torch.zeros_like(dl), dl)
    m_lse = (dl.max().item(), dl.pow(2).mean().sqrt().item(), 0.0)

    dq, dk, dv = itf.flash_attn_flex_flash_bwd_f32(
        q, k, v, o, lse, dout, mask)
    m_dq = err_metrics(dq, ref['dq'])
    m_dk = err_metrics(dk, ref['dk'])
    m_dv = err_metrics(dv, ref['dv'])
    print(f"  [{name}]")
    print(f"    out: {fmt(m_out)}  lse_max={m_lse[0]:.6f}")
    print(f"    dq:  {fmt(m_dq)}")
    print(f"    dk:  {fmt(m_dk)}")
    print(f"    dv:  {fmt(m_dv)}")
    return dict(out=m_out, lse=m_lse, dq=m_dq, dk=m_dk, dv=m_dv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibrate', action='store_true',
                    help='print metrics only, no gating')
    ap.add_argument('--hdim', type=int, default=128,
                    help='head dim to exercise (slim builds only have 128)')
    args = ap.parse_args()

    from flex_flash_attention import interface as itf   # early fail if not built
    d = args.hdim

    masks = {}
    sq = sk = 512
    masks['full'] = torch.ones(sq, sk, dtype=torch.bool, device=DEV)
    masks['causal'] = make_causal_mask(sq, sk, device=DEV)
    masks['swindow'] = make_sliding_window_mask(sq, sk, 128, 0, device=DEV)
    masks['rand10'] = random_bitmask(sq, sk, 0.10, 3, DEV).to(DEV)
    masks['rand50'] = random_bitmask(sq, sk, 0.50, 3, DEV).to(DEV)

    section(f"fp32/TF32 kernel guard (hdim={d})")
    cases = []
    for h, hk, bsz, tag in ((4, 4, 2, 'mha'), (4, 2, 2, 'gqa')):
        for mname, m in masks.items():
            cases.append((f"{tag}/{mname}", h, hk, bsz, m))

    all_metrics = []
    for tag, h, hk, bsz, m in cases:
        for seed in SEEDS:
            # several draws keep the gate from fitting one lucky sample
            torch.manual_seed(seed)
            q = torch.randn(bsz, sq, h, d, device=DEV, dtype=torch.float32)
            k = torch.randn(bsz, sk, hk, d, device=DEV, dtype=torch.float32)
            v = torch.randn(bsz, sk, hk, d, device=DEV, dtype=torch.float32)
            dout = torch.randn(bsz, sq, h, d, device=DEV,
                               dtype=torch.float32)
            met = run_case(f"{tag}#s{seed}", q, k, v, m, dout)
            all_metrics.append((f"{tag}#s{seed}", met))
            del q, k, v, dout
            torch.cuda.empty_cache()

    section("fp32/TF32 gate summary")
    if args.calibrate:
        print("calibration mode — no gating applied")
        return
    for tag, met in all_metrics:
        m = [met['out'], met['dq'], met['dk'], met['dv']]
        check(f"{tag} within TF32 gate (worst rel="
              f"{max(x[2] for x in m):.5f}, worst max="
              f"{max(x[0] for x in m):.5f})",
              rel_gate_ok(m) and met['lse'][0] < F32_MAX_TOL)
    print()
    from test_arb_suite import _passed, _failed, _fail_names
    print(f"FP32 GUARD RESULT: passed={_passed} failed={_failed}")
    if _failed:
        for n in _fail_names:
            print(f"  FAILED: {n}")
        sys.exit(1)


if __name__ == '__main__':
    main()
