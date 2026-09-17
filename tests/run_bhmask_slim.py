"""Slim-build P3 verification: bf16 / hdim64 only.

Mirrors suite_bhmask's heterogeneous cases (four structurally different
masks incl. the bitmask fallback, MHA + GQA, group sharing across batch AND
head) plus the single-group degeneration, restricted to the dtypes/hdims
compiled into the slim debug wheel.
"""
import sys

import test_arb_suite as ts


def main():
    itf = ts._load_kernel()
    if itf is None:
        print("kernel extension unavailable")
        return 2
    import torch
    DEV = ts.DEV
    Sq, Sk = 197, 233
    atol = 2e-2

    torch.manual_seed(11)
    rand_m = torch.rand(Sq, Sk, device=DEV) < 0.45
    masks4 = [
        torch.ones(Sq, Sk, dtype=torch.bool, device=DEV),
        ts.make_causal_mask(Sq, Sk, device=DEV),
        ts.make_sliding_window_mask(Sq, Sk, 40, 20, device=DEV),
        rand_m,
    ]

    def bh_ref(q, k, v, mask_bh, do):
        b, s_q, h, d = q.shape
        h_k = k.shape[2]
        rep = h // h_k
        outs, lses = [], []
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        for ib in range(b):
            for ih in range(h):
                r = ts.dense_ref(q[ib:ib + 1, :, ih:ih + 1],
                                 k[ib:ib + 1, :, ih // rep:ih // rep + 1],
                                 v[ib:ib + 1, :, ih // rep:ih // rep + 1],
                                 mask_bh[ib, ih],
                                 dout=(do[ib:ib + 1, :, ih:ih + 1]
                                       if do is not None else None))
                outs.append(r['out'])
                lses.append(r['lse'])
                if do is not None:
                    dq[ib, :, ih] = r['dq'][0, :, 0]
                    dk[ib, :, ih // rep] += r['dk'][0, :, 0]
                    dv[ib, :, ih // rep] += r['dv'][0, :, 0]
        # (b, h) major order: reshape to (b, h, s_q, d) first, then permute.
        out = torch.cat(outs, dim=0).reshape(b, h, s_q, -1).permute(0, 2, 1, 3)
        lse = torch.cat(lses, dim=0).reshape(b, h, s_q)
        return {'out': out, 'lse': lse, 'dq': dq, 'dk': dk, 'dv': dv}

    for (h, h_k) in ((4, 4), (4, 2), (8, 2)):
        mask_bh = torch.stack([
            torch.stack([masks4[hh % 4] for hh in range(h)])
            for _ in range(2)
        ]).to(DEV)
        tag = f"bhmask h={h} hk={h_k}"
        with ts.guard(tag):
            q, k, v, do = ts._rand_qkv(2, Sq, Sk, h, h_k, 64,
                                       torch.bfloat16, want_dout=True,
                                       seed=17)
            out, lse = itf.flash_attn_flex_flash_bh(q, k, v, mask_bh)
            ref = bh_ref(q, k, v, mask_bh, do)
            mx, _, rel = ts.err_metrics(out, ref['out'])
            ts.check(f"{tag} fwd mx={mx:.4f} rel={rel:.4f}",
                     mx < atol and rel < ts.REL_TOL)
            mxl, _, _ = ts.err_metrics(lse, ref['lse'])
            ts.check(f"{tag} lse mx={mxl:.4f}", mxl < atol)
            grads = itf.flash_attn_flex_flash_bh_bwd(
                q, k, v, out, lse, do, mask_bh)
            for gname, g in zip(('dq', 'dk', 'dv'), grads):
                mx, _, rel = ts.err_metrics(g, ref[gname])
                ts.check(f"{tag} {gname} mx={mx:.4f} rel={rel:.4f}",
                         mx < atol and rel < ts.REL_TOL)

    with ts.guard("bhmask degeneration (slim)"):
        q, k, v, do = ts._rand_qkv(2, Sq, Sk, 4, 2, 64, torch.bfloat16,
                                   want_dout=True, seed=19)
        m = ts.make_causal_mask(Sq, Sk, device=DEV)
        mask_bh = m.unsqueeze(0).unsqueeze(0).expand(2, 4, Sq, Sk)
        out_bh, lse_bh = itf.flash_attn_flex_flash_bh(q, k, v, mask_bh)
        out1, lse1 = itf.flash_attn_flex_flash(q, k, v, m)
        ts.check("single-group fwd bitwise equal",
                 torch.equal(out_bh, out1) and torch.equal(lse_bh, lse1))
        g_bh = itf.flash_attn_flex_flash_bh_bwd(q, k, v, out_bh, lse_bh,
                                               do, mask_bh)
        g1 = itf.flash_attn_flex_flash_bwd(q, k, v, out1, lse1, do, m)
        ts.check("single-group bwd bitwise equal",
                 all(torch.equal(a, b) for a, b in zip(g_bh, g1)))

    print("SLIM BHMASK RESULT: passed=%d failed=%d"
          % (ts._passed, ts._failed))
    return 1 if ts._failed else 0


if __name__ == "__main__":
    sys.exit(main())
