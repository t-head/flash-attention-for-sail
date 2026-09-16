"""Standalone P3 (per-(b,h) heterogeneous mask) guard runner.

Runs only suite_bhmask from the unified suite — used by the background
chain right after the wheel rebuild, so the new heterogeneous-layout path
(rebase_slices + build_bh_layout + API guards) gets GPU end-to-end
verification without rerunning the full suite.
"""
import sys

import test_arb_suite as ts


def main():
    itf = ts._load_kernel()
    if itf is None:
        print("kernel extension unavailable")
        return 2
    ts.suite_bhmask(itf)
    print("BHMASK RESULT: passed=%d failed=%d" % (ts._passed, ts._failed))
    return 1 if ts._failed else 0


if __name__ == "__main__":
    sys.exit(main())
