"""Standalone varlen (P1) guard runner.

Runs only suite_varlen from the unified suite — used by the background
chain after the route-2 regression + perf baseline finish, so the new
varlen entry points get GPU end-to-end verification without rerunning
the full suite.
"""
import sys

import test_arb_suite as ts


def main():
    itf = ts._load_kernel()
    if itf is None:
        print("kernel extension unavailable")
        return 2
    ts.suite_varlen(itf)
    print("VARLEN RESULT: passed=%d failed=%d" % (ts._passed, ts._failed))
    return 1 if ts._failed else 0


if __name__ == "__main__":
    sys.exit(main())
