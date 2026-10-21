# Holmes FA2 raw ABI regression tests

These tests use Torch tensors and the PPU SDK, with no Holmes frontend, compiler
or runtime. Run from a **new isolated checkout** of the FA source under test,
with its pinned `csrc/actlize` submodule initialized. Do not reuse a shared build
or install over an existing package.

```bash
source /usr/local/PPU_SDK/envsetup.sh
python -m venv --system-site-packages /absolute/new/venv
source /absolute/new/venv/bin/activate
python -m pip install 'setuptools==75.8.0' 'packaging==24.2' psutil ninja wheel pytest
# Select an available device after checking ppu-smi.
export CUDA_VISIBLE_DEVICES=1 HGGC_VISIBLE_DEVICES=1
MAX_JOBS=8 bash tests/holmes/run.sh /absolute/new/results
```

`run.sh` builds a Holmes-only wheel, extracts its sole shared library into the
result directory, and loads that exact path. It does not install the wheel.
The result directory must not already exist. Build failures, missing devices,
missing symbols, skipped tests and missing/incomplete JUnit reports fail the run.
It saves build/pytest/pipeline diagnostics, JUnit, source and dependency commits,
source hashes, environment details, ELF symbols/dependencies and wheel/SO hashes.

`FA_CPP_INFER_BUILD=1` and `FA_CPP_INFER_BUILD_FOR_HOLMES=1` enable Holmes mode.
The additional `FA_HOLMES_ABI_TEST=1` enables a host-only dispatch observation
hook. It reports the actual `run_mha_fwd` split count, kernel swap flag and dense GQA reshape flag; tests
assert single-split, split-K and GQA dispatch, then verify numerical output.
Dense GQA currently reshapes Q/O but does not set the special varlen kernel
swap flag; the tests observe these separately rather than assuming they match.
The hook is absent when that test flag is unset. The tested wheel is an
**instrumented test artifact**, not a release wheel. The hook observes host
dispatch during graph capture; it is not a device-side replay counter.

The fixture compiles `raw_bridge.cpp` into a pytest temporary directory using
`CXX` (default `c++`) and Torch's C++ ABI setting. The bridge catches C++
exceptions from both exported C++ entry points; exceptions never unwind through
ctypes. It is not a Python binding and is never added to the wheel. The exact
mangled symbols intentionally make an ABI change fail loudly.

Coverage includes FP16/BF16 CPU FP32 reference comparisons, bottom-right causal
masking, padded positive strides, empty Q/K, unchanged queried capacity,
external streams, no Torch device allocations after warmup, graph capture and
three replays with updated Q/K/V, invalid layouts/dtypes and invalid workspace.
Capture begins on the external stream while the raw call runs with a different
current stream; ignoring the passed stream cannot produce the required replay.
Workspace guards on both sides and output padding check writes outside the
supplied ranges. A smaller workspace may legally select fewer splits: rejection
is required below the queried **single-split minimum**, not below the heuristic
budget. This is why the tests do not demand that `query(shape)-1` always fail.

Detection limits: guards detect writes into the checked regions, not arbitrary
out-of-range reads/writes, pointer allocation sizes, lifetime bugs beyond these
replays, or concurrent use of one arena. The ABI receives bare tensor pointers
without allocation capacities; callers must ensure their storage/lifetimes.
The tests do not intentionally submit undersized tensor allocations to a device
kernel. They do not validate full Holmes execution, performance, or every head
dimension. Replay correctness checks do not independently identify device kernel
names. Hardware coverage is the hardware on which the run actually executes.

CI adds a `holmes` variant to the existing PPU scheduler matrix on v2.8.2 and
retains normal FA2/FA3 variants. FA PRs check out and verify the exact PR head;
ACTLIZE source mode verifies the dependency PR head. The parent PR workflow has
no paths filter, so source, build, workflow and submodule changes all trigger it.
NAS evidence is uploaded even on failure; logs/JUnit/hashes and the wheel are
retained, while intermediate compiler objects are not uploaded.

Scheduler tar extraction may preserve another uid. `common.sh` adds only this
checkout and its ACTLIZE directory to process-scoped `safe.directory` settings,
preserving existing process configuration and leaving global Git config alone.
The FA3-specific runner and tests are documented in `tests/holmes_fa3/README.md`.

Library acceptance is loading and running in the current Torch process after
SDK initialization. The fixture imports Torch, loads the freshly extracted SO
with `ctypes.CDLL(..., mode=os.RTLD_NOW)`, resolves the raw ABI symbols, and runs
the numerical/stream/graph tests. Missing libraries or unresolved symbols fail
fixture setup and CI. Standalone loading without Torch, `ldd` resolution, and
relocation to an arbitrary machine are outside this suite's acceptance contract.
