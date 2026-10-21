# Holmes FA3 CI

This suite covers the dedicated wheel mode introduced by PR #13. It uses Torch
and the PPU SDK without importing Holmes frontend/compiler/runtime.

From a new isolated checkout with the pinned ACTLIZE submodule initialized:

```bash
source /usr/local/PPU_SDK/envsetup.sh
# Activate an isolated venv containing pytest, wheel, ninja, packaging and psutil.
# Select an available PPU after checking ppu-smi.
export CUDA_VISIBLE_DEVICES=1 HGGC_VISIBLE_DEVICES=1
MAX_JOBS=8 bash tests/holmes_fa3/run.sh /absolute/new/result-directory
```

The runner first inspects five Holmes/HLLM configurations: `(0,0)`, `(0,1)`,
`(1,unset)`, `(1,0)`, `(1,1)`. These are setup configuration checks, not five
full compilations. It then builds with `FA3_HOLMES_WHEEL=1`, `FA3_HLLM_BUILD=1`,
backward and SM90 disabled. It verifies the emitted macros, sole Holmes SO,
Python-interface/bridge exclusion, dependencies and artifact hashes. The wheel
is extracted into the new result directory, without installing a shared package.

Eight FP16/BF16 cases directly load `holmes_fa3::mha_fwd_raw` and
`hllm_fa3::get_workspace_size`, compare contiguous/transposed strides against a CPU FP32 reference, exercise
external streams, and capture/replay after Q/K/V updates. The temporary C++
bridge is built from `tests/holmes/raw_bridge.cpp` and is excluded from wheels.
The FA3 workspace query is conservative: this suite explicitly sets
`FA_MAX_WORKSPACE_SIZE=67108864` before loading/querying and passes that exact
budget to forward. Workspace guards check writes outside this supplied budget.

FA3 has a different workspace and validation implementation from FA2. This
suite does not assert FA2-specific invalid-workspace, empty-layout or allocator
contracts for FA3. It does not add FP8 raw coverage: the current PR #13 raw dtype
switch accepts FP16/BF16 only. QSA namespace compilation is covered by the
Holmes build; QSA-specific runtime execution and full Holmes E2E are not tested.

Both runners trust only the current checkout and its ACTLIZE directory through
process-scoped Git configuration. This supports scheduler tar extraction with
a different uid without modifying global Git configuration. JUnit checking
requires all eight cases to pass, with no skips. Failures retain diagnostics
and fail CI. Ordinary FA2/FA3 variants remain separate.

Library acceptance is loading and running in the current Torch process after
SDK initialization. The fixture imports Torch, loads the freshly extracted SO
with `ctypes.CDLL(..., mode=os.RTLD_NOW)`, resolves the raw ABI symbols, and runs
the numerical/stream/graph tests. Missing libraries or unresolved symbols fail
fixture setup and CI. Standalone loading without Torch, `ldd` resolution, and
relocation to an arbitrary machine are outside this suite's acceptance contract.
