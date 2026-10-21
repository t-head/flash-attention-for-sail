#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:?usage: run.sh NEW_RESULT_DIRECTORY}
mkdir "$OUT"
OUT=$(cd "$OUT" && pwd)
exec > >(tee "$OUT/pipeline.log") 2>&1
export OUT
trap 'rc=$?; echo "EXIT_CODE=$rc"; exit "$rc"' EXIT
cd "$ROOT"
source tests/holmes/common.sh
holmes_trust_checkout "$ROOT"
if [[ -n "${EXPECTED_FA_SHA:-}" ]]; then test "$(git rev-parse HEAD)" = "$EXPECTED_FA_SHA"; fi
if [[ -n "${EXPECTED_ACTLIZE_SHA:-}" ]]; then test "$(git -C csrc/actlize rev-parse HEAD)" = "$EXPECTED_ACTLIZE_SHA"; fi
{
  git rev-parse HEAD
  git status --short
  git diff --binary
  git -C csrc/actlize remote -v
  git -C csrc/actlize rev-parse HEAD
  python -m pip freeze
  python -c 'import torch,platform; print(platform.platform(),torch.__version__); print(torch.__config__.show())'
  hgcc --version
  echo "PPU_SDK=${PPU_SDK:-unknown} IMAGE=${SAIL_IMAGE:-unspecified} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} HGGC_VISIBLE_DEVICES=${HGGC_VISIBLE_DEVICES:-unset}"
  ppu-smi
} > "$OUT/environment.log" 2>&1
sha256sum hopper/{setup.py,flash_api.cpp} tests/holmes_fa3/*.{py,sh} tests/holmes/{common.sh,raw_bridge.cpp} > "$OUT/source.sha256"
cp tests/holmes_fa3/check_junit.py "$OUT/check_junit.py"
python tests/holmes_fa3/check_build_modes.py 2>&1 | tee "$OUT/build-modes.log"
cd "$ROOT/hopper"
FA3_HOLMES_WHEEL=1 \
FA3_HLLM_BUILD=1 \
FLASH_ATTENTION_DISABLE_SM90=TRUE \
FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
PYTORCH_SAIL_ARCH="${PYTORCH_SAIL_ARCH:-ppu_10;ppu_15}" \
MAX_JOBS=${MAX_JOBS:-8} \
python setup.py bdist_wheel --dist-dir "$OUT/dist" --bdist-dir "$OUT/bdist" 2>&1 | tee "$OUT/build.log"
cd "$ROOT"
python - <<'PY'
import hashlib,os,pathlib,zipfile
out=pathlib.Path(os.environ['OUT'])
wheels=list((out/'dist').glob('*.whl'))
if len(wheels)!=1: raise RuntimeError(wheels)
with zipfile.ZipFile(wheels[0]) as z:
    names=z.namelist()
    if any('flash_attn_interface' in n or 'bridge' in n for n in names): raise RuntimeError(names)
    sos=[n for n in names if n.endswith('.so')]
    if len(sos)!=1 or not sos[0].startswith('flash_attn_3_holmes/_C'): raise RuntimeError(sos)
    z.extractall(out/'wheel')
lib=out/'wheel'/sos[0]
(out/'library.path').write_text(str(lib))
with (out/'artifact.sha256').open('w') as f:
    for p in [wheels[0],lib]: f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p)+'\n')
PY
export HOLMES_FA3_LIB_PATH
HOLMES_FA3_LIB_PATH=$(cat "$OUT/library.path")
nm -D --defined-only "$HOLMES_FA3_LIB_PATH" | c++filt > "$OUT/symbols.log"
grep -q -- '-DFA3_HOLMES_BUILD' "$OUT/build.log"
grep -q -- '-DFA3_HLLM_BUILD' "$OUT/build.log"
FA_MAX_WORKSPACE_SIZE=67108864 python -m pytest tests/holmes_fa3 -v -s --junitxml="$OUT/junit.xml" 2>&1 | tee "$OUT/pytest.log"
python tests/holmes_fa3/check_junit.py "$OUT/junit.xml"
