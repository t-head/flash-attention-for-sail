#!/usr/bin/env bash
# Run from an isolated checkout; does not install into shared site-packages.
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
if [[ -n "${EXPECTED_FA_SHA:-}" ]]; then
  test "$(git rev-parse HEAD)" = "$EXPECTED_FA_SHA"
fi
if [[ -n "${EXPECTED_ACTLIZE_SHA:-}" ]]; then
  test "$(git -C csrc/actlize rev-parse HEAD)" = "$EXPECTED_ACTLIZE_SHA"
fi
{
  git rev-parse HEAD
  git status --short
  git diff --binary
  git -C csrc/actlize remote -v
  git -C csrc/actlize rev-parse HEAD
  git submodule status
  python -m pip freeze
  python -c 'import torch,platform; print(platform.platform(),torch.__version__,torch.version.cuda,torch.version.git_version); print(torch.__config__.show())'
  command -v hgcc
  hgcc --version
  echo "PPU_SDK=${PPU_SDK:-unknown} IMAGE=${SAIL_IMAGE:-unspecified} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} HGGC_VISIBLE_DEVICES=${HGGC_VISIBLE_DEVICES:-unset}"
  ppu-smi
} > "$OUT/environment.log" 2>&1
sha256sum setup.py csrc/flash_attn/{flash_api.cpp,holmes_api.h,holmes_workspace.h,holmes_splitkv.h} tests/holmes/*.{py,cpp,sh} > "$OUT/source.sha256"
cp tests/holmes/check_junit.py "$OUT/check_junit.py"
# The checkout must be isolated; a dedicated dist directory prevents stale wheels.
MAX_JOBS=${MAX_JOBS:-8} \
PYTORCH_SAIL_ARCH="${PYTORCH_SAIL_ARCH:-ppu_10;ppu_15}" \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FA_CPP_INFER_BUILD=1 \
FA_CPP_INFER_BUILD_FOR_HOLMES=1 \
FA_HOLMES_ABI_TEST=1 \
python setup.py bdist_wheel --dist-dir "$OUT/dist" --bdist-dir "$OUT/bdist" 2>&1 | tee "$OUT/build.log"
python - <<'PY'
import hashlib,os,pathlib,zipfile
out=pathlib.Path(os.environ['OUT'])
wheels=list((out/'dist').glob('*.whl'))
assert len(wheels)==1,wheels
with zipfile.ZipFile(wheels[0]) as z:
    names=z.namelist()
    assert not any('bridge' in x or 'tests/holmes' in x for x in names),names
    sos=[x for x in names if x.endswith('.so')]
    assert len(sos)==1 and pathlib.Path(sos[0]).name.startswith('flashattn2_holmes'),sos
    z.extractall(out/'wheel')
lib=out/'wheel'/sos[0]
(out/'library.path').write_text(str(lib))
with (out/'artifact.sha256').open('w') as f:
    for p in [wheels[0],lib]:
        f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p)+'\n')
PY
export HOLMES_FA2_LIB_PATH
HOLMES_FA2_LIB_PATH=$(cat "$OUT/library.path")
nm -D --defined-only "$HOLMES_FA2_LIB_PATH" | c++filt > "$OUT/symbols.log"
python -m pytest tests/holmes -v -s --junitxml="$OUT/junit.xml" 2>&1 | tee "$OUT/pytest.log"
python tests/holmes/check_junit.py "$OUT/junit.xml"
