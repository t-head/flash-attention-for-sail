#!/bin/bash
set -ex

# Run PyTorch integration tests for flash-attention v2.7.2
# Expected environment:
#   FA_SRC        - flash-attention-for-sail repository root (default: /workspace/source)
#   PYTORCH_SRC   - PyTorch source checkout (default: /workspace/deps/pytorch)

FA_SRC=${FA_SRC:-/workspace/source}
PYTORCH_SRC=${PYTORCH_SRC:-/workspace/deps/pytorch}

echo "========= FA2 PyTorch Test CI Start ========="
cat /etc/os-release || true
echo "Working directory: $(pwd)"
echo "FA_SRC:      $FA_SRC"
echo "PYTORCH_SRC: $PYTORCH_SRC"

# 1. Environment setup
source /usr/local/PPU_SDK/envsetup.sh
pip uninstall torch -y || true
pip install "${PYTORCH_SRC}/dist"/*.whl
pip install numpy==1.26.4 junit-xml 'pytest<9' expecttest

# 2. Run only the two ACTest PyTorch cases outside the source root.
TEST_ROOT=$(mktemp -d /tmp/fa-pytorch-tests.XXXXXX)
mkdir -p "${TEST_ROOT}/test/distributed/fsdp"
cp "${PYTORCH_SRC}/test/test_native_mha.py" "${TEST_ROOT}/test/"
cp "${PYTORCH_SRC}/test/distributed/fsdp/test_fsdp_flatten_params.py" \
  "${TEST_ROOT}/test/distributed/fsdp/"
cd "${TEST_ROOT}"
export PYTHONPATH="${PYTORCH_SRC}/test:${PYTHONPATH}"

python "${FA_SRC}/.github/workflows/scripts/pr-run-pytorch-ut.py" \
  --caselist "${FA_SRC}/.github/workflows/scripts/pytorch-casecmd.json" \
  --suite pytorch \
  --output "${FA_SRC}/test-results-pytorch.xml"

echo "========= FA2 PyTorch Test CI Done ========="
