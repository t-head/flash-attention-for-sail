#!/bin/bash
set -ex

# Build PyTorch from source with flash-attention integrated.
# Expected environment:
#   FA_SRC        - flash-attention-for-sail repository root (default: /workspace/source)
#   PYTORCH_SRC   - PyTorch source checkout (default: /workspace/deps/pytorch)
#   python_version - e.g. py312 (default: py312)
#   product       - ppu / ppu1.5 (default: ppu)

FA_SRC=${FA_SRC:-/workspace/source}
PYTORCH_SRC=${PYTORCH_SRC:-/workspace/deps/pytorch}

python_version=${python_version:-py312}
product=${product:-ppu}
cpu=${cpu:-$(nproc)}

echo "========================================"
echo "FA_SRC:        $FA_SRC"
echo "PYTORCH_SRC:   $PYTORCH_SRC"
echo "python_version: $python_version"
echo "product:       $product"
echo "cpu:           $cpu"
echo "========================================"

echo "========= Setup PPU SDK ====="
source /usr/local/PPU_SDK/envsetup.sh
source "${PCCL_HOME:-/usr/local/pccl}/envsetup.sh"
export PCCL_INCLUDE_DIR="${PCCL_HOME}/include"
export PCCL_LIB_DIR="${PCCL_HOME}/lib"
export USE_SAIL=1
unset TORCH_CUDA_ARCH_LIST

case "${product}" in
    ppu) export PYTORCH_SAIL_ARCH=ppu_10 ;;
    ppu1.5) export PYTORCH_SAIL_ARCH=ppu_15 ;;
    *) printf 'Unsupported product for SAIL: %s\n' "${product}" >&2; exit 1 ;;
esac

echo "========= Build PyTorch ====="
cd "${PYTORCH_SRC}"

export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
if [[ ${python_version} != "py38" ]]; then
    pip install numpy==2.0
fi

echo "Build for PYTORCH_SAIL_ARCH=${PYTORCH_SAIL_ARCH}"
pip install -r requirements.txt

export PYTORCH_VERSION=2.10.0
export PYTORCH_BUILD_VERSION=2.10.0
export PYTORCH_BUILD_NUMBER=0
export MAX_JOBS=${cpu}

USE_FLASH_ATTENTION=True \
USE_MEM_EFF_ATTENTION=True \
USE_PCCL=True \
USE_DISTRIBUTED=True \
BUILD_CAFFE2=False \
BUILD_TEST=True \
CUDA_VERBOSE_BUILD=True \
CMAKE_VERBOSE_MAKEFILE=True \
python3 setup.py bdist_wheel

shopt -s nullglob
wheels=("${PYTORCH_SRC}/dist"/torch-*.whl)
if (( ${#wheels[@]} == 0 )); then
    printf 'No torch wheel produced in %s/dist\n' "${PYTORCH_SRC}" >&2
    exit 1
fi
python3 -m pip install "${wheels[@]}"

echo "========================================"
echo "========= PyTorch Build Success ========"
