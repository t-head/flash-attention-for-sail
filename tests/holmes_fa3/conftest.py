import ctypes as C
import os
from pathlib import Path
import shlex
import subprocess

import pytest
import torch


@pytest.fixture(scope="session")
def abi(tmp_path_factory):
    assert torch.cuda.is_available(), "FA3 Holmes ABI tests require a PPU device"
    # FA3's default query is a conservative budget. Keep this smoke suite's
    # budget explicit; it is passed unchanged to the raw entry point.
    assert os.environ["FA_MAX_WORKSPACE_SIZE"] == "67108864"
    path = Path(os.environ["HOLMES_FA3_LIB_PATH"]).resolve(strict=True)
    # Acceptance is loading and running inside the current Torch process.
    # Resolve symbols immediately; missing dependencies fail fixture setup.
    lib = C.CDLL(str(path), mode=os.RTLD_NOW)
    target = tmp_path_factory.mktemp("fa3-bridge") / "bridge.so"
    subprocess.run(
        shlex.split(os.environ.get("CXX", "c++"))
        + [
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            str(Path(__file__).parents[1] / "holmes/raw_bridge.cpp"),
            "-o",
            str(target),
        ],
        check=True,
    )
    bridge = C.CDLL(str(target))
    forward = getattr(
        lib,
        "_ZN10holmes_fa311mha_fwd_rawEPvS0_RKSt6vectorIlSaIlEES5_S0_S5_S5_S0_S5_S5_S0_S5_S5_fbS0_mi",
    )
    query = getattr(lib, "_ZN8hllm_fa318get_workspace_sizeEiiiiiiibib")
    bridge.call_raw.restype = C.c_char_p
    bridge.call_raw.argtypes = (
        [C.c_void_p] * 6
        + [C.POINTER(C.c_int64)] * 2
        + [C.c_float, C.c_bool, C.c_void_p, C.c_uint64, C.c_int32]
    )
    bridge.query_raw.restype = C.c_char_p
    bridge.query_raw.argtypes = [
        C.c_void_p,
        C.POINTER(C.c_int),
        C.c_int,
        C.c_bool,
        C.POINTER(C.c_uint64),
    ]

    class ABI:
        def query(self, dims):
            size = C.c_uint64()
            err = bridge.query_raw(
                C.cast(query, C.c_void_p), (C.c_int * 7)(*dims), 0, False, C.byref(size)
            )
            if err:
                raise RuntimeError(err.decode())
            return size.value

        def call(self, tensors, workspace, size, stream, dtype, causal):
            err = bridge.call_raw(
                C.cast(forward, C.c_void_p),
                stream,
                *[x.data_ptr() for x in tensors],
                (C.c_int64 * 16)(*[n for x in tensors for n in x.shape]),
                (C.c_int64 * 16)(*[n for x in tensors for n in x.stride()]),
                tensors[0].shape[-1] ** -0.5,
                causal,
                workspace,
                size,
                dtype,
            )
            if err:
                raise RuntimeError(err.decode())

    print("FA3_RAW_ABI", path, torch.__version__, torch.cuda.get_device_name(0), flush=True)
    return ABI()
