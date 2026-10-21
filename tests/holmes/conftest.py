import ctypes as C
import os
from pathlib import Path
import shlex
import subprocess

import pytest
import torch


@pytest.fixture(scope="session")
def abi(tmp_path_factory):
    # Missing hardware/artifacts are errors, never an all-skipped green suite.
    assert torch.cuda.is_available(), "Holmes raw ABI tests require a PPU/CUDA device"
    before = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
    probe = torch.empty(1024, device="cuda")
    assert (
        torch.cuda.memory_stats()["allocation.all.allocated"] > before
    ), "Allocator counter must observe device allocations"
    del probe
    path = Path(os.environ["HOLMES_FA2_LIB_PATH"]).resolve(strict=True)
    # Acceptance is loading and running inside the current Torch process.
    # Resolve symbols immediately; missing dependencies fail fixture setup.
    lib = C.CDLL(str(path), mode=os.RTLD_NOW)
    build = tmp_path_factory.mktemp("raw-bridge") / "bridge.so"
    subprocess.run(
        shlex.split(os.environ.get("CXX", "c++"))
        + [
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            str(Path(__file__).with_name("raw_bridge.cpp")),
            "-o",
            str(build),
        ],
        check=True,
    )
    bridge = C.CDLL(str(build))
    forward = getattr(
        lib,
        "_ZN10holmes_mha11mha_fwd_rawEPvS0_RKSt6vectorIlSaIlEES5_S0_S5_S5_S0_S5_S5_S0_S5_S5_fbS0_mi",
    )
    query = getattr(lib, "_ZN10holmes_mha18get_workspace_sizeEiiiiiiibib")
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
    dispatch = lib.holmes_raw_test_dispatch
    dispatch.argtypes = [C.POINTER(C.c_int)] * 3
    dispatch.restype = None

    class ABI:
        def query(self, dims, splits=0, tune=False):
            result = C.c_uint64()
            err = bridge.query_raw(
                C.cast(query, C.c_void_p), (C.c_int * 7)(*dims), splits, tune, C.byref(result)
            )
            if err:
                raise RuntimeError(err.decode())
            return result.value

        def call(
            self,
            tensors,
            workspace,
            capacity,
            stream,
            dtype,
            causal=False,
            shapes=None,
            strides=None,
            pointers=None,
        ):
            shapes = shapes or [n for x in tensors for n in x.shape]
            strides = strides or [n for x in tensors for n in x.stride()]
            pointers = pointers or [x.data_ptr() for x in tensors]
            return bridge.call_raw(
                C.cast(forward, C.c_void_p),
                stream,
                *pointers,
                (C.c_int64 * 16)(*shapes),
                (C.c_int64 * 16)(*strides),
                tensors[0].shape[-1] ** -0.5,
                causal,
                workspace,
                capacity,
                dtype,
            )

        def dispatch(self):
            splits, swapped, dense_swapped = C.c_int(), C.c_int(), C.c_int()
            dispatch(C.byref(splits), C.byref(swapped), C.byref(dense_swapped))
            return splits.value, swapped.value, dense_swapped.value

    print("RAW_ABI", path, torch.__version__, torch.cuda.get_device_name(0), flush=True)
    return ABI()
