"""Inspect setup output for five build modes; this does not compile kernels."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

if len(sys.argv) == 1:
    for holmes, hllm in [("0", "0"), ("0", "1"), ("1", None), ("1", "0"), ("1", "1")]:
        env = dict(os.environ, FA3_HOLMES_WHEEL=holmes)
        if hllm is None:
            env.pop("FA3_HLLM_BUILD", None)
        else:
            env["FA3_HLLM_BUILD"] = hllm
        subprocess.run([sys.executable, __file__, "child"], env=env, check=True)
    print("BUILD_MODE_MATRIX_PASS")
else:
    import setuptools
    import torch.utils.cpp_extension as extension

    captured = {}

    def capture_extension(**kwargs):
        captured["extension"] = kwargs
        return SimpleNamespace(**kwargs)

    def capture_setup(**kwargs):
        captured["setup"] = kwargs

    extension.CUDAExtension = capture_extension
    setuptools.setup = capture_setup
    root = Path(__file__).resolve().parents[2] / "hopper"
    os.chdir(root)
    result = runpy.run_path(str(root / "setup.py"))
    holmes = os.environ["FA3_HOLMES_WHEEL"] == "1"
    hllm = holmes or os.getenv("FA3_HLLM_BUILD") == "1"
    cfg = captured["setup"]
    assert result["FA3_HLLM_BUILD"] == hllm
    assert cfg["name"] == ("flash_attn_3_holmes" if holmes else "flash_attn_3")
    assert cfg["py_modules"] == ([] if holmes else ["flash_attn_interface"])
    ext = captured["extension"]
    assert ext["name"] == cfg["name"] + "._C"
    for flags in ext["extra_compile_args"].values():
        assert flags.count("-DFA3_HLLM_BUILD") == int(hllm)
        assert flags.count("-DFA3_HOLMES_BUILD") == int(holmes)
    print(
        json.dumps(
            {
                "holmes": holmes,
                "hllm_env": os.getenv("FA3_HLLM_BUILD"),
                "effective_hllm": hllm,
                "package": cfg["name"],
            }
        )
    )
